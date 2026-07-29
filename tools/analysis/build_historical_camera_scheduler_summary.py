#!/usr/bin/env python3
"""Extract, validate, and summarize HistoricalCameraScheduler events."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


PREFIX = "[HistoricalCameraScheduler]"
SCHEMA_VERSION = 1
SUPPORTED_MODES = frozenset(
    {
        "baseline_random",
        "deterministic_stratified",
    }
)
REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "event_type",
        "event_id",
        "process_role",
        "mode",
        "status",
        "reason",
        "mapper_update_id",
        "mapping_iteration",
        "call_key",
        "scheduler_call_count",
        "scheduler_call_count_after",
        "active_selection_tick",
        "active_selection_count_after",
        "selection_active",
        "history_count",
        "new_camera_count",
        "n_last_frames",
        "n_rand_frames",
        "history_budget",
        "effective_history_budget",
        "preserve_all_history_until",
        "eligible_old_history_count",
        "eligible_old_history_uids",
        "protected_recent_count",
        "protected_recent_uids",
        "selected_old_history_count",
        "selected_old_history_uids",
        "selected_history_count",
        "selected_history_uids",
        "protected_new_count",
        "protected_new_uids",
        "final_selected_count",
        "final_selected_uids",
        "logging_sample_every",
        "strategy",
        "base_positions",
        "rotated_positions",
    }
)
LIST_FIELDS = (
    "eligible_old_history_uids",
    "protected_recent_uids",
    "selected_old_history_uids",
    "selected_history_uids",
    "protected_new_uids",
    "final_selected_uids",
)
COUNT_LIST_PAIRS = (
    ("eligible_old_history_count", "eligible_old_history_uids"),
    ("protected_recent_count", "protected_recent_uids"),
    ("selected_old_history_count", "selected_old_history_uids"),
    ("selected_history_count", "selected_history_uids"),
    ("protected_new_count", "protected_new_uids"),
    ("final_selected_count", "final_selected_uids"),
)


class ValidationError(RuntimeError):
    pass


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(
    event: dict[str, Any],
    field: str,
    *,
    minimum: Optional[int] = None,
) -> int:
    value = event.get(field)
    if not _is_int(value):
        raise ValidationError(
            f"{event.get('event_id', '<unknown>')}: {field} must be int, "
            f"got {value!r}."
        )
    if minimum is not None and value < minimum:
        raise ValidationError(
            f"{event.get('event_id', '<unknown>')}: {field} must be >= "
            f"{minimum}, got {value}."
        )
    return value


def _require_uid_list(event: dict[str, Any], field: str) -> list[int]:
    value = event.get(field)
    if not isinstance(value, list):
        raise ValidationError(
            f"{event.get('event_id', '<unknown>')}: {field} must be a list."
        )
    if any(not _is_int(uid) or uid < 0 for uid in value):
        raise ValidationError(
            f"{event.get('event_id', '<unknown>')}: {field} must contain "
            "non-negative integer uids."
        )
    if len(value) != len(set(value)):
        raise ValidationError(
            f"{event.get('event_id', '<unknown>')}: {field} contains duplicates."
        )
    return value


def extract_events_from_lines(lines: Iterable[str]) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        cursor = 0
        while True:
            marker = line.find(PREFIX, cursor)
            if marker < 0:
                break
            brace = line.find("{", marker + len(PREFIX))
            if brace < 0:
                raise ValidationError(
                    f"Line {line_number}: marker has no following JSON object."
                )
            try:
                event, consumed = decoder.raw_decode(line[brace:])
            except json.JSONDecodeError as error:
                raise ValidationError(
                    f"Line {line_number}: invalid scheduler JSON: {error}."
                ) from error
            if not isinstance(event, dict):
                raise ValidationError(
                    f"Line {line_number}: scheduler payload must be an object."
                )
            events.append(event)
            cursor = brace + consumed
    if not events:
        raise ValidationError("No HistoricalCameraScheduler events were found.")
    return events


def extract_events(run_log: Path) -> list[dict[str, Any]]:
    if not run_log.is_file():
        raise ValidationError(f"Input run log does not exist: {run_log}.")
    with run_log.open("r", encoding="utf-8", errors="replace") as handle:
        return extract_events_from_lines(handle)


def _validate_common_fields(event: dict[str, Any]) -> None:
    event_id = event.get("event_id", "<unknown>")
    fields = set(event)
    missing = sorted(REQUIRED_FIELDS - fields)
    extra = sorted(fields - REQUIRED_FIELDS)
    if missing or extra:
        raise ValidationError(
            f"{event_id}: field mismatch; missing={missing}, extra={extra}."
        )
    if not _is_int(event["schema"]) or event["schema"] != SCHEMA_VERSION:
        raise ValidationError(
            f"{event_id}: schema must be {SCHEMA_VERSION}, "
            f"got {event['schema']!r}."
        )
    if event["event_type"] != "selection":
        raise ValidationError(f"{event_id}: unsupported event_type.")
    if not isinstance(event_id, str) or not event_id:
        raise ValidationError("event_id must be a non-empty string.")
    if event["process_role"] != "mapper":
        raise ValidationError(f"{event_id}: process_role must be 'mapper'.")
    if event["mode"] not in SUPPORTED_MODES:
        raise ValidationError(f"{event_id}: unsupported mode {event['mode']!r}.")
    if event["status"] != "ok":
        raise ValidationError(f"{event_id}: status must be 'ok'.")
    if event["reason"] not in {
        "partial_budget",
        "preserve_all_or_full_budget",
    }:
        raise ValidationError(f"{event_id}: unsupported reason.")
    if not isinstance(event["selection_active"], bool):
        raise ValidationError(f"{event_id}: selection_active must be bool.")

    non_negative = (
        "mapper_update_id",
        "mapping_iteration",
        "scheduler_call_count",
        "scheduler_call_count_after",
        "active_selection_count_after",
        "history_count",
        "new_camera_count",
        "n_last_frames",
        "n_rand_frames",
        "history_budget",
        "effective_history_budget",
        "preserve_all_history_until",
        "eligible_old_history_count",
        "protected_recent_count",
        "selected_old_history_count",
        "selected_history_count",
        "protected_new_count",
        "final_selected_count",
    )
    for field in non_negative:
        _require_int(event, field, minimum=0)
    _require_int(event, "logging_sample_every", minimum=1)

    if event["active_selection_tick"] is not None:
        _require_int(event, "active_selection_tick", minimum=0)
    if not isinstance(event["call_key"], list) or len(event["call_key"]) != 2:
        raise ValidationError(f"{event_id}: call_key must be a two-item list.")
    if any(not _is_int(value) or value < 0 for value in event["call_key"]):
        raise ValidationError(
            f"{event_id}: call_key must contain non-negative integers."
        )
    if event["call_key"] != [
        event["mapper_update_id"],
        event["mapping_iteration"],
    ]:
        raise ValidationError(f"{event_id}: call_key does not match its fields.")
    if event_id != f"hcs:{event['scheduler_call_count']}":
        raise ValidationError(f"{event_id}: event_id does not match call count.")
    if event["scheduler_call_count_after"] != event["scheduler_call_count"] + 1:
        raise ValidationError(f"{event_id}: scheduler call count is inconsistent.")
    if event["scheduler_call_count"] % event["logging_sample_every"] != 0:
        raise ValidationError(
            f"{event_id}: event violates logging_sample_every."
        )

    for field in LIST_FIELDS:
        _require_uid_list(event, field)
    for count_field, list_field in COUNT_LIST_PAIRS:
        if event[count_field] != len(event[list_field]):
            raise ValidationError(
                f"{event_id}: {count_field} does not match {list_field}."
            )

    if event["history_count"] != (
        event["eligible_old_history_count"]
        + event["protected_recent_count"]
    ):
        raise ValidationError(f"{event_id}: historical partition is incomplete.")
    if event["new_camera_count"] != event["protected_new_count"]:
        raise ValidationError(f"{event_id}: new camera count is inconsistent.")
    if event["selected_history_uids"] != (
        event["protected_recent_uids"]
        + event["selected_old_history_uids"]
    ):
        raise ValidationError(f"{event_id}: selected history order is invalid.")
    if event["final_selected_uids"] != (
        event["selected_history_uids"]
        + event["protected_new_uids"]
    ):
        raise ValidationError(f"{event_id}: final frame order is invalid.")

    groups = (
        set(event["eligible_old_history_uids"]),
        set(event["protected_recent_uids"]),
        set(event["protected_new_uids"]),
    )
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValidationError(f"{event_id}: camera partitions overlap.")
    if not set(event["selected_old_history_uids"]).issubset(groups[0]):
        raise ValidationError(
            f"{event_id}: selected old history is outside the eligible pool."
        )

    active = event["selection_active"]
    if active:
        if event["active_selection_tick"] is None:
            raise ValidationError(f"{event_id}: active event has no active tick.")
        if event["reason"] != "partial_budget":
            raise ValidationError(f"{event_id}: active reason is invalid.")
        if (
            event["active_selection_count_after"]
            != event["active_selection_tick"] + 1
        ):
            raise ValidationError(f"{event_id}: active count is inconsistent.")
    else:
        if event["active_selection_tick"] is not None:
            raise ValidationError(f"{event_id}: inactive event has an active tick.")
        if event["reason"] != "preserve_all_or_full_budget":
            raise ValidationError(f"{event_id}: inactive reason is invalid.")


def _validate_deterministic_event(event: dict[str, Any]) -> None:
    event_id = event["event_id"]
    if event["strategy"] != "deterministic_stratified_v1":
        raise ValidationError(f"{event_id}: deterministic strategy is invalid.")
    eligible = event["eligible_old_history_uids"]
    selected = event["selected_old_history_uids"]
    if eligible != sorted(eligible):
        raise ValidationError(f"{event_id}: eligible uids must be sorted.")
    if selected != sorted(selected):
        raise ValidationError(f"{event_id}: selected old uids must be sorted.")
    for field in ("base_positions", "rotated_positions"):
        positions = event[field]
        if positions is not None and (
            not isinstance(positions, list)
            or any(
                not _is_int(position) or position < 0
                for position in positions
            )
            or len(positions) != len(set(positions))
        ):
            raise ValidationError(
                f"{event_id}: {field} must be null or unique non-negative "
                "integer positions."
            )

    if event["history_count"] <= event["preserve_all_history_until"]:
        if eligible or selected:
            raise ValidationError(
                f"{event_id}: preserve-all event must not expose an old pool."
            )
        if event["selected_history_uids"] != event["protected_recent_uids"]:
            raise ValidationError(
                f"{event_id}: preserve-all history is inconsistent."
            )
        if event["effective_history_budget"] != 0:
            raise ValidationError(
                f"{event_id}: preserve-all effective budget must be zero."
            )
        if event["base_positions"] is not None or event["rotated_positions"] is not None:
            raise ValidationError(
                f"{event_id}: preserve-all event must not have positions."
            )
        return

    pool_size = len(eligible)
    effective = min(event["history_budget"], pool_size)
    if event["effective_history_budget"] != effective:
        raise ValidationError(f"{event_id}: effective budget is incorrect.")
    if len(selected) != effective:
        raise ValidationError(f"{event_id}: selected count violates budget.")
    expected_active = 0 < effective < pool_size
    if event["selection_active"] != expected_active:
        raise ValidationError(f"{event_id}: active flag is incorrect.")

    if effective == 0:
        if event["base_positions"] is not None or event["rotated_positions"] is not None:
            raise ValidationError(f"{event_id}: zero budget must have null positions.")
        expected_selected: list[int] = []
    elif effective == pool_size:
        positions = list(range(pool_size))
        if event["base_positions"] != positions:
            raise ValidationError(f"{event_id}: full-budget base positions differ.")
        if event["rotated_positions"] != positions:
            raise ValidationError(f"{event_id}: full-budget positions differ.")
        expected_selected = eligible
    else:
        tick = event["active_selection_tick"]
        base_positions = [
            (j * pool_size) // effective
            for j in range(effective)
        ]
        rotated_positions = [
            (position + tick) % pool_size
            for position in base_positions
        ]
        if event["base_positions"] != base_positions:
            raise ValidationError(f"{event_id}: base positions fail replay.")
        if event["rotated_positions"] != rotated_positions:
            raise ValidationError(f"{event_id}: rotated positions fail replay.")
        expected_selected = sorted(eligible[position] for position in rotated_positions)
    if selected != expected_selected:
        raise ValidationError(f"{event_id}: selected uids fail deterministic replay.")


def _validate_baseline_event(event: dict[str, Any]) -> None:
    event_id = event["event_id"]
    if event["strategy"] != "original_select_keyframes":
        raise ValidationError(f"{event_id}: baseline strategy is invalid.")
    if event["base_positions"] is not None or event["rotated_positions"] is not None:
        raise ValidationError(f"{event_id}: baseline event must not have positions.")

    threshold = event["n_last_frames"] + event["n_rand_frames"]
    if event["history_count"] <= threshold:
        if event["eligible_old_history_uids"]:
            raise ValidationError(f"{event_id}: baseline preserve-all pool is invalid.")
        if event["selected_history_count"] != event["history_count"]:
            raise ValidationError(f"{event_id}: baseline did not preserve all history.")
        if event["effective_history_budget"] != 0:
            raise ValidationError(f"{event_id}: baseline preserve budget must be zero.")
    else:
        pool_size = event["eligible_old_history_count"]
        effective = min(event["n_rand_frames"], pool_size)
        if event["effective_history_budget"] != effective:
            raise ValidationError(f"{event_id}: baseline budget is inconsistent.")
        if event["selected_old_history_count"] != effective:
            raise ValidationError(f"{event_id}: baseline old selection count differs.")
        if event["selection_active"] != (0 < effective < pool_size):
            raise ValidationError(f"{event_id}: baseline active flag differs.")


def validate_events(events: Sequence[dict[str, Any]]) -> None:
    seen_event_ids: set[str] = set()
    seen_call_keys: set[tuple[int, int]] = set()
    previous_call_key: Optional[tuple[int, int]] = None
    previous_call_count: Optional[int] = None
    previous_active_count: Optional[int] = None
    immutable_config: Optional[tuple[Any, ...]] = None
    for event in events:
        _validate_common_fields(event)
        event_id = event["event_id"]
        if event_id in seen_event_ids:
            raise ValidationError(f"Duplicate event_id: {event_id}.")
        seen_event_ids.add(event_id)

        call_key = tuple(event["call_key"])
        if call_key in seen_call_keys:
            raise ValidationError(f"Duplicate call_key: {call_key}.")
        seen_call_keys.add(call_key)
        if previous_call_key is not None and call_key <= previous_call_key:
            raise ValidationError(
                f"Non-monotonic call_key: {previous_call_key} then {call_key}."
            )
        previous_call_key = call_key

        call_count = event["scheduler_call_count"]
        if previous_call_count is not None and call_count <= previous_call_count:
            raise ValidationError("scheduler_call_count must be strictly increasing.")
        previous_call_count = call_count
        active_count = event["active_selection_count_after"]
        if (
            previous_active_count is not None
            and active_count < previous_active_count
        ):
            raise ValidationError(
                "active_selection_count_after must be non-decreasing."
            )
        previous_active_count = active_count

        event_config = (
            event["mode"],
            event["n_last_frames"],
            event["n_rand_frames"],
            event["history_budget"],
            event["preserve_all_history_until"],
            event["logging_sample_every"],
        )
        if immutable_config is None:
            immutable_config = event_config
        elif event_config != immutable_config:
            raise ValidationError(
                "Scheduler configuration changed within one parsed run."
            )

        if event["mode"] == "deterministic_stratified":
            _validate_deterministic_event(event)
        else:
            _validate_baseline_event(event)


def build_summary(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    validate_events(events)
    selected_counts: Counter[int] = Counter()
    for event in events:
        selected_counts.update(event["selected_old_history_uids"])
    modes = sorted({event["mode"] for event in events})
    return {
        "schema": SCHEMA_VERSION,
        "validation_pass": True,
        "diagnostic_only": True,
        "event_count": len(events),
        "modes": modes,
        "sample_every_values": sorted(
            {event["logging_sample_every"] for event in events}
        ),
        "first_call_key": list(events[0]["call_key"]),
        "last_call_key": list(events[-1]["call_key"]),
        "first_scheduler_call_count": events[0]["scheduler_call_count"],
        "last_scheduler_call_count": events[-1]["scheduler_call_count"],
        "active_selection_event_count": sum(
            int(event["selection_active"]) for event in events
        ),
        "preserve_or_full_event_count": sum(
            int(not event["selection_active"]) for event in events
        ),
        "total_selected_old_history": sum(
            event["selected_old_history_count"] for event in events
        ),
        "total_selected_history": sum(
            event["selected_history_count"] for event in events
        ),
        "total_protected_new": sum(
            event["protected_new_count"] for event in events
        ),
        "old_history_selection_count_by_uid": {
            str(uid): count
            for uid, count in sorted(selected_counts.items())
        },
        "invariant_violations": [],
        "deterministic_replay_pass": True,
    }


def _ensure_output_paths(
    run_log: Path,
    jsonl_path: Path,
    summary_path: Path,
) -> None:
    resolved = [
        run_log.resolve(),
        jsonl_path.resolve(),
        summary_path.resolve(),
    ]
    if len(set(resolved)) != len(resolved):
        raise ValidationError(
            "Input run log, JSONL output, and summary output must be distinct."
        )
    existing = [path for path in (jsonl_path, summary_path) if path.exists()]
    if existing:
        raise ValidationError(
            "Refusing to overwrite existing output files: "
            + ", ".join(str(path) for path in existing)
        )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def write_outputs(
    events: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    *,
    run_log: Path,
    jsonl_path: Path,
    summary_path: Path,
) -> None:
    _ensure_output_paths(run_log, jsonl_path, summary_path)
    jsonl_content = "".join(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        for event in events
    )
    summary_content = json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    ) + "\n"

    _atomic_write_text(jsonl_path, jsonl_content)
    try:
        _atomic_write_text(summary_path, summary_content)
    except Exception:
        jsonl_path.unlink(missing_ok=True)
        raise


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract and fail-closed validate HistoricalCameraScheduler "
            "events from a DROID-Splat run log."
        )
    )
    parser.add_argument("run_log", type=Path, help="DROID-Splat run log")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help=(
            "Output JSONL path (default: "
            "<run_log_dir>/historical_camera_scheduler_events.jsonl)"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help=(
            "Output summary path (default: "
            "<run_log_dir>/historical_camera_scheduler_summary.json)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_log = args.run_log
    jsonl_path = args.jsonl or (
        run_log.parent / "historical_camera_scheduler_events.jsonl"
    )
    summary_path = args.summary or (
        run_log.parent / "historical_camera_scheduler_summary.json"
    )
    try:
        events = extract_events(run_log)
        validate_events(events)
        summary = build_summary(events)
        write_outputs(
            events,
            summary,
            run_log=run_log,
            jsonl_path=jsonl_path,
            summary_path=summary_path,
        )
    except (OSError, ValidationError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(f"events={len(events)}")
    print(f"jsonl={jsonl_path}")
    print(f"summary={summary_path}")
    print("validation_pass=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
