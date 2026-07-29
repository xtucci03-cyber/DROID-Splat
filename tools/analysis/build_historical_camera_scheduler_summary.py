#!/usr/bin/env python3
"""Extract, validate, and summarize HistoricalCameraScheduler events."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


PREFIX = "[HistoricalCameraScheduler]"
SCHEMA_VERSION = 1
DETERMINISTIC_ALGORITHM = "rotating_stratified_v1"
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
        "configured_budget",
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
        "duplicate_uid_count",
        "invariant_failures",
        "exploitation_uids",
        "exploration_uids",
        "starvation_forced_uids",
        "pool_uid_checksum",
        "deterministic_state",
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
    "exploitation_uids",
    "exploration_uids",
    "starvation_forced_uids",
)
COUNT_LIST_PAIRS = (
    ("eligible_old_history_count", "eligible_old_history_uids"),
    ("protected_recent_count", "protected_recent_uids"),
    ("selected_old_history_count", "selected_old_history_uids"),
    ("selected_history_count", "selected_history_uids"),
    ("protected_new_count", "protected_new_uids"),
    ("final_selected_count", "final_selected_uids"),
)
DETERMINISTIC_STATE_FIELDS = frozenset(
    {
        "algorithm",
        "call_key",
        "scheduler_call_count",
        "active_selection_tick",
        "base_positions",
        "rotated_positions",
        "rotation",
    }
)


class ValidationError(RuntimeError):
    pass


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _pool_uid_checksum(uids: Sequence[int]) -> str:
    encoded = json.dumps(
        list(uids),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


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


def _require_string_list(event: dict[str, Any], field: str) -> list[str]:
    value = event.get(field)
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValidationError(
            f"{event.get('event_id', '<unknown>')}: {field} must be a string list."
        )
    return value


def _require_positions(
    event: dict[str, Any],
    field: str,
) -> Optional[list[int]]:
    positions = event.get(field)
    if positions is None:
        return None
    if (
        not isinstance(positions, list)
        or any(not _is_int(position) or position < 0 for position in positions)
        or len(positions) != len(set(positions))
    ):
        raise ValidationError(
            f"{event.get('event_id', '<unknown>')}: {field} must be null or "
            "unique non-negative integer positions."
        )
    return positions


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


def _validate_deterministic_state(event: dict[str, Any]) -> None:
    event_id = event.get("event_id", "<unknown>")
    state = event.get("deterministic_state")
    if not isinstance(state, dict):
        raise ValidationError(f"{event_id}: deterministic_state must be an object.")
    fields = set(state)
    if fields != DETERMINISTIC_STATE_FIELDS:
        raise ValidationError(
            f"{event_id}: deterministic_state field mismatch; "
            f"missing={sorted(DETERMINISTIC_STATE_FIELDS - fields)}, "
            f"extra={sorted(fields - DETERMINISTIC_STATE_FIELDS)}."
        )

    expected_algorithm = (
        DETERMINISTIC_ALGORITHM
        if event["mode"] == "deterministic_stratified"
        else None
    )
    if state["algorithm"] != expected_algorithm:
        raise ValidationError(
            f"{event_id}: deterministic_state.algorithm is invalid."
        )
    if state["call_key"] != event["call_key"]:
        raise ValidationError(f"{event_id}: deterministic_state call_key mismatch.")
    if state["scheduler_call_count"] != event["scheduler_call_count"]:
        raise ValidationError(
            f"{event_id}: deterministic_state scheduler count mismatch."
        )
    if state["active_selection_tick"] != event["active_selection_tick"]:
        raise ValidationError(
            f"{event_id}: deterministic_state active tick mismatch."
        )
    if state["base_positions"] != event["base_positions"]:
        raise ValidationError(
            f"{event_id}: deterministic_state base_positions mismatch."
        )
    if state["rotated_positions"] != event["rotated_positions"]:
        raise ValidationError(
            f"{event_id}: deterministic_state rotated_positions mismatch."
        )
    expected_rotation = (
        event["active_selection_tick"]
        if (
            event["mode"] == "deterministic_stratified"
            and event["selection_active"]
        )
        else None
    )
    if state["rotation"] != expected_rotation:
        raise ValidationError(f"{event_id}: deterministic_state rotation mismatch.")


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
        "configured_budget",
        "effective_history_budget",
        "preserve_all_history_until",
        "eligible_old_history_count",
        "protected_recent_count",
        "selected_old_history_count",
        "selected_history_count",
        "protected_new_count",
        "final_selected_count",
        "duplicate_uid_count",
    )
    for field in non_negative:
        _require_int(event, field, minimum=0)
    _require_int(event, "logging_sample_every", minimum=1)

    if event["configured_budget"] != event["history_budget"]:
        raise ValidationError(f"{event_id}: configured_budget mismatch.")
    if event["duplicate_uid_count"] != 0:
        raise ValidationError(f"{event_id}: duplicate_uid_count must be zero.")
    if _require_string_list(event, "invariant_failures"):
        raise ValidationError(f"{event_id}: invariant_failures must be empty.")

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

    if event["exploitation_uids"]:
        raise ValidationError(f"{event_id}: HCS-v0 exploitation_uids must be empty.")
    if event["starvation_forced_uids"]:
        raise ValidationError(
            f"{event_id}: HCS-v0 starvation_forced_uids must be empty."
        )
    if event["mode"] == "deterministic_stratified":
        if event["exploration_uids"] != event["selected_old_history_uids"]:
            raise ValidationError(f"{event_id}: exploration_uids mismatch.")
    elif event["exploration_uids"]:
        raise ValidationError(
            f"{event_id}: baseline_random exploration_uids must be empty."
        )

    if event["pool_uid_checksum"] != _pool_uid_checksum(
        event["eligible_old_history_uids"]
    ):
        raise ValidationError(f"{event_id}: pool_uid_checksum mismatch.")

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

    _require_positions(event, "base_positions")
    _require_positions(event, "rotated_positions")
    _validate_deterministic_state(event)

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
    if event["strategy"] != DETERMINISTIC_ALGORITHM:
        raise ValidationError(f"{event_id}: deterministic strategy is invalid.")
    eligible = event["eligible_old_history_uids"]
    selected = event["selected_old_history_uids"]
    if eligible != sorted(eligible):
        raise ValidationError(f"{event_id}: eligible uids must be sorted.")
    if selected != sorted(selected):
        raise ValidationError(f"{event_id}: selected old uids must be sorted.")

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
            event["configured_budget"],
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


def extract_mapping_activity_camera_uids(
    activity_events_path: Path,
) -> dict[tuple[int, int], list[int]]:
    if not activity_events_path.is_file():
        raise ValidationError(
            f"Mapping Activity events file does not exist: {activity_events_path}."
        )
    mapping: dict[tuple[int, int], list[int]] = {}
    with activity_events_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValidationError(
                    f"Activity line {line_number}: invalid JSON: {error}."
                ) from error
            if not isinstance(event, dict):
                raise ValidationError(
                    f"Activity line {line_number}: payload must be an object."
                )
            if event.get("event_type") != "iteration_activity":
                continue
            update_id = event.get("update_id")
            iteration_id = event.get("iteration_id")
            camera_uids = event.get("camera_uids")
            if (
                not _is_int(update_id)
                or update_id < 0
                or not _is_int(iteration_id)
                or iteration_id < 0
                or not isinstance(camera_uids, list)
                or any(not _is_int(uid) or uid < 0 for uid in camera_uids)
            ):
                raise ValidationError(
                    f"Activity line {line_number}: invalid update, iteration, "
                    "or camera_uids."
                )
            key = (update_id, iteration_id)
            if key in mapping:
                raise ValidationError(
                    f"Activity line {line_number}: duplicate activity key {key}."
                )
            mapping[key] = list(camera_uids)
    if not mapping:
        raise ValidationError("No Mapping Activity iteration_activity events found.")
    return mapping


def _build_activity_cross_validation(
    events: Sequence[dict[str, Any]],
    activity_events_path: Optional[Path],
) -> dict[str, Any]:
    if activity_events_path is None:
        return {
            "performed": False,
            "activity_events_path": None,
            "matched_iteration_count": 0,
            "exact_match_count": 0,
            "exact_match_ratio": None,
            "mismatch_details": [],
            "missing_activity_call_keys": [],
        }

    activity_by_key = extract_mapping_activity_camera_uids(activity_events_path)
    mismatches: list[dict[str, Any]] = []
    missing: list[list[int]] = []
    exact_match_count = 0
    for event in events:
        key = tuple(event["call_key"])
        expected = event["final_selected_uids"]
        observed = activity_by_key.get(key)
        if observed is None:
            missing.append(list(key))
            continue
        if observed == expected:
            exact_match_count += 1
        else:
            mismatches.append(
                {
                    "call_key": list(key),
                    "scheduler_final_selected_uids": expected,
                    "activity_camera_uids": observed,
                }
            )

    matched_count = len(events) - len(missing)
    result = {
        "performed": True,
        "activity_events_path": str(activity_events_path),
        "matched_iteration_count": matched_count,
        "exact_match_count": exact_match_count,
        "exact_match_ratio": (
            exact_match_count / matched_count
            if matched_count
            else None
        ),
        "mismatch_details": mismatches,
        "missing_activity_call_keys": missing,
    }
    if missing or mismatches or matched_count == 0:
        raise ValidationError(
            "Mapping Activity camera_uids cross validation failed: "
            f"missing={missing}, mismatches={len(mismatches)}."
        )
    return result


def _budget_category(event: dict[str, Any]) -> str:
    if event["mode"] == "baseline_random":
        if event["selection_active"]:
            return "baseline_random_active"
        return "baseline_random_preserve_or_full"
    if event["history_count"] <= event["preserve_all_history_until"]:
        return "preserve_all"
    if event["effective_history_budget"] == 0:
        return "zero_budget"
    if event["effective_history_budget"] == event["eligible_old_history_count"]:
        return "full_budget"
    if event["selection_active"]:
        return "partial_budget"
    return "inactive_other"


def _mean(values: Sequence[int]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _selection_statistics(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    eligible_counts: Counter[int] = Counter()
    selected_counts: Counter[int] = Counter()
    first_selected_call: dict[int, int] = {}
    last_selected_call: dict[int, int] = {}
    current_unselected_streak: dict[int, int] = {}
    longest_unselected_interval: dict[int, int] = {}

    for event in events:
        call_count = event["scheduler_call_count"]
        eligible = set(event["eligible_old_history_uids"])
        selected = set(event["selected_old_history_uids"])
        eligible_counts.update(eligible)
        selected_counts.update(selected)
        for uid in selected:
            first_selected_call.setdefault(uid, call_count)
            last_selected_call[uid] = call_count

        if not event["selection_active"]:
            continue

        for uid in list(current_unselected_streak):
            if uid not in eligible:
                current_unselected_streak[uid] = 0
        for uid in eligible:
            longest_unselected_interval.setdefault(uid, 0)
            if uid in selected:
                longest_unselected_interval[uid] = max(
                    longest_unselected_interval[uid],
                    current_unselected_streak.get(uid, 0),
                )
                current_unselected_streak[uid] = 0
            else:
                current_unselected_streak[uid] = (
                    current_unselected_streak.get(uid, 0) + 1
                )
                longest_unselected_interval[uid] = max(
                    longest_unselected_interval[uid],
                    current_unselected_streak[uid],
                )

    return {
        "eligible_count_by_uid": {
            str(uid): count for uid, count in sorted(eligible_counts.items())
        },
        "selected_old_history_count_by_uid": {
            str(uid): count for uid, count in sorted(selected_counts.items())
        },
        "selection_frequency_by_uid": {
            str(uid): (
                selected_counts[uid] / eligible_counts[uid]
                if eligible_counts[uid]
                else None
            )
            for uid in sorted(eligible_counts)
        },
        "first_selected_scheduler_call_by_uid": {
            str(uid): value for uid, value in sorted(first_selected_call.items())
        },
        "last_selected_scheduler_call_by_uid": {
            str(uid): value for uid, value in sorted(last_selected_call.items())
        },
        "longest_unselected_active_interval_by_uid": {
            str(uid): value
            for uid, value in sorted(longest_unselected_interval.items())
        },
        "max_longest_unselected_active_interval": (
            max(longest_unselected_interval.values())
            if longest_unselected_interval
            else 0
        ),
    }


def build_summary(
    events: Sequence[dict[str, Any]],
    *,
    mapping_activity_events_path: Optional[Path] = None,
) -> dict[str, Any]:
    validate_events(events)
    modes = sorted({event["mode"] for event in events})
    effective_budgets = [event["effective_history_budget"] for event in events]
    budget_categories = Counter(_budget_category(event) for event in events)
    checksum_values = sorted({event["pool_uid_checksum"] for event in events})
    activity_cross_validation = _build_activity_cross_validation(
        events,
        mapping_activity_events_path,
    )
    selection_stats = _selection_statistics(events)
    sample_every_values = sorted(
        {event["logging_sample_every"] for event in events}
    )
    return {
        "schema": SCHEMA_VERSION,
        "validation_pass": True,
        "diagnostic_only": True,
        "event_count": len(events),
        "modes": modes,
        "algorithm": (
            DETERMINISTIC_ALGORITHM
            if "deterministic_stratified" in modes
            else None
        ),
        "sample_every_values": sample_every_values,
        "selection_gap_scope": (
            "full_event_stream"
            if sample_every_values == [1]
            else "sampled_events_only"
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
        "budget_event_count_by_category": dict(sorted(budget_categories.items())),
        "configured_budget": events[0]["configured_budget"],
        "effective_history_budget": {
            "min": min(effective_budgets),
            "max": max(effective_budgets),
            "mean": _mean(effective_budgets),
            "count_by_value": {
                str(value): count
                for value, count in sorted(Counter(effective_budgets).items())
            },
        },
        "total_selected_old_history": sum(
            event["selected_old_history_count"] for event in events
        ),
        "total_selected_history": sum(
            event["selected_history_count"] for event in events
        ),
        "total_protected_new": sum(
            event["protected_new_count"] for event in events
        ),
        "old_history_selection_count_by_uid": (
            selection_stats["selected_old_history_count_by_uid"]
        ),
        "eligible_count_by_uid": selection_stats["eligible_count_by_uid"],
        "selection_frequency_by_uid": selection_stats["selection_frequency_by_uid"],
        "first_selected_scheduler_call_by_uid": (
            selection_stats["first_selected_scheduler_call_by_uid"]
        ),
        "last_selected_scheduler_call_by_uid": (
            selection_stats["last_selected_scheduler_call_by_uid"]
        ),
        "longest_unselected_active_interval_by_uid": (
            selection_stats["longest_unselected_active_interval_by_uid"]
        ),
        "max_longest_unselected_active_interval": (
            selection_stats["max_longest_unselected_active_interval"]
        ),
        "pool_uid_checksum_unique_count": len(checksum_values),
        "pool_uid_checksum_values": checksum_values,
        "pool_checksum_validation_pass": True,
        "activity_cross_validation": activity_cross_validation,
        "activity_cross_validation_performed": activity_cross_validation["performed"],
        "invariant_violations": [],
        "deterministic_replay_pass": True,
    }


def _format_dict_table(mapping: dict[str, Any], columns: tuple[str, str]) -> str:
    lines = [
        f"| {columns[0]} | {columns[1]} |",
        "|---|---:|",
    ]
    for key, value in mapping.items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def render_markdown_summary(summary: dict[str, Any]) -> str:
    activity = summary["activity_cross_validation"]
    lines = [
        "# Historical Camera Scheduler v0 Evidence Summary",
        "",
        "## 1. 运行与配置",
        "",
        f"- validation_pass: {summary['validation_pass']}",
        f"- diagnostic_only: {summary['diagnostic_only']}",
        f"- modes: {summary['modes']}",
        f"- algorithm: {summary['algorithm']}",
        f"- configured_budget: {summary['configured_budget']}",
        f"- sample_every_values: {summary['sample_every_values']}",
        f"- selection_gap_scope: {summary['selection_gap_scope']}",
        "",
        "## 2. 事件数",
        "",
        f"- event_count: {summary['event_count']}",
        f"- first_call_key: {summary['first_call_key']}",
        f"- last_call_key: {summary['last_call_key']}",
        f"- active_selection_event_count: {summary['active_selection_event_count']}",
        "",
        "## 3. 预算事件数",
        "",
        _format_dict_table(
            summary["budget_event_count_by_category"],
            ("category", "count"),
        ),
        "",
        "## 4. 每UID选择频率",
        "",
        _format_dict_table(
            summary["selection_frequency_by_uid"],
            ("uid", "selected/eligible"),
        ),
        "",
        "## 5. 最长未选择间隔",
        "",
        _format_dict_table(
            summary["longest_unselected_active_interval_by_uid"],
            ("uid", "active iterations"),
        ),
        "",
        "## 6. 确定性重放",
        "",
        f"- deterministic_replay_pass: {summary['deterministic_replay_pass']}",
        "",
        "## 7. checksum验证",
        "",
        f"- pool_checksum_validation_pass: {summary['pool_checksum_validation_pass']}",
        f"- pool_uid_checksum_unique_count: {summary['pool_uid_checksum_unique_count']}",
        "",
        "## 8. UID与预算不变量",
        "",
        f"- invariant_violations: {summary['invariant_violations']}",
        f"- max_longest_unselected_active_interval: "
        f"{summary['max_longest_unselected_active_interval']}",
        "",
        "## 9. Activity交叉验证",
        "",
        f"- performed: {activity['performed']}",
        f"- matched_iteration_count: {activity['matched_iteration_count']}",
        f"- exact_match_count: {activity['exact_match_count']}",
        f"- exact_match_ratio: {activity['exact_match_ratio']}",
        f"- mismatch_details: {activity['mismatch_details']}",
        f"- missing_activity_call_keys: {activity['missing_activity_call_keys']}",
        "",
        "## 10. 诊断性质说明",
        "",
        "- 本summary只用于HCS-v0证据合同审计。",
        "- 若日志采样sample_every不为1，最长未选择间隔只代表采样事件流。",
        "- Activity交叉验证仅比较同一call_key上的camera_uids，不替代数据集质量评价。",
        "",
    ]
    return "\n".join(lines)


def _ensure_output_paths(
    run_log: Path,
    jsonl_path: Path,
    summary_path: Path,
    markdown_path: Path,
    activity_events_path: Optional[Path],
) -> None:
    resolved = [
        run_log.resolve(),
        jsonl_path.resolve(),
        summary_path.resolve(),
        markdown_path.resolve(),
    ]
    if activity_events_path is not None:
        resolved.append(activity_events_path.resolve())
    if len(set(resolved)) != len(resolved):
        raise ValidationError(
            "Input log, optional activity events, JSONL output, summary output, "
            "and Markdown output must be distinct."
        )
    existing = [
        path
        for path in (jsonl_path, summary_path, markdown_path)
        if path.exists()
    ]
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
    markdown_path: Path,
    activity_events_path: Optional[Path] = None,
) -> None:
    _ensure_output_paths(
        run_log,
        jsonl_path,
        summary_path,
        markdown_path,
        activity_events_path,
    )
    jsonl_content = "".join(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        for event in events
    )
    summary_content = json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    ) + "\n"
    markdown_content = render_markdown_summary(summary)

    written: list[Path] = []
    try:
        _atomic_write_text(jsonl_path, jsonl_content)
        written.append(jsonl_path)
        _atomic_write_text(summary_path, summary_content)
        written.append(summary_path)
        _atomic_write_text(markdown_path, markdown_content)
        written.append(markdown_path)
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
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
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help=(
            "Output Chinese Markdown summary path (default: "
            "<run_log_dir>/historical_camera_scheduler_summary_zh.md)"
        ),
    )
    parser.add_argument(
        "--mapping-activity-events",
        type=Path,
        default=None,
        help=(
            "Optional Mapping Activity JSONL file for call_key camera_uids "
            "cross validation."
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
    markdown_path = args.markdown or (
        run_log.parent / "historical_camera_scheduler_summary_zh.md"
    )
    try:
        events = extract_events(run_log)
        validate_events(events)
        summary = build_summary(
            events,
            mapping_activity_events_path=args.mapping_activity_events,
        )
        write_outputs(
            events,
            summary,
            run_log=run_log,
            jsonl_path=jsonl_path,
            summary_path=summary_path,
            markdown_path=markdown_path,
            activity_events_path=args.mapping_activity_events,
        )
    except (OSError, ValidationError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(f"events={len(events)}")
    print(f"jsonl={jsonl_path}")
    print(f"summary={summary_path}")
    print(f"markdown={markdown_path}")
    print("validation_pass=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
