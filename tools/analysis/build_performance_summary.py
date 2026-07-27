#!/usr/bin/env python3
"""Build validated Performance Resource Monitor artifacts from a DROID-Splat log."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


MARKER = "[PerformanceResourceMonitor]"
SCHEMA_VERSION = 1

REQUIRED_COMMON_FIELDS = {
    "schema",
    "event_id",
    "timestamp_utc",
    "perf_counter_ns",
    "process_role",
    "pid",
    "cuda_device",
    "event_type",
    "stage",
    "phase",
    "status",
    "reason",
}

REQUIRED_MAPPER_FIELDS = {
    "mapper_update_id",
    "mapping_iter",
    "new_camera_count",
    "gaussian_count",
    "cpu_wall_ms",
    "gpu_elapsed_ms",
    "memory_allocated_bytes",
    "memory_reserved_bytes",
    "max_memory_allocated_bytes",
    "max_memory_reserved_bytes",
    "resource_admission_mode",
    "lifecycle_observer_enabled",
}


class MonitorParseError(ValueError):
    """Raised when a monitor log violates the fail-closed input contract."""


def _validate_event(event: Any, line_number: int, occurrence: int) -> dict[str, Any]:
    location = f"line {line_number}, marker {occurrence}"
    if not isinstance(event, dict):
        raise MonitorParseError(f"{location}: event must be a JSON object.")

    missing = REQUIRED_COMMON_FIELDS.difference(event)
    if missing:
        raise MonitorParseError(
            f"{location}: missing common fields: {sorted(missing)}."
        )
    if event["schema"] != SCHEMA_VERSION:
        raise MonitorParseError(
            f"{location}: unsupported schema {event['schema']!r}; "
            f"expected {SCHEMA_VERSION}."
        )
    if not isinstance(event["event_id"], (str, int)):
        raise MonitorParseError(f"{location}: event_id must be a string or integer.")
    if not isinstance(event["perf_counter_ns"], int):
        raise MonitorParseError(f"{location}: perf_counter_ns must be an integer.")
    if not isinstance(event["pid"], int):
        raise MonitorParseError(f"{location}: pid must be an integer.")
    if event["phase"] not in {"begin", "end", "summary"}:
        raise MonitorParseError(
            f"{location}: invalid phase {event['phase']!r}."
        )
    if event["status"] not in {"ok", "skipped", "error"}:
        raise MonitorParseError(
            f"{location}: invalid status {event['status']!r}."
        )
    if event["status"] == "skipped" and (
        not isinstance(event["reason"], str) or not event["reason"].strip()
    ):
        raise MonitorParseError(
            f"{location}: skipped events require a non-empty reason."
        )
    if (
        event["event_type"] in {"stage_timing", "process_lifecycle"}
        and event["phase"] == "begin"
        and event["status"] != "ok"
    ):
        raise MonitorParseError(
            f"{location}: begin events must use status='ok', "
            f"got {event['status']!r}."
        )

    if event["process_role"] == "mapper":
        missing_mapper = REQUIRED_MAPPER_FIELDS.difference(event)
        if missing_mapper:
            raise MonitorParseError(
                f"{location}: missing mapper fields: {sorted(missing_mapper)}."
            )

    return event


def parse_monitor_events(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Parse all marker-delimited JSON objects, including TQDM-prefixed events."""

    decoder = json.JSONDecoder()
    events: list[dict[str, Any]] = []
    seen_event_ids: dict[str, int] = {}

    for line_number, line in enumerate(lines, start=1):
        search_from = 0
        occurrence = 0
        while True:
            marker_index = line.find(MARKER, search_from)
            if marker_index < 0:
                break

            occurrence += 1
            json_start = line.find("{", marker_index + len(MARKER))
            if json_start < 0:
                raise MonitorParseError(
                    f"line {line_number}, marker {occurrence}: "
                    "no JSON object follows the marker."
                )

            try:
                event, consumed = decoder.raw_decode(line[json_start:])
            except json.JSONDecodeError as exc:
                raise MonitorParseError(
                    f"line {line_number}, marker {occurrence}: "
                    f"invalid JSON: {exc.msg}."
                ) from exc

            event = _validate_event(event, line_number, occurrence)
            event_id = str(event["event_id"])
            if event_id in seen_event_ids:
                raise MonitorParseError(
                    f"line {line_number}, marker {occurrence}: duplicate event_id "
                    f"{event_id!r}; first seen on line {seen_event_ids[event_id]}."
                )
            seen_event_ids[event_id] = line_number
            events.append(event)
            search_from = json_start + consumed

    if not events:
        raise MonitorParseError(f"No {MARKER} events were found.")

    validate_monitor_events(events)
    return events


def _pair_key(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event["process_role"],
        event["pid"],
        event["stage"],
        event.get("mapper_update_id"),
        event.get("mapping_iter"),
    )


def _format_process_key(key: tuple[str, int]) -> dict[str, Any]:
    process_role, pid = key
    return {
        "process_role": process_role,
        "pid": pid,
    }


def _validate_stage_pairs(events: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    error_stages = []
    for event in events:
        if event["event_type"] != "stage_timing":
            continue
        if event["status"] == "error":
            error_stages.append(
                {
                    "event_id": str(event["event_id"]),
                    "process_role": event["process_role"],
                    "pid": event["pid"],
                    "stage": event["stage"],
                    "mapper_update_id": event.get("mapper_update_id"),
                    "mapping_iter": event.get("mapping_iter"),
                    "phase": event["phase"],
                    "reason": event["reason"],
                }
            )
        if event["phase"] == "summary":
            continue
        pairs[_pair_key(event)][event["phase"]] += 1

    violations = []
    for key, counts in sorted(pairs.items(), key=lambda item: repr(item[0])):
        if counts["begin"] != 1 or counts["end"] != 1:
            violations.append(
                f"{key!r}: begin={counts['begin']}, end={counts['end']}"
            )
    if violations:
        raise MonitorParseError(
            "Missing or duplicate stage begin/end events: " + "; ".join(violations)
        )
    if error_stages:
        raise MonitorParseError(
            "Stage error event(s) make the performance run invalid: "
            + json.dumps(error_stages, ensure_ascii=False, sort_keys=True)
        )

    return {
        "stage_pairs_complete": True,
        "stage_pair_count": len(pairs),
        "stage_pair_violations": 0,
        "error_stages": [],
    }


def _validate_process_lifecycles(
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    process_keys = {
        (str(event["process_role"]), int(event["pid"]))
        for event in events
    }
    lifecycle_events: dict[
        tuple[str, int], dict[str, list[dict[str, Any]]]
    ] = {
        key: {"begin": [], "end": []}
        for key in process_keys
    }
    invalid_phases = []

    for event in events:
        if event["event_type"] != "process_lifecycle":
            continue
        key = (str(event["process_role"]), int(event["pid"]))
        if event["phase"] not in {"begin", "end"}:
            invalid_phases.append(
                {
                    **_format_process_key(key),
                    "event_id": str(event["event_id"]),
                    "phase": event["phase"],
                }
            )
            continue
        lifecycle_events[key][event["phase"]].append(event)

    incomplete_processes = []
    duplicate_lifecycle_events = []
    lifecycle_order_violations = []
    error_processes = []

    for key in sorted(process_keys):
        begin_events = lifecycle_events[key]["begin"]
        end_events = lifecycle_events[key]["end"]
        process = _format_process_key(key)

        if len(begin_events) != 1 or len(end_events) != 1:
            incomplete_processes.append(
                {
                    **process,
                    "begin_count": len(begin_events),
                    "end_count": len(end_events),
                }
            )
        if len(begin_events) > 1:
            duplicate_lifecycle_events.append(
                {
                    **process,
                    "phase": "begin",
                    "count": len(begin_events),
                }
            )
        if len(end_events) > 1:
            duplicate_lifecycle_events.append(
                {
                    **process,
                    "phase": "end",
                    "count": len(end_events),
                }
            )

        if len(begin_events) == 1 and len(end_events) == 1:
            begin_ns = begin_events[0]["perf_counter_ns"]
            end_ns = end_events[0]["perf_counter_ns"]
            if end_ns <= begin_ns:
                lifecycle_order_violations.append(
                    {
                        **process,
                        "begin_perf_counter_ns": begin_ns,
                        "end_perf_counter_ns": end_ns,
                    }
                )

        process_error_events = [
            event
            for event in begin_events + end_events
            if event["status"] == "error"
        ]
        if process_error_events:
            error_processes.append(
                {
                    **process,
                    "event_ids": [
                        str(event["event_id"])
                        for event in process_error_events
                    ],
                }
            )

    if (
        invalid_phases
        or incomplete_processes
        or duplicate_lifecycle_events
        or lifecycle_order_violations
        or error_processes
    ):
        details = {
            "invalid_lifecycle_phases": invalid_phases,
            "incomplete_processes": incomplete_processes,
            "duplicate_lifecycle_events": duplicate_lifecycle_events,
            "lifecycle_order_violations": lifecycle_order_violations,
            "error_processes": error_processes,
        }
        raise MonitorParseError(
            "Invalid process lifecycle: "
            + json.dumps(details, ensure_ascii=False, sort_keys=True)
        )

    return {
        "process_lifecycle_complete": True,
        "process_count": len(process_keys),
        "incomplete_processes": [],
        "duplicate_lifecycle_events": [],
        "lifecycle_order_violations": [],
        "error_processes": [],
    }


def validate_monitor_events(
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    stage_validation = _validate_stage_pairs(events)
    lifecycle_validation = _validate_process_lifecycles(events)
    other_error_events = [
        {
            "event_id": str(event["event_id"]),
            "event_type": event["event_type"],
            "process_role": event["process_role"],
            "pid": event["pid"],
            "stage": event["stage"],
            "phase": event["phase"],
            "reason": event["reason"],
        }
        for event in events
        if event["status"] == "error"
        and event["event_type"] not in {"stage_timing", "process_lifecycle"}
    ]
    if other_error_events:
        raise MonitorParseError(
            "Error event(s) make the performance run invalid: "
            + json.dumps(
                other_error_events,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return {
        "validation_pass": True,
        "error_events": [],
        **stage_validation,
        **lifecycle_validation,
    }


def _nullable_int_sort_key(value: Any) -> tuple[int, int]:
    if value is None:
        return (0, -1)
    return (1, int(value))


def sort_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        events,
        key=lambda event: (
            str(event["process_role"]),
            str(event["stage"]),
            _nullable_int_sort_key(event.get("mapper_update_id")),
            _nullable_int_sort_key(event.get("mapping_iter")),
            int(event["perf_counter_ns"]),
            str(event["event_id"]),
        ),
    )


def _numeric_values(
    events: Iterable[dict[str, Any]], field: str
) -> list[float]:
    values = []
    for event in events:
        value = event.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def build_summary(events: list[dict[str, Any]], source: Path) -> dict[str, Any]:
    validation = validate_monitor_events(events)
    role_counts = Counter(str(event["process_role"]) for event in events)
    stage_counts = Counter(str(event["stage"]) for event in events)
    stage_end_events: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for event in events:
        is_completed_stage = (
            event["event_type"] == "stage_timing"
            and event["phase"] == "end"
        )
        is_process_summary = (
            event["event_type"] == "process_summary"
            and event["phase"] == "summary"
        )
        if event["status"] == "ok" and (
            is_completed_stage or is_process_summary
        ):
            stage_end_events[str(event["stage"])].append(event)

    stage_metrics: dict[str, dict[str, Any]] = {}
    for stage, matching_events in sorted(stage_end_events.items()):
        cpu_values = _numeric_values(matching_events, "cpu_wall_ms")
        gpu_values = _numeric_values(matching_events, "gpu_elapsed_ms")
        allocated_values = _numeric_values(
            matching_events, "memory_allocated_bytes"
        )
        reserved_values = _numeric_values(
            matching_events, "memory_reserved_bytes"
        )
        max_allocated_values = _numeric_values(
            matching_events, "max_memory_allocated_bytes"
        )
        max_reserved_values = _numeric_values(
            matching_events, "max_memory_reserved_bytes"
        )

        stage_metrics[stage] = {
            "event_count": len(matching_events),
            "cpu_wall_total_ms": sum(cpu_values) if cpu_values else None,
            "cpu_wall_mean_ms": fmean(cpu_values) if cpu_values else None,
            "gpu_elapsed_total_ms": sum(gpu_values) if gpu_values else None,
            "gpu_elapsed_mean_ms": fmean(gpu_values) if gpu_values else None,
            "memory_allocated_peak_bytes": (
                int(max(allocated_values)) if allocated_values else None
            ),
            "memory_reserved_peak_bytes": (
                int(max(reserved_values)) if reserved_values else None
            ),
            "max_memory_allocated_peak_bytes": (
                int(max(max_allocated_values)) if max_allocated_values else None
            ),
            "max_memory_reserved_peak_bytes": (
                int(max(max_reserved_values)) if max_reserved_values else None
            ),
        }

    mapper_update_ids = sorted(
        {
            int(event["mapper_update_id"])
            for event in events
            if event["process_role"] == "mapper"
            and event.get("mapper_update_id") is not None
        }
    )
    skipped_events = [
        {
            "process_role": event["process_role"],
            "stage": event["stage"],
            "mapper_update_id": event.get("mapper_update_id"),
            "reason": event["reason"],
        }
        for event in events
        if event["status"] == "skipped"
    ]

    return {
        "schema": SCHEMA_VERSION,
        "source_log": str(source.resolve()),
        "event_count": len(events),
        "process_role_counts": dict(sorted(role_counts.items())),
        "stage_counts": dict(sorted(stage_counts.items())),
        "mapper_update_ids": mapper_update_ids,
        "skipped_events": skipped_events,
        "stage_metrics": stage_metrics,
        "validation": {
            **validation,
            "duplicate_event_ids": 0,
            "jsonl_sort_order": [
                "process_role",
                "stage",
                "mapper_update_id",
                "mapping_iter",
                "perf_counter_ns",
                "event_id",
            ],
        },
    }


def _check_output_paths(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing output(s): {formatted}. "
            "Pass --overwrite only after explicit review."
        )


def _validate_output_targets(
    run_log: Path,
    output_jsonl: Path,
    summary_json: Path,
) -> None:
    source = run_log.resolve()
    output = output_jsonl.resolve()
    summary = summary_json.resolve()
    if source in {output, summary}:
        raise MonitorParseError("Output paths must not overwrite the input run log.")
    if output == summary:
        raise MonitorParseError(
            "JSONL and summary outputs must use different paths."
        )


def write_outputs(
    events: list[dict[str, Any]],
    summary: dict[str, Any],
    output_jsonl: Path,
    summary_json: Path,
    overwrite: bool = False,
) -> None:
    _check_output_paths((output_jsonl, summary_json), overwrite=overwrite)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    sorted_events = sort_events(events)
    with output_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for event in sorted_events:
            handle.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )

    with summary_json.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse [PerformanceResourceMonitor] JSON events from a DROID-Splat "
            "run log and build validated JSONL and summary JSON outputs."
        )
    )
    parser.add_argument("run_log", type=Path, help="Input run.log path.")
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        help="Output JSONL path (default: <run_log_dir>/resource_monitor.jsonl).",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Output summary path (default: <run_log_dir>/performance_summary.json).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output files after explicit review.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    run_log = args.run_log
    if not run_log.is_file():
        print(f"ERROR: input log is not a readable file: {run_log}", file=sys.stderr)
        return 2

    output_jsonl = args.output_jsonl or run_log.parent / "resource_monitor.jsonl"
    summary_json = args.summary_json or run_log.parent / "performance_summary.json"

    try:
        _validate_output_targets(run_log, output_jsonl, summary_json)
        with run_log.open("r", encoding="utf-8", errors="replace") as handle:
            events = parse_monitor_events(handle)
        summary = build_summary(events, source=run_log)
        write_outputs(
            events,
            summary,
            output_jsonl=output_jsonl,
            summary_json=summary_json,
            overwrite=args.overwrite,
        )
    except (MonitorParseError, FileExistsError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"events={len(events)}")
    print(f"resource_monitor_jsonl={output_jsonl.resolve()}")
    print(f"performance_summary_json={summary_json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
