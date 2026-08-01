#!/usr/bin/env python3
"""Extract and validate GaussianCandidateSelectorDryRun v0 events."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Optional


PREFIX = "[GaussianCandidateSelectorDryRun]"
SCHEMA_VERSION = 1
MODE = "dry_run"
ALGORITHM = "occupied_voxel_cap_k_v0"
COVERAGE_METHOD = "voxel_occupancy"
K_VALUES = (1, 2, 4, 8)

REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "event_type",
        "event_id",
        "event_sequence",
        "status",
        "reason",
        "mode",
        "algorithm",
        "mapper_update_id",
        "source_camera_id",
        "init",
        "protected_init",
        "coverage_method",
        "coverage_voxel_size",
        "k_values",
        "candidate_count",
        "candidate_finite_count",
        "candidate_nonfinite_count",
        "existing_gaussian_count",
        "candidate_unique_voxel_count",
        "occupied_candidate_count",
        "novel_candidate_count",
        "occupied_unique_voxel_count",
        "novel_unique_voxel_count",
        "occupied_multiplicity_histogram",
        "occupied_multiplicity_mean",
        "occupied_multiplicity_max",
        "occupied_multiplicity_p50",
        "occupied_multiplicity_p90",
        "occupied_multiplicity_p95",
        "occupied_multiplicity_p99",
        "k_scan",
        "selection_applied",
        "selected_indices",
        "all_candidates_forwarded",
        "gaussian_before",
        "actual_admitted_count",
        "actual_dropped_count",
        "gaussian_after_extend",
        "actual_conservation_pass",
        "empty_candidate",
        "dry_run_wall_ms",
        "error",
    }
)

COUNT_FIELDS = (
    "event_sequence",
    "mapper_update_id",
    "source_camera_id",
    "candidate_count",
    "candidate_finite_count",
    "candidate_nonfinite_count",
    "existing_gaussian_count",
    "candidate_unique_voxel_count",
    "occupied_candidate_count",
    "novel_candidate_count",
    "occupied_unique_voxel_count",
    "novel_unique_voxel_count",
    "occupied_multiplicity_max",
    "gaussian_before",
    "actual_admitted_count",
    "actual_dropped_count",
    "gaussian_after_extend",
)

OUTPUT_FILENAMES = (
    "candidate_selector_dry_run_events.jsonl",
    "candidate_selector_dry_run_summary.json",
    "candidate_selector_dry_run_summary.csv",
    "CANDIDATE_SELECTOR_DRY_RUN_SUMMARY_中文.md",
)


class ValidationError(RuntimeError):
    """Formal evidence is incomplete or violates schema v1."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _assert_no_nonfinite(value: Any, path: str = "event") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationError(f"{path} contains a non-finite number.")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_no_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_nonfinite(item, f"{path}[{index}]")


def _require_count(event: dict[str, Any], field: str) -> int:
    value = event[field]
    if not _is_int(value) or value < 0:
        raise ValidationError(
            f"{event.get('event_id', '<unknown>')}: {field} must be a "
            f"non-negative integer, got {value!r}."
        )
    return value


def _require_bool(event: dict[str, Any], field: str) -> bool:
    value = event[field]
    if not isinstance(value, bool):
        raise ValidationError(
            f"{event.get('event_id', '<unknown>')}: {field} must be bool, "
            f"got {value!r}."
        )
    return value


def _assert_close(
    event_id: str,
    field: str,
    actual: float,
    expected: float,
) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValidationError(
            f"{event_id}: {field}={actual} does not match {expected}."
        )


def extract_events_from_lines(
    lines: Iterable[str],
) -> list[dict[str, Any]]:
    """Recover every marked JSON object, including TQDM-prefixed lines."""

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
                    f"Line {line_number}: invalid selector JSON: {error}."
                ) from error
            if not isinstance(event, dict):
                raise ValidationError(
                    f"Line {line_number}: selector payload must be an object."
                )
            _assert_no_nonfinite(event, f"line[{line_number}]")
            events.append(event)
            cursor = brace + consumed
    if not events:
        raise ValidationError(
            "No GaussianCandidateSelectorDryRun events were found."
        )
    return events


def extract_events(run_log: Path) -> list[dict[str, Any]]:
    if not run_log.is_file():
        raise ValidationError(f"Input run log does not exist: {run_log}.")
    with run_log.open("r", encoding="utf-8", errors="replace") as handle:
        return extract_events_from_lines(handle)


def _validate_histogram(
    event: dict[str, Any],
) -> tuple[list[int], int, int]:
    event_id = event["event_id"]
    histogram = event["occupied_multiplicity_histogram"]
    if not isinstance(histogram, list):
        raise ValidationError(f"{event_id}: histogram must be a list.")
    multiplicities: list[int] = []
    last_multiplicity = 0
    voxel_total = 0
    candidate_total = 0
    for index, bucket in enumerate(histogram):
        if not isinstance(bucket, dict) or set(bucket) != {
            "multiplicity",
            "voxel_count",
        }:
            raise ValidationError(
                f"{event_id}: histogram[{index}] has invalid fields."
            )
        multiplicity = bucket["multiplicity"]
        voxel_count = bucket["voxel_count"]
        if (
            not _is_int(multiplicity)
            or multiplicity <= 0
            or not _is_int(voxel_count)
            or voxel_count <= 0
        ):
            raise ValidationError(
                f"{event_id}: histogram counts must be positive integers."
            )
        if multiplicity <= last_multiplicity:
            raise ValidationError(
                f"{event_id}: histogram multiplicities must be increasing."
            )
        last_multiplicity = multiplicity
        voxel_total += voxel_count
        candidate_total += multiplicity * voxel_count
        multiplicities.extend([multiplicity] * voxel_count)
    return multiplicities, voxel_total, candidate_total


def _nearest_rank(values: list[int], percentile: float) -> Optional[int]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _expected_k_scan(
    event: dict[str, Any],
    multiplicities: list[int],
) -> list[dict[str, Any]]:
    candidate_count = event["candidate_count"]
    results = []
    for k_value in K_VALUES:
        if event["protected_init"]:
            admitted = candidate_count
            dropped = 0
        else:
            occupied_admitted = sum(
                min(multiplicity, k_value)
                for multiplicity in multiplicities
            )
            admitted = (
                event["novel_candidate_count"]
                + event["candidate_nonfinite_count"]
                + occupied_admitted
            )
            dropped = candidate_count - admitted
        results.append(
            {
                "k": k_value,
                "estimated_admitted_count": admitted,
                "estimated_dropped_count": dropped,
                "admitted_ratio": (
                    admitted / candidate_count if candidate_count else None
                ),
                "dropped_ratio": (
                    dropped / candidate_count if candidate_count else None
                ),
            }
        )
    return results


def _validate_k_scan(
    event: dict[str, Any],
    multiplicities: list[int],
) -> None:
    event_id = event["event_id"]
    if event["k_values"] != list(K_VALUES):
        raise ValidationError(
            f"{event_id}: k_values must be {list(K_VALUES)}."
        )
    actual = event["k_scan"]
    expected = _expected_k_scan(event, multiplicities)
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise ValidationError(f"{event_id}: invalid k_scan length.")
    required = {
        "k",
        "estimated_admitted_count",
        "estimated_dropped_count",
        "admitted_ratio",
        "dropped_ratio",
    }
    for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
        if not isinstance(actual_item, dict) or set(actual_item) != required:
            raise ValidationError(
                f"{event_id}: k_scan[{index}] has invalid fields."
            )
        for field in (
            "k",
            "estimated_admitted_count",
            "estimated_dropped_count",
        ):
            if actual_item[field] != expected_item[field]:
                raise ValidationError(
                    f"{event_id}: k_scan[{index}].{field} was not "
                    "recomputed from the multiplicity histogram."
                )
        for field in ("admitted_ratio", "dropped_ratio"):
            actual_ratio = actual_item[field]
            expected_ratio = expected_item[field]
            if expected_ratio is None:
                if actual_ratio is not None:
                    raise ValidationError(
                        f"{event_id}: {field} must be null for no candidates."
                    )
            elif not _is_finite_number(actual_ratio):
                raise ValidationError(
                    f"{event_id}: {field} must be finite."
                )
            else:
                _assert_close(
                    event_id,
                    f"k_scan[{index}].{field}",
                    float(actual_ratio),
                    float(expected_ratio),
                )


def validate_event(event: dict[str, Any]) -> None:
    event_id = event.get("event_id", "<unknown>")
    fields = set(event)
    if fields != REQUIRED_FIELDS:
        raise ValidationError(
            f"{event_id}: field mismatch; "
            f"missing={sorted(REQUIRED_FIELDS - fields)}, "
            f"extra={sorted(fields - REQUIRED_FIELDS)}."
        )
    _assert_no_nonfinite(event, str(event_id))
    if event["schema"] != SCHEMA_VERSION:
        raise ValidationError(f"{event_id}: schema must be {SCHEMA_VERSION}.")
    if event["event_type"] != "candidate_selector_dry_run":
        raise ValidationError(f"{event_id}: invalid event_type.")
    if event["mode"] != MODE or event["algorithm"] != ALGORITHM:
        raise ValidationError(f"{event_id}: invalid mode or algorithm.")
    if event["coverage_method"] != COVERAGE_METHOD:
        raise ValidationError(f"{event_id}: invalid coverage_method.")
    if (
        not _is_finite_number(event["coverage_voxel_size"])
        or float(event["coverage_voxel_size"]) <= 0.0
    ):
        raise ValidationError(
            f"{event_id}: coverage_voxel_size must be positive and finite."
        )
    if not isinstance(event_id, str) or not event_id:
        raise ValidationError("event_id must be a non-empty string.")
    for field in COUNT_FIELDS:
        _require_count(event, field)
    for field in (
        "init",
        "protected_init",
        "selection_applied",
        "all_candidates_forwarded",
        "actual_conservation_pass",
        "empty_candidate",
    ):
        _require_bool(event, field)
    if event["init"] != event["protected_init"]:
        raise ValidationError(
            f"{event_id}: protected_init must equal init."
        )
    if event["status"] not in {"ok", "error"}:
        raise ValidationError(f"{event_id}: invalid status.")
    if not isinstance(event["reason"], str) or not event["reason"]:
        raise ValidationError(f"{event_id}: reason must be non-empty.")
    expected_id = (
        f"gcs-v0:{event['event_sequence']}:"
        f"{event['mapper_update_id']}:{event['source_camera_id']}"
    )
    if event_id != expected_id:
        raise ValidationError(
            f"{event_id}: expected event_id {expected_id!r}."
        )
    if event["selection_applied"] or event["selected_indices"] is not None:
        raise ValidationError(
            f"{event_id}: dry-run cannot apply selection or emit indices."
        )
    if event["candidate_count"] != (
        event["candidate_finite_count"]
        + event["candidate_nonfinite_count"]
    ):
        raise ValidationError(f"{event_id}: finite count conservation failed.")
    if event["candidate_finite_count"] != (
        event["occupied_candidate_count"] + event["novel_candidate_count"]
    ):
        raise ValidationError(
            f"{event_id}: occupied/novel conservation failed."
        )
    if event["candidate_unique_voxel_count"] != (
        event["occupied_unique_voxel_count"]
        + event["novel_unique_voxel_count"]
    ):
        raise ValidationError(
            f"{event_id}: finite unique voxel conservation failed."
        )
    if event["existing_gaussian_count"] != event["gaussian_before"]:
        raise ValidationError(
            f"{event_id}: existing_gaussian_count != gaussian_before."
        )
    if event["empty_candidate"] != (event["candidate_count"] == 0):
        raise ValidationError(f"{event_id}: empty_candidate mismatch.")

    multiplicities, voxel_total, occupied_total = _validate_histogram(event)
    if voxel_total != event["occupied_unique_voxel_count"]:
        raise ValidationError(
            f"{event_id}: histogram occupied voxel count mismatch."
        )
    if occupied_total != event["occupied_candidate_count"]:
        raise ValidationError(
            f"{event_id}: histogram candidate count mismatch."
        )
    expected_mean = (
        occupied_total / voxel_total if voxel_total > 0 else None
    )
    actual_mean = event["occupied_multiplicity_mean"]
    if expected_mean is None:
        if actual_mean is not None:
            raise ValidationError(
                f"{event_id}: multiplicity mean must be null."
            )
    elif not _is_finite_number(actual_mean):
        raise ValidationError(f"{event_id}: multiplicity mean must be finite.")
    else:
        _assert_close(
            event_id,
            "occupied_multiplicity_mean",
            float(actual_mean),
            expected_mean,
        )
    expected_max = max(multiplicities, default=0)
    if event["occupied_multiplicity_max"] != expected_max:
        raise ValidationError(f"{event_id}: multiplicity max mismatch.")
    for field, percentile in (
        ("occupied_multiplicity_p50", 0.50),
        ("occupied_multiplicity_p90", 0.90),
        ("occupied_multiplicity_p95", 0.95),
        ("occupied_multiplicity_p99", 0.99),
    ):
        expected = _nearest_rank(multiplicities, percentile)
        if event[field] != expected:
            raise ValidationError(f"{event_id}: {field} mismatch.")

    _validate_k_scan(event, multiplicities)

    if (
        event["candidate_count"] != event["actual_admitted_count"]
        or event["actual_dropped_count"] != 0
        or not event["all_candidates_forwarded"]
        or event["gaussian_after_extend"]
        != event["gaussian_before"] + event["actual_admitted_count"]
        or not event["actual_conservation_pass"]
    ):
        raise ValidationError(
            f"{event_id}: dry-run actual-path conservation failed."
        )
    if event["dry_run_wall_ms"] is not None and (
        not _is_finite_number(event["dry_run_wall_ms"])
        or float(event["dry_run_wall_ms"]) < 0.0
    ):
        raise ValidationError(
            f"{event_id}: dry_run_wall_ms must be null or non-negative."
        )
    if event["status"] == "ok":
        if event["error"] is not None:
            raise ValidationError(f"{event_id}: ok event needs error=null.")
        if event["candidate_nonfinite_count"] != 0:
            raise ValidationError(
                f"{event_id}: non-finite candidates require error status."
            )
        if event["dry_run_wall_ms"] is None:
            raise ValidationError(
                f"{event_id}: ok event needs finite dry_run_wall_ms."
            )
    else:
        error = event["error"]
        if (
            not isinstance(error, dict)
            or set(error) != {"type", "message"}
            or not all(
                isinstance(value, str) and value for value in error.values()
            )
        ):
            raise ValidationError(
                f"{event_id}: error event needs structured error details."
            )


def validate_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    event_ids: set[str] = set()
    sequences: list[int] = []
    voxel_size: Optional[float] = None
    error_events: list[str] = []
    for event in events:
        validate_event(event)
        event_id = event["event_id"]
        if event_id in event_ids:
            raise ValidationError(f"Duplicate event_id: {event_id}.")
        event_ids.add(event_id)
        sequences.append(event["event_sequence"])
        current_voxel_size = float(event["coverage_voxel_size"])
        if voxel_size is None:
            voxel_size = current_voxel_size
        elif voxel_size != current_voxel_size:
            raise ValidationError(
                "All selector events must use the same inherited voxel size."
            )
        if event["status"] == "error":
            error_events.append(event_id)
    if sequences != list(range(len(events))):
        raise ValidationError(
            "event_sequence must be complete and monotonically increasing "
            "in extraction order."
        )
    if error_events:
        raise ValidationError(
            "Selector error events make the formal summary invalid: "
            f"{error_events}."
        )
    return sorted(events, key=lambda event: event["event_sequence"])


def _stats(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "total": None, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "total": float(sum(values)),
        "mean": float(statistics.fmean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _config_fingerprint(event: dict[str, Any]) -> str:
    payload = {
        "schema": SCHEMA_VERSION,
        "mode": MODE,
        "algorithm": ALGORITHM,
        "coverage_method": COVERAGE_METHOD,
        "coverage_voxel_size": event["coverage_voxel_size"],
        "k_values": list(K_VALUES),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def build_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    events = validate_events(events)
    candidate_total = sum(event["candidate_count"] for event in events)
    histogram_counts: dict[int, int] = {}
    for event in events:
        for bucket in event["occupied_multiplicity_histogram"]:
            multiplicity = bucket["multiplicity"]
            histogram_counts[multiplicity] = (
                histogram_counts.get(multiplicity, 0)
                + bucket["voxel_count"]
            )
    aggregate_histogram = [
        {"multiplicity": multiplicity, "voxel_count": voxel_count}
        for multiplicity, voxel_count in sorted(histogram_counts.items())
    ]
    aggregate_multiplicities = [
        multiplicity
        for bucket in aggregate_histogram
        for multiplicity in [bucket["multiplicity"]] * bucket["voxel_count"]
    ]
    aggregate_occupied_count = sum(
        event["occupied_candidate_count"] for event in events
    )
    aggregate_occupied_unique = len(aggregate_multiplicities)
    k_totals = []
    for k_value in K_VALUES:
        admitted = sum(
            next(item for item in event["k_scan"] if item["k"] == k_value)[
                "estimated_admitted_count"
            ]
            for event in events
        )
        dropped = candidate_total - admitted
        k_totals.append(
            {
                "k": k_value,
                "estimated_admitted_total": admitted,
                "estimated_dropped_total": dropped,
                "admitted_ratio": (
                    admitted / candidate_total if candidate_total else None
                ),
                "dropped_ratio": (
                    dropped / candidate_total if candidate_total else None
                ),
            }
        )
    return {
        "schema": SCHEMA_VERSION,
        "validation_pass": True,
        "errors": [],
        "event_count": len(events),
        "init_event_count": sum(event["init"] for event in events),
        "empty_event_count": sum(event["empty_candidate"] for event in events),
        "error_event_count": 0,
        "candidate_total": candidate_total,
        "candidate_finite_total": sum(
            event["candidate_finite_count"] for event in events
        ),
        "candidate_nonfinite_total": sum(
            event["candidate_nonfinite_count"] for event in events
        ),
        "candidate_unique_voxel_total": sum(
            event["candidate_unique_voxel_count"] for event in events
        ),
        "occupied_candidate_total": sum(
            event["occupied_candidate_count"] for event in events
        ),
        "novel_candidate_total": sum(
            event["novel_candidate_count"] for event in events
        ),
        "occupied_unique_voxel_total": sum(
            event["occupied_unique_voxel_count"] for event in events
        ),
        "novel_unique_voxel_total": sum(
            event["novel_unique_voxel_count"] for event in events
        ),
        "occupied_multiplicity_histogram": aggregate_histogram,
        "occupied_multiplicity_mean": (
            aggregate_occupied_count / aggregate_occupied_unique
            if aggregate_occupied_unique
            else None
        ),
        "occupied_multiplicity_max": max(
            aggregate_multiplicities,
            default=0,
        ),
        "occupied_multiplicity_p50": _nearest_rank(
            aggregate_multiplicities,
            0.50,
        ),
        "occupied_multiplicity_p90": _nearest_rank(
            aggregate_multiplicities,
            0.90,
        ),
        "occupied_multiplicity_p95": _nearest_rank(
            aggregate_multiplicities,
            0.95,
        ),
        "occupied_multiplicity_p99": _nearest_rank(
            aggregate_multiplicities,
            0.99,
        ),
        "k_scan_totals": k_totals,
        "all_candidates_forwarded": all(
            event["all_candidates_forwarded"] for event in events
        ),
        "all_actual_conservation_pass": all(
            event["actual_conservation_pass"] for event in events
        ),
        "dry_run_wall_ms": _stats(
            [float(event["dry_run_wall_ms"]) for event in events]
        ),
        "coverage_voxel_size": events[0]["coverage_voxel_size"],
        "algorithm": ALGORITHM,
        "config_fingerprint": _config_fingerprint(events[0]),
        "interpretation_limit": (
            "Counterfactual per-call estimates on the unmodified map "
            "trajectory; not a prediction of a controlled run."
        ),
    }


def _jsonl_text(events: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(
            event,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for event in events
    )


def _csv_text(events: list[dict[str, Any]]) -> str:
    columns = [
        "event_sequence",
        "event_id",
        "mapper_update_id",
        "source_camera_id",
        "init",
        "candidate_count",
        "occupied_candidate_count",
        "novel_candidate_count",
        "occupied_unique_voxel_count",
        "novel_unique_voxel_count",
        "dry_run_wall_ms",
    ]
    columns.extend(
        f"k{k}_{field}"
        for k in K_VALUES
        for field in ("estimated_admitted_count", "estimated_dropped_count")
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for event in events:
        row = {column: event.get(column) for column in columns}
        for item in event["k_scan"]:
            row[f"k{item['k']}_estimated_admitted_count"] = item[
                "estimated_admitted_count"
            ]
            row[f"k{item['k']}_estimated_dropped_count"] = item[
                "estimated_dropped_count"
            ]
        writer.writerow(row)
    return output.getvalue()


def _markdown_text(summary: dict[str, Any]) -> str:
    lines = [
        "# Gaussian Candidate Selector Dry-Run v0 汇总",
        "",
        "- 验证状态：PASS",
        f"- 事件数：{summary['event_count']}",
        f"- 候选总数：{summary['candidate_total']}",
        f"- occupied 候选：{summary['occupied_candidate_total']}",
        f"- novel 候选：{summary['novel_candidate_total']}",
        f"- 继承体素大小：{summary['coverage_voxel_size']} m",
        "- 说明：仅对未修改地图轨迹进行逐调用反事实估算，不代表真实筛选运行。",
        "",
        "| K | 预计保留 | 预计拒绝 | 保留率 | 拒绝率 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in summary["k_scan_totals"]:
        admitted_ratio = item["admitted_ratio"]
        dropped_ratio = item["dropped_ratio"]
        lines.append(
            "| {k} | {admitted} | {dropped} | {ar} | {dr} |".format(
                k=item["k"],
                admitted=item["estimated_admitted_total"],
                dropped=item["estimated_dropped_total"],
                ar=(f"{admitted_ratio:.6f}" if admitted_ratio is not None else "null"),
                dr=(f"{dropped_ratio:.6f}" if dropped_ratio is not None else "null"),
            )
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    events: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    paths = {name: output_dir / name for name in OUTPUT_FILENAMES}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise ValidationError(
            "Refusing to overwrite existing outputs: " + ", ".join(existing)
        )
    rendered = {
        OUTPUT_FILENAMES[0]: _jsonl_text(events),
        OUTPUT_FILENAMES[1]: json.dumps(
            summary,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        OUTPUT_FILENAMES[2]: _csv_text(events),
        OUTPUT_FILENAMES[3]: _markdown_text(summary),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in rendered.items():
        paths[name].write_text(content, encoding="utf-8", newline="")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate GaussianCandidateSelectorDryRun events and build "
            "auditable summaries."
        )
    )
    parser.add_argument("run_log", type=Path, help="Input run.log")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty summary output directory",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        events = extract_events(args.run_log)
        events = validate_events(events)
        summary = build_summary(events)
        write_outputs(args.output_dir, events, summary)
    except (OSError, ValidationError, ValueError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "GaussianCandidateSelectorDryRun summary validation PASS: "
        f"events={len(events)}, output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
