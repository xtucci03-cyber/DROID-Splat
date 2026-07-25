#!/usr/bin/env python3
"""Rebuild the offline M01/Lifecycle ledger from immutable experiment logs.

This script is intentionally separate from the DROID-Splat runtime.  It accepts
TQDM or other terminal output before and after a structured log event and uses
JSONDecoder.raw_decode() to consume only the first JSON object after a marker.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


M01_MARKER = "[M01:ResourceAdmission]"
LIFECYCLE_MARKER = "[LifecycleObserver]"
COVISIBILITY_RE = re.compile(
    r"Covisibility pruning took .*?pruned:\s*(\d+)\s+Gaussians"
)
FINAL_GAUSSIANS_RE = re.compile(r"Rendering finished with\s+(\d+)\s+Gaussians")
NAN_RE = re.compile(r"\bNaN\b", re.IGNORECASE)
INF_RE = re.compile(r"\bInf\b", re.IGNORECASE)
INVALID_INDEX_RE = re.compile(r"tensor\(\[([^\]]*)\]")

EXPERIMENTS = {
    "006": {
        "directory": "006_m01_fixed_budget600_lifecycle_observer_fr1desk_full_eval_r1",
        "expected": {
            "admission_events": 69,
            "candidate_total": 64684,
            "admission_total": 47855,
            "dropped_total": 16829,
            "lifecycle_events": 18,
            "lifecycle_net": 15274,
            "covisibility_events": 3,
            "covisibility_pruned_total": 9137,
            "invalid_pruned_total": 0,
            "final_gaussians": 53992,
        },
    },
    "007": {
        "directory": "007_m01_observe_lifecycle_observer_fr1desk_full_eval_r1",
        "expected": {
            "admission_events": 69,
            "candidate_total": 64685,
            "admission_total": 64685,
            "dropped_total": 0,
            "lifecycle_events": 18,
            "lifecycle_net": 14446,
            "covisibility_events": 3,
            "covisibility_pruned_total": 11782,
            "invalid_pruned_total": 0,
            "final_gaussians": 67349,
        },
    },
}


def decode_first_json_after_marker(line: str, marker: str) -> tuple[dict[str, Any], int]:
    """Decode the first complete JSON object occurring after marker in line."""
    marker_column = line.find(marker)
    if marker_column < 0:
        raise ValueError(f"Marker {marker!r} is not present.")

    json_column = line.find("{", marker_column + len(marker))
    if json_column < 0:
        raise ValueError(f"No JSON object follows marker {marker!r}: {line!r}")

    value, _end = json.JSONDecoder().raw_decode(line[json_column:])
    if not isinstance(value, dict):
        raise ValueError(f"Structured event after {marker!r} is not an object.")
    return value, marker_column


def parse_invalid_index_count(line: str) -> int:
    """Count printed invalid indices, failing rather than guessing on truncation."""
    match = INVALID_INDEX_RE.search(line)
    if match is None:
        raise ValueError(
            "Cannot strictly parse invalid Gaussian indices from log line: "
            f"{line!r}"
        )
    body = match.group(1).strip()
    if not body:
        return 0
    if "..." in body:
        raise ValueError(
            "Invalid Gaussian index tensor was abbreviated; exact deletion "
            f"count is unavailable: {line!r}"
        )
    entries = [item.strip() for item in body.split(",") if item.strip()]
    if not all(re.fullmatch(r"\d+", item) for item in entries):
        raise ValueError(f"Unexpected invalid Gaussian index format: {line!r}")
    return len(entries)


def parse_log(log_path: Path) -> dict[str, Any]:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    admission_events: list[dict[str, Any]] = []
    lifecycle_events: list[dict[str, Any]] = []
    covisibility_events: list[dict[str, Any]] = []
    invalid_events: list[dict[str, Any]] = []
    cleanup_lines: list[int] = []
    nan_lines: list[int] = []
    inf_lines: list[int] = []
    final_matches: list[tuple[int, int]] = []

    for line_number, line in enumerate(lines, start=1):
        if M01_MARKER in line:
            event, marker_column = decode_first_json_after_marker(line, M01_MARKER)
            event["_log_line_number"] = line_number
            event["_marker_column"] = marker_column
            event["_prefix_at_line_start"] = marker_column == 0
            admission_events.append(event)

        if LIFECYCLE_MARKER in line:
            event, marker_column = decode_first_json_after_marker(
                line, LIFECYCLE_MARKER
            )
            event["_log_line_number"] = line_number
            event["_marker_column"] = marker_column
            lifecycle_events.append(event)

        covisibility = COVISIBILITY_RE.search(line)
        if covisibility:
            covisibility_events.append(
                {
                    "log_line_number": line_number,
                    "pruned": int(covisibility.group(1)),
                    "raw": line,
                }
            )

        if "Found degenerate Gaussians" in line:
            invalid_events.append(
                {
                    "log_line_number": line_number,
                    "pruned": parse_invalid_index_count(line),
                    "raw": line,
                }
            )
        if "Cleaning up by removal" in line:
            cleanup_lines.append(line_number)
        if NAN_RE.search(line):
            nan_lines.append(line_number)
        if INF_RE.search(line):
            inf_lines.append(line_number)

        final_match = FINAL_GAUSSIANS_RE.search(line)
        if final_match:
            final_matches.append((line_number, int(final_match.group(1))))

    if len(final_matches) != 1:
        raise ValueError(
            f"{log_path}: expected exactly one final Gaussian line, "
            f"found {len(final_matches)}."
        )
    if len(cleanup_lines) != len(invalid_events):
        raise ValueError(
            f"{log_path}: found {len(invalid_events)} invalid-index messages but "
            f"{len(cleanup_lines)} cleanup messages."
        )

    return {
        "line_count": len(lines),
        "admission_events": admission_events,
        "lifecycle_events": lifecycle_events,
        "covisibility_events": covisibility_events,
        "invalid_events": invalid_events,
        "cleanup_lines": cleanup_lines,
        "nan_lines": nan_lines,
        "inf_lines": inf_lines,
        "final_log_line_number": final_matches[0][0],
        "final_gaussians": final_matches[0][1],
    }


def csv_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def verify_lifecycle_csv(
    lifecycle_events: list[dict[str, Any]], csv_path: Path
) -> None:
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(lifecycle_events):
        raise ValueError(
            f"{csv_path}: CSV has {len(rows)} events but log has "
            f"{len(lifecycle_events)}."
        )

    for index, (event, row) in enumerate(zip(lifecycle_events, rows), start=1):
        for field, csv_value in row.items():
            if field not in event:
                raise ValueError(
                    f"{csv_path}: event {index} is missing field {field!r}."
                )
            if csv_scalar(event[field]) != csv_value:
                raise ValueError(
                    f"{csv_path}: event {index} field {field!r} differs: "
                    f"log={event[field]!r}, csv={csv_value!r}."
                )


def sum_field(events: list[dict[str, Any]], field: str) -> int:
    return sum(int(event[field]) for event in events)


def build_ledger(label: str, directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = parse_log(directory / "关键日志.txt")
    admissions = parsed["admission_events"]
    lifecycles = parsed["lifecycle_events"]
    covisibility = parsed["covisibility_events"]
    invalid_events = parsed["invalid_events"]

    verify_lifecycle_csv(lifecycles, directory / "lifecycle_events.csv")
    lifecycle_summary = json.loads(
        (directory / "lifecycle_summary.json").read_text(encoding="utf-8")
    )
    old_ledger = json.loads(
        (directory / "unified_lifecycle_ledger.json").read_text(encoding="utf-8")
    )

    camera_uids = [int(event["camera_uid"]) for event in admissions]
    unique_camera_uids = sorted(set(camera_uids))
    admission_contract_violations = sum(
        int(event["gaussian_before"]) + int(event["admitted_count"])
        != int(event["gaussian_after_extend"])
        for event in admissions
    )
    lifecycle_conservation_violations = sum(
        not bool(event["conservation_ok"]) for event in lifecycles
    )

    candidate_total = sum_field(admissions, "candidate_count")
    admission_total = sum_field(admissions, "admitted_count")
    dropped_total = sum_field(admissions, "dropped_count")
    clone_added = sum_field(lifecycles, "clone_added")
    split_net = sum_field(lifecycles, "split_net")
    general_pruned = sum_field(lifecycles, "general_pruned")
    lifecycle_net = sum_field(lifecycles, "densify_prune_net")
    covisibility_pruned_total = sum(
        int(event["pruned"]) for event in covisibility
    )
    invalid_pruned_total = sum(int(event["pruned"]) for event in invalid_events)
    external_pruned_total = covisibility_pruned_total + invalid_pruned_total
    final_gaussians = int(parsed["final_gaussians"])
    conservation_lhs = admission_total + lifecycle_net - external_pruned_total
    conservation_residual = final_gaussians - conservation_lhs

    if candidate_total != admission_total + dropped_total:
        raise ValueError(f"{label}: candidate/admitted/dropped totals do not balance.")
    if admission_contract_violations:
        raise ValueError(f"{label}: admission contract violations were found.")
    if lifecycle_conservation_violations:
        raise ValueError(f"{label}: lifecycle conservation violations were found.")
    if not lifecycle_summary["all_conservation_ok"]:
        raise ValueError(f"{label}: lifecycle_summary reports a conservation failure.")
    if int(lifecycle_summary["event_count"]) != len(lifecycles):
        raise ValueError(f"{label}: lifecycle_summary event count differs.")
    if int(lifecycle_summary["total_densify_prune_net"]) != lifecycle_net:
        raise ValueError(f"{label}: lifecycle_summary net differs.")
    if int(lifecycle_summary["last_gaussian_after"]) != final_gaussians:
        raise ValueError(f"{label}: last lifecycle count differs from final rendering.")

    ledger = {
        "schema": 2,
        "experiment": label,
        "source_log": "关键日志.txt",
        "experiment_rerun": False,
        "source_log_modified": False,
        "correction_scope": "offline_parser_only",
        "admission_events": len(admissions),
        "camera_uid_unique_count": len(unique_camera_uids),
        "camera_uid_min": min(unique_camera_uids),
        "camera_uid_max": max(unique_camera_uids),
        "camera_uid_complete_0_to_68": unique_camera_uids == list(range(69)),
        "candidate_total": candidate_total,
        "admission_total": admission_total,
        "dropped_total": dropped_total,
        "admission_contract_violations": admission_contract_violations,
        "lifecycle_events": len(lifecycles),
        "lifecycle_csv_matches_log": True,
        "lifecycle_conservation_violations": lifecycle_conservation_violations,
        "clone_added": clone_added,
        "split_net": split_net,
        "general_pruned": general_pruned,
        "lifecycle_net": lifecycle_net,
        "covisibility_events": len(covisibility),
        "covisibility_pruned_total": covisibility_pruned_total,
        "invalid_event_count": len(invalid_events),
        "invalid_cleanup_message_count": len(parsed["cleanup_lines"]),
        "nan_log_line_count": len(parsed["nan_lines"]),
        "inf_log_line_count": len(parsed["inf_lines"]),
        "invalid_pruned_total": invalid_pruned_total,
        "external_pruned_total": external_pruned_total,
        "final_gaussians": final_gaussians,
        "final_log_line_number": int(parsed["final_log_line_number"]),
        "conservation_lhs": conservation_lhs,
        "conservation_residual": conservation_residual,
        "conservation_ok": conservation_residual == 0,
        "v1_admission_events": int(old_ledger["admission_events"]),
        "v1_admission_total": int(old_ledger["admission_total"]),
        "v1_unobserved_other_net": int(old_ledger["unobserved_other_net"]),
    }

    expected = EXPERIMENTS[label]["expected"]
    for field, expected_value in expected.items():
        if ledger[field] != expected_value:
            raise ValueError(
                f"{label}: {field}={ledger[field]!r}, expected {expected_value!r}."
            )
    if not ledger["camera_uid_complete_0_to_68"]:
        raise ValueError(f"{label}: camera_uid is not exactly the range 0..68.")
    if ledger["conservation_residual"] != 0:
        raise ValueError(f"{label}: V2 conservation residual is not zero.")

    return ledger, parsed


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def write_admission_csv(path: Path, events: list[dict[str, Any]]) -> None:
    metadata_fields = [
        "log_line_number",
        "marker_column",
        "prefix_at_line_start",
    ]
    event_fields = sorted(
        {
            key
            for event in events
            for key in event
            if not key.startswith("_")
        }
    )
    fieldnames = metadata_fields + event_fields
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            row = {
                "log_line_number": event["_log_line_number"],
                "marker_column": event["_marker_column"],
                "prefix_at_line_start": event["_prefix_at_line_start"],
            }
            row.update(
                {
                    key: csv_value(value)
                    for key, value in event.items()
                    if not key.startswith("_")
                }
            )
            writer.writerow(row)


def summary_markdown(label: str, ledger: dict[str, Any]) -> str:
    return f"""# {label} Lifecycle Ledger V2

## 修正范围

- 原实验未重跑。
- `关键日志.txt` 等原始日志未修改。
- 本次只修正离线M01日志解析规则。
- 旧V1账本漏掉了被TQDM进度输出拼接、因而不位于行首的M01事件。
- V1文件保留作为审计历史，不再作为正确统计结果。

## V2统计

| 指标 | 数值 |
|---|---:|
| M01事件数 | {ledger["admission_events"]} |
| camera_uid唯一数 | {ledger["camera_uid_unique_count"]} |
| camera_uid范围 | {ledger["camera_uid_min"]}–{ledger["camera_uid_max"]} |
| candidate_total | {ledger["candidate_total"]} |
| admission_total | {ledger["admission_total"]} |
| dropped_total | {ledger["dropped_total"]} |
| Lifecycle事件数 | {ledger["lifecycle_events"]} |
| lifecycle_net | {ledger["lifecycle_net"]} |
| Covisibility事件数 | {ledger["covisibility_events"]} |
| covisibility_pruned_total | {ledger["covisibility_pruned_total"]} |
| invalid_pruned_total | {ledger["invalid_pruned_total"]} |
| final_gaussians | {ledger["final_gaussians"]} |

## 数量守恒

```text
{ledger["admission_total"]}
+ {ledger["lifecycle_net"]}
- {ledger["covisibility_pruned_total"]}
- {ledger["invalid_pruned_total"]}
= {ledger["final_gaussians"]}
```

守恒残差：`{ledger["conservation_residual"]}`，检查结果：**PASS**。

## 完整性

- camera_uid恰为0到68且每个出现一次。
- M01每个事件满足`gaussian_before + admitted_count = gaussian_after_extend`。
- 18个LifecycleObserver事件与`lifecycle_events.csv`一致。
- 每个Lifecycle事件`conservation_ok=true`。
- 三个Covisibility事件均已解析。
- 未发现invalid删除事件。
- 最终Gaussian与评价日志及最后一个Lifecycle事件一致。
"""


def correction_note_markdown() -> str:
    return """# Lifecycle Ledger V2 Correction Note

## 修正性质

- 006和007原实验均未重跑。
- 原始日志及V1文件均未修改。
- 修正对象仅为离线日志解析规则。
- 旧解析过程漏掉了被TQDM进度输出拼接、`[M01:ResourceAdmission]`
  不位于行首的事件。
- 旧V1文件保留作为审计历史，不再作为正确统计结果。

## 006漏计事件

```text
camera_uid = 54
candidate_count = 607
admitted_count = 600
dropped_count = 7
```

旧账本的`unobserved_other_net=-8537`实际组成为：

```text
-9137 Covisibility pruning
+600 漏计Admission
= -8537
```

## 007漏计事件

```text
camera_uid = 53
candidate_count = 696
admitted_count = 696
dropped_count = 0
```

旧账本的`unobserved_other_net=-11086`实际组成为：

```text
-11782 Covisibility pruning
+696 漏计Admission
= -11086
```

## 差异修正

旧差异2549不能命名为Covisibility pruning差异。真实Covisibility删除差异为：

```text
11782 - 9137 = 2645
```
"""


def comparison_values(
    ledger_006: dict[str, Any], ledger_007: dict[str, Any]
) -> dict[str, Any]:
    admission_reduction = (
        ledger_007["admission_total"] - ledger_006["admission_total"]
    )
    final_reduction = (
        ledger_007["final_gaussians"] - ledger_006["final_gaussians"]
    )
    lifecycle_rebound = ledger_006["lifecycle_net"] - ledger_007["lifecycle_net"]
    covisibility_less_removed = (
        ledger_007["covisibility_pruned_total"]
        - ledger_006["covisibility_pruned_total"]
    )
    downstream_offset = lifecycle_rebound + covisibility_less_removed
    retained_percent = final_reduction / admission_reduction * 100.0
    difference_residual = (
        -admission_reduction
        + lifecycle_rebound
        + covisibility_less_removed
        + final_reduction
    )
    values = {
        "admission_reduction": admission_reduction,
        "final_reduction": final_reduction,
        "lifecycle_rebound": lifecycle_rebound,
        "covisibility_less_removed": covisibility_less_removed,
        "downstream_offset": downstream_offset,
        "retained_percent": retained_percent,
        "difference_residual": difference_residual,
    }
    expected = {
        "admission_reduction": 16830,
        "final_reduction": 13357,
        "lifecycle_rebound": 828,
        "covisibility_less_removed": 2645,
        "downstream_offset": 3473,
        "difference_residual": 0,
    }
    for field, expected_value in expected.items():
        if values[field] != expected_value:
            raise ValueError(
                f"Comparison {field}={values[field]}, expected {expected_value}."
            )
    return values


def comparison_markdown(values: dict[str, Any]) -> str:
    return f"""# 006 vs 007 Lifecycle Ledger V2

## 数据来源与修正范围

- 两个原实验均未重跑，原始日志未修改。
- 本比较使用修正M01解析规则后生成的V2账本。
- V1文件保留作为审计历史，不再作为正确统计结果。

## 差异分解

| 指标 | 数值 |
|---|---:|
| 006相对007 Admission减少 | {values["admission_reduction"]} |
| 006相对007 最终Gaussian减少 | {values["final_reduction"]} |
| Densify/prune反弹 | {values["lifecycle_rebound"]} |
| Covisibility少删除 | {values["covisibility_less_removed"]} |
| 总下游抵消 | {values["downstream_offset"]} |
| 入口收益最终保留率 | {values["retained_percent"]:.6f}% |

## 差异守恒

```text
-{values["admission_reduction"]}
+ {values["lifecycle_rebound"]}
+ {values["covisibility_less_removed"]}
= -{values["final_reduction"]}
```

差异守恒残差：`{values["difference_residual"]}`。

旧账本中的2549不能命名为Covisibility pruning差异；真实Covisibility删除差异为2645。
"""


def write_new_text(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records-root",
        required=True,
        type=Path,
        help="Directory containing the immutable 006 and 007 experiment folders.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Create V2 outputs. Existing V2 files are never overwritten.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records_root = args.records_root.resolve()
    ledgers: dict[str, dict[str, Any]] = {}
    parsed_logs: dict[str, dict[str, Any]] = {}
    directories: dict[str, Path] = {}

    for label, spec in EXPERIMENTS.items():
        directory = records_root / spec["directory"]
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        directories[label] = directory
        ledgers[label], parsed_logs[label] = build_ledger(label, directory)

    comparison = comparison_values(ledgers["006"], ledgers["007"])

    if args.write:
        for label in ("006", "007"):
            directory = directories[label]
            write_admission_csv(
                directory / "admission_events_v2.csv",
                parsed_logs[label]["admission_events"],
            )
            write_new_text(
                directory / "unified_lifecycle_ledger_v2.json",
                json.dumps(ledgers[label], ensure_ascii=False, indent=2) + "\n",
            )
            write_new_text(
                directory / "SUMMARY_v2.md",
                summary_markdown(label, ledgers[label]),
            )
            write_new_text(
                directory / "CORRECTION_NOTE.md",
                correction_note_markdown(),
            )
        write_new_text(
            directories["007"] / "COMPARE_006_vs_007_v2.md",
            comparison_markdown(comparison),
        )

    result = {
        "records_root": str(records_root),
        "write": args.write,
        "006": ledgers["006"],
        "007": ledgers["007"],
        "comparison": comparison,
        "validation": "PASS",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
