"""Synthetic CUDA microbenchmark for GCS-v1 Active Top-k600.

The script measures the existing production Active selector against the
existing M01 deterministic fixed600 admission path.  It does not read config,
records, or datasets and does not create worker or background processes.
"""

import argparse
import gc
import json
import math
import statistics
import sys
import time
from typing import Any, Callable

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from src.resource_management.m01_resource_admission.resource_admission import (
    ResourceAdmission,
)
from tests.test_gaussian_candidate_active_topk_v1_cuda import (
    CUDA_DEVICE,
    _cuda_fixture,
    _select,
)


MIB = 1024 * 1024
SIZES = (600, 601, 2400, 10000)
PRODUCTION_SCOPE = "active_production_scope"
HARNESS_BOUNDARY_SCOPE = "active_harness_boundary_sync_scope"


class SyncProfileBlocked(RuntimeError):
    """A profiler compatibility failure, distinct from Active/CUDA failure."""

    def __init__(self, blocked_stage: str, error: BaseException) -> None:
        super().__init__(f"{blocked_stage}: {type(error).__name__}: {error}")
        self.blocked_stage = blocked_stage
        self.original_error = error

    def to_record(self) -> dict[str, Any]:
        return {
            "status": "BLOCKED",
            "profiler_status": "BLOCKED",
            "blocked_stage": self.blocked_stage,
            "exception_type": type(self.original_error).__name__,
            "exception_message": str(self.original_error),
            "SYNC_PROFILE": "BLOCKED",
            "CUDA_GATE": "BLOCKED",
        }


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA benchmark requires torch.cuda.is_available() == True; "
            "refusing to skip."
        )
    if torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA benchmark requires at least one GPU.")
    torch.cuda.set_device(CUDA_DEVICE)
    torch.cuda.synchronize(CUDA_DEVICE)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sample.")
    ordered = sorted(values)
    rank = max(0, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[rank])


def _latency_summary(milliseconds: list[float]) -> dict[str, float]:
    if not milliseconds or not all(math.isfinite(value) for value in milliseconds):
        raise RuntimeError("Latency samples must be non-empty and finite.")
    return {
        "p50_ms": _percentile(milliseconds, 0.50),
        "p95_ms": _percentile(milliseconds, 0.95),
        "mean_ms": float(statistics.fmean(milliseconds)),
    }


def _cleanup_cuda() -> tuple[int, int]:
    torch.cuda.synchronize(CUDA_DEVICE)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(CUDA_DEVICE)
    return (
        int(torch.cuda.memory_allocated(CUDA_DEVICE)),
        int(torch.cuda.memory_reserved(CUDA_DEVICE)),
    )


def _measure_latency(
    operation: Callable[[], Any],
    *,
    warmup: int,
    repetitions: int,
) -> dict[str, float]:
    for _ in range(warmup):
        output = operation()
        del output
    torch.cuda.synchronize(CUDA_DEVICE)

    samples_ms: list[float] = []
    for _ in range(repetitions):
        torch.cuda.synchronize(CUDA_DEVICE)
        started_ns = time.perf_counter_ns()
        output = operation()
        torch.cuda.synchronize(CUDA_DEVICE)
        finished_ns = time.perf_counter_ns()
        samples_ms.append((finished_ns - started_ns) / 1_000_000.0)
        del output
    return _latency_summary(samples_ms)


def _measure_memory(
    operation: Callable[[], Any],
    *,
    repetitions: int,
) -> dict[str, Any]:
    warm = operation()
    torch.cuda.synchronize(CUDA_DEVICE)
    del warm
    baseline_allocated, baseline_reserved = _cleanup_cuda()
    torch.cuda.reset_peak_memory_stats(CUDA_DEVICE)

    for _ in range(repetitions):
        output = operation()
        del output
    torch.cuda.synchronize(CUDA_DEVICE)
    peak_allocated = int(torch.cuda.max_memory_allocated(CUDA_DEVICE))
    peak_reserved = int(torch.cuda.max_memory_reserved(CUDA_DEVICE))

    cleanup_attempts = []
    after_allocated = after_reserved = 0
    for attempt in range(1, 4):
        after_allocated, after_reserved = _cleanup_cuda()
        allocated_delta = after_allocated - baseline_allocated
        cleanup_attempts.append(
            {
                "attempt": attempt,
                "allocated_bytes": after_allocated,
                "reserved_bytes": after_reserved,
                "allocated_residual_bytes": allocated_delta,
                "reserved_residual_bytes": after_reserved - baseline_reserved,
            }
        )
        if allocated_delta <= MIB:
            break

    allocated_residual = after_allocated - baseline_allocated
    result = {
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_delta_bytes": peak_allocated - baseline_allocated,
        "peak_reserved_delta_bytes": peak_reserved - baseline_reserved,
        "after_cleanup_allocated_bytes": after_allocated,
        "after_cleanup_reserved_bytes": after_reserved,
        "allocated_residual_bytes": allocated_residual,
        "reserved_residual_bytes": after_reserved - baseline_reserved,
        "cleanup_attempts": cleanup_attempts,
        "cleanup_status": "PASS" if allocated_residual <= MIB else "FAIL",
    }
    if result["cleanup_status"] != "PASS":
        raise RuntimeError(
            "Active CUDA allocations remain more than 1 MiB above the warm "
            f"baseline after three cleanups: {result}."
        )
    return result


def _make_workloads(count: int):
    torch.manual_seed(1729 + count)
    torch.cuda.manual_seed_all(1729 + count)
    scores = torch.rand(count, dtype=torch.float32, device=CUDA_DEVICE)
    selector, _, camera, candidates = _cuda_fixture(count, scores)
    fixed600 = ResourceAdmission(
        mode="fixed_budget",
        fixed_budget=600,
        selection="deterministic_uniform",
    )

    def active_operation():
        return _select(selector, camera, candidates)

    def fixed_operation():
        return fixed600.admit_before_extend(
            xyz=candidates[0],
            features=candidates[1],
            scales=candidates[2],
            rotations=candidates[3],
            opacities=candidates[4],
            camera_uid=camera.uid,
            kf_id=camera.uid,
            init=False,
            gaussian_before=11,
        )

    return active_operation, fixed_operation


def _benchmark_size(
    count: int,
    *,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    active_operation, fixed_operation = _make_workloads(count)
    active_latency = _measure_latency(
        active_operation,
        warmup=warmup,
        repetitions=repetitions,
    )
    fixed_latency = _measure_latency(
        fixed_operation,
        warmup=warmup,
        repetitions=repetitions,
    )
    active_memory = _measure_memory(active_operation, repetitions=repetitions)
    fixed_memory = _measure_memory(fixed_operation, repetitions=repetitions)

    mean_overhead = active_latency["mean_ms"] - fixed_latency["mean_ms"]
    if fixed_latency["mean_ms"] <= 0:
        raise RuntimeError("Fixed600 mean latency must be positive.")
    result = {
        "candidate_count": count,
        "warmup": warmup,
        "repetitions": repetitions,
        "active_latency": active_latency,
        "fixed600_latency": fixed_latency,
        "active_absolute_mean_overhead_ms": mean_overhead,
        "active_relative_mean_ratio": (
            active_latency["mean_ms"] / fixed_latency["mean_ms"]
        ),
        "active_memory": active_memory,
        "fixed600_memory": fixed_memory,
        "active_minus_fixed_peak_allocated_bytes": (
            active_memory["peak_allocated_delta_bytes"]
            - fixed_memory["peak_allocated_delta_bytes"]
        ),
        "active_minus_fixed_peak_reserved_bytes": (
            active_memory["peak_reserved_delta_bytes"]
            - fixed_memory["peak_reserved_delta_bytes"]
        ),
    }
    numeric_values = (
        list(active_latency.values())
        + list(fixed_latency.values())
        + [mean_overhead, result["active_relative_mean_ratio"]]
    )
    if not all(math.isfinite(float(value)) for value in numeric_values):
        raise RuntimeError(f"Non-finite benchmark result: {result}.")
    return result


def _raise_captured(error_info) -> None:
    error_type, error, traceback = error_info
    del error_type
    raise error.with_traceback(traceback)


def _run_recorded_scope(
    scope_name: str,
    operation: Callable[[], Any],
) -> Any:
    try:
        scope = record_function(scope_name)
    except Exception as error:
        raise SyncProfileBlocked(f"{scope_name}_construct", error) from error
    try:
        scope.__enter__()
    except Exception as error:
        raise SyncProfileBlocked(f"{scope_name}_enter", error) from error

    operation_error = None
    result = None
    try:
        result = operation()
    except BaseException:
        operation_error = sys.exc_info()
    try:
        scope.__exit__(*(operation_error or (None, None, None)))
    except Exception as error:
        if operation_error is None:
            raise SyncProfileBlocked(f"{scope_name}_exit", error) from error
    if operation_error is not None:
        _raise_captured(operation_error)
    return result


def _capture_profile(active_operation: Callable[[], Any]):
    try:
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        )
    except Exception as error:
        raise SyncProfileBlocked("profiler_construct", error) from error
    try:
        profiler.__enter__()
    except Exception as error:
        raise SyncProfileBlocked("profiler_enter", error) from error

    captured_error = None
    try:
        output = _run_recorded_scope(PRODUCTION_SCOPE, active_operation)
        del output
        _run_recorded_scope(
            HARNESS_BOUNDARY_SCOPE,
            lambda: torch.cuda.synchronize(CUDA_DEVICE),
        )
    except BaseException:
        captured_error = sys.exc_info()
    try:
        profiler.__exit__(*(captured_error or (None, None, None)))
    except Exception as error:
        if captured_error is None:
            raise SyncProfileBlocked("profiler_exit", error) from error
    if captured_error is not None:
        _raise_captured(captured_error)
    return profiler


def _event_name(event: Any) -> str:
    name = getattr(event, "name", None)
    if name is None:
        name = getattr(event, "key", None)
    if name is None:
        raise RuntimeError("Profiler event has neither name nor key.")
    return str(name)


def _scope_descendants(scope: Any) -> set[int]:
    children = getattr(scope, "cpu_children", None)
    if children is None:
        raise RuntimeError("Profiler event hierarchy has no cpu_children field.")
    descendants: set[int] = set()
    pending = list(children)
    while pending:
        event = pending.pop()
        identity = id(event)
        if identity in descendants:
            continue
        descendants.add(identity)
        nested = getattr(event, "cpu_children", None)
        if nested is None:
            raise RuntimeError("Profiler child event has no cpu_children field.")
        pending.extend(nested)
    return descendants


def _event_interval(event: Any) -> tuple[float, float]:
    event_name = _event_name(event)
    interval = getattr(event, "time_range", None)
    start = getattr(interval, "start", None)
    end = getattr(interval, "end", None)
    if start is None or end is None:
        raise RuntimeError(
            f"Profiler event {event_name!r} has no compatible time_range: "
            f"start={start!r}, end={end!r}."
        )
    try:
        normalized_start = float(start)
        normalized_end = float(end)
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(
            f"Profiler event {event_name!r} has non-numeric time_range: "
            f"start={start!r}, end={end!r}."
        ) from error
    if not math.isfinite(normalized_start) or not math.isfinite(normalized_end):
        raise RuntimeError(
            f"Profiler event {event_name!r} has non-finite time_range: "
            f"start={normalized_start!r}, end={normalized_end!r}."
        )
    if normalized_start > normalized_end:
        raise RuntimeError(
            f"Profiler event {event_name!r} has reversed time_range: "
            f"start={normalized_start!r}, end={normalized_end!r}."
        )
    return normalized_start, normalized_end


def _sync_summary(events: list[Any]) -> dict[str, Any]:
    counts = {
        "aten::item": 0,
        "aten::_local_scalar_dense": 0,
        "cudaDeviceSynchronize": 0,
        "cudaStreamSynchronize": 0,
        "dtoh_memcpy_related": 0,
    }
    diagnostic_copy_counts = {
        "aten::to": 0,
        "aten::_to_copy": 0,
        "aten::copy_": 0,
    }
    matched = []
    for event in events:
        key = _event_name(event)
        lower = key.lower()
        if key == "aten::item":
            counts["aten::item"] += 1
        if key == "aten::_local_scalar_dense":
            counts["aten::_local_scalar_dense"] += 1
        if "cudadevicesynchronize" in lower:
            counts["cudaDeviceSynchronize"] += 1
        if "cudastreamsynchronize" in lower:
            counts["cudaStreamSynchronize"] += 1
        if (
            "dtoh" in lower
            or ("memcpy" in lower and "device to host" in lower)
            or "memcpy dtoh" in lower
        ):
            counts["dtoh_memcpy_related"] += 1
        if key in diagnostic_copy_counts:
            diagnostic_copy_counts[key] += 1
        if any(counts.values()) or key in diagnostic_copy_counts:
            if (
                key in {"aten::item", "aten::_local_scalar_dense"}
                or "synchronize" in lower
                or "memcpy" in lower
                or key in diagnostic_copy_counts
            ):
                matched.append(key)
    explicit_sync_total = sum(counts.values())
    return {
        "explicit_sync_counts": counts,
        "explicit_sync_total": explicit_sync_total,
        "diagnostic_copy_counts": diagnostic_copy_counts,
        "matched_event_names": matched,
    }


def _analyze_profile_events(profiler: Any) -> dict[str, Any]:
    try:
        events = list(profiler.events())
        production_scopes = [
            event for event in events if _event_name(event) == PRODUCTION_SCOPE
        ]
        harness_scopes = [
            event
            for event in events
            if _event_name(event) == HARNESS_BOUNDARY_SCOPE
        ]
        if len(production_scopes) != 1 or len(harness_scopes) != 1:
            raise RuntimeError(
                "Expected one production scope and one harness scope, got "
                f"production={len(production_scopes)}, "
                f"harness={len(harness_scopes)}."
            )
        production_scope = production_scopes[0]
        harness_scope = harness_scopes[0]
        production_ids = _scope_descendants(production_scope)
        harness_ids = _scope_descendants(harness_scope)
        event_ids = {id(event) for event in events}
        if not production_ids or not harness_ids:
            raise RuntimeError(
                "Profiler scope hierarchy is empty and cannot support reliable "
                "synchronization attribution."
            )
        if not production_ids.issubset(event_ids) or not harness_ids.issubset(
            event_ids
        ):
            raise RuntimeError(
                "Profiler scope children cannot be matched to the extracted "
                "event list."
            )
        production_events = [event for event in events if id(event) in production_ids]
        harness_events = [event for event in events if id(event) in harness_ids]
        production_summary = _sync_summary(production_events)
        harness_summary = _sync_summary(harness_events)

        production_interval = _event_interval(production_scope)
        harness_interval = _event_interval(harness_scope)
        if production_ids & harness_ids:
            raise RuntimeError(
                "Production and harness profiler scopes have overlapping "
                "event hierarchies."
            )
        production_start, production_end = production_interval
        harness_start, harness_end = harness_interval
        if not (
            production_start
            <= production_end
            <= harness_start
            <= harness_end
        ):
            raise RuntimeError(
                "Profiler scope order is invalid; expected "
                "production_start <= production_end <= harness_start <= "
                "harness_end, got "
                f"production=({production_start!r}, {production_end!r}), "
                f"harness=({harness_start!r}, {harness_end!r})."
            )
        infrastructure_events = []
        unattributed_events = []
        excluded = production_ids | harness_ids | {
            id(production_scope),
            id(harness_scope),
        }
        for event in events:
            if id(event) in excluded:
                continue
            event_summary = _sync_summary([event])
            if event_summary["explicit_sync_total"] == 0:
                continue
            start, end = _event_interval(event)
            if end <= production_interval[0] or start >= harness_interval[1]:
                infrastructure_events.append(event)
            else:
                unattributed_events.append(event)

        infrastructure_summary = _sync_summary(infrastructure_events)
        unattributed_summary = _sync_summary(unattributed_events)
        if unattributed_summary["explicit_sync_total"]:
            raise RuntimeError(
                "Explicit synchronization events could not be attributed to "
                "the production, harness-boundary, or profiler-infrastructure "
                "scope."
            )
    except SyncProfileBlocked:
        raise
    except Exception as error:
        raise SyncProfileBlocked("profiler_event_extraction", error) from error

    production_observed = production_summary["explicit_sync_total"] > 0
    return {
        "status": "PASS",
        "profiler_status": "PASS",
        "blocked_stage": None,
        "exception_type": None,
        "exception_message": None,
        "candidate_count": 2400,
        "production_scope_name": PRODUCTION_SCOPE,
        "harness_boundary_scope_name": HARNESS_BOUNDARY_SCOPE,
        "PRODUCTION_SCOPE_SYNC": production_summary,
        "HARNESS_BOUNDARY_SYNC": {
            "known_explicit_synchronize_calls": 1,
            "profiled_events": harness_summary,
        },
        "PROFILER_INFRASTRUCTURE_SYNC": infrastructure_summary,
        "UNATTRIBUTED_SYNC": unattributed_summary,
        "SYNC_RISK": "OBSERVED" if production_observed else "NOT_OBSERVED",
        "SYNC_PROFILE": "PASS",
        "scope_note": (
            "Only explicit synchronization evidence inside active_production_scope "
            "can set SYNC_RISK=OBSERVED. aten::to/copy diagnostics do not."
        ),
    }


def _profile_active_sync_points() -> dict[str, Any]:
    active_operation, _ = _make_workloads(2400)
    for _ in range(10):
        output = active_operation()
        del output
    torch.cuda.synchronize(CUDA_DEVICE)
    profiler = _capture_profile(active_operation)
    return _analyze_profile_events(profiler)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic CUDA benchmark for GCS-v1 Active Top-k600."
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path for the JSON report; stdout is always emitted.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _require_cuda()
    environment = {
        "python_cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "device": str(CUDA_DEVICE),
        "gpu_name": torch.cuda.get_device_name(CUDA_DEVICE),
        "compute_capability": list(torch.cuda.get_device_capability(CUDA_DEVICE)),
        "multiprocessing_used": False,
        "cuda_ipc_used": False,
        "background_process_used": False,
        "synthetic_inputs_only": True,
    }
    results = []
    for count in SIZES:
        repetitions = 20 if count == 10000 else 50
        results.append(
            _benchmark_size(count, warmup=10, repetitions=repetitions)
        )
    try:
        sync_profile = _profile_active_sync_points()
    except SyncProfileBlocked as error:
        sync_profile = error.to_record()
        cuda_gate = "BLOCKED"
        exit_code = 2
    else:
        cuda_gate = None
        exit_code = 0
    report = {
        "schema": 1,
        "event_type": "gcs_v1_active_topk600_cuda_microbenchmark",
        "environment": environment,
        "results": results,
        "sync_profile": sync_profile,
        "SYNC_PROFILE": sync_profile["status"],
        "CUDA_GATE": cuda_gate,
        "memory_cleanup": {
            str(result["candidate_count"]): result["active_memory"][
                "cleanup_status"
            ]
            for result in results
        },
    }
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(payload, flush=True)
    if args.json_output is not None:
        with open(args.json_output, "x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.write("\n")
    print(f"SYNC_PROFILE={sync_profile['status']}", flush=True)
    if cuda_gate is not None:
        print(f"CUDA_GATE={cuda_gate}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
