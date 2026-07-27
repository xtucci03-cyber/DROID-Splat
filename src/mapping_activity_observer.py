import json
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch


_LOG_PREFIX = "[MappingActivityObserver] "
_MEMORY_FIELDS = (
    "memory_allocated_bytes",
    "memory_reserved_bytes",
    "max_memory_allocated_bytes",
    "max_memory_reserved_bytes",
)
_GRADIENT_PARAMETERS = (
    "xyz",
    "features_dc",
    "features_rest",
    "opacity",
    "scaling",
    "rotation",
)
_RENDER_PROXY_PARAMETERS = (
    "xyz",
    "features_dc",
    "features_rest",
    "opacity",
    "rotation",
)
_CUDA_STAGE_NAMES = (
    "iteration_total",
    "render_forward",
    "loss_computation",
    "backward",
    "densification_stats",
    "densify_and_prune",
    "optimizer_step",
    "zero_grad_housekeeping",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def _minimum(values: Sequence[float]) -> Optional[float]:
    return float(min(values)) if values else None


def _maximum(values: Sequence[float]) -> Optional[float]:
    return float(max(values)) if values else None


def _ratio(count: Optional[int], total: int) -> Optional[float]:
    if count is None or total <= 0:
        return None
    return float(count / total)


class MappingActivityObserver:
    """Low-frequency, default-off diagnostics for Gaussian mapping activity."""

    schema = 1

    def __init__(self, config: Mapping[str, Any], device: Any):
        self.sample_every = self._validate_sample_every(
            config.get("sample_every", 1)
        )
        self.gradient_eps = self._validate_gradient_eps(
            config.get("gradient_eps", 0.0)
        )
        self.log_iteration_events = bool(
            config.get("log_iteration_events", True)
        )
        self.log_update_summary = bool(config.get("log_update_summary", True))
        self.collect_visible = bool(config.get("collect_visible", True))
        self.collect_touched_proxy = bool(
            config.get("collect_touched_proxy", True)
        )
        self.collect_gradient_active = bool(
            config.get("collect_gradient_active", True)
        )
        self.collect_cuda_timing = bool(
            config.get("collect_cuda_timing", True)
        )
        self.collect_memory = bool(config.get("collect_memory", True))

        self.device = torch.device(device)
        self.cuda_available = (
            self.device.type == "cuda" and torch.cuda.is_available()
        )
        self._event_counter = 0
        self._updates_completed = 0
        self._iteration_events_emitted = 0
        self._active_update: Optional[Dict[str, Any]] = None
        self._finalized = False
        self._observer_errors: List[Dict[str, Any]] = []

    @property
    def has_active_update(self) -> bool:
        return self._active_update is not None

    @staticmethod
    def _validate_sample_every(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"activity_observer.sample_every must be an integer >= 1, got {value!r}."
            )
        if value < 1:
            raise ValueError(
                f"activity_observer.sample_every must be >= 1, got {value!r}."
            )
        return value

    @staticmethod
    def _validate_gradient_eps(value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"activity_observer.gradient_eps must be a finite number >= 0, got {value!r}."
            )
        normalized = float(value)
        if not math.isfinite(normalized) or normalized < 0:
            raise ValueError(
                f"activity_observer.gradient_eps must be a finite number >= 0, got {value!r}."
            )
        return normalized

    @classmethod
    def validate_config(cls, config: Mapping[str, Any]) -> None:
        cls._validate_sample_every(config.get("sample_every", 1))
        cls._validate_gradient_eps(config.get("gradient_eps", 0.0))

    def record_error(
        self,
        operation: str,
        error: BaseException,
        update_id: Optional[int] = None,
        iteration_id: Optional[int] = None,
    ) -> None:
        diagnostic = {
            "operation": str(operation),
            "error_type": type(error).__name__,
            "message": str(error),
            "update_id": (
                int(update_id)
                if update_id is not None
                else (
                    int(self._active_update["update_id"])
                    if self._active_update is not None
                    else None
                )
            ),
            "iteration_id": (
                int(iteration_id) if iteration_id is not None else None
            ),
        }
        self._observer_errors.append(diagnostic)
        if self._active_update is not None:
            self._active_update["invariant_failures"].append(
                "observer_error:"
                + diagnostic["operation"]
                + ":"
                + diagnostic["error_type"]
            )

    def _common_event(
        self,
        event_type: str,
        update_id: Optional[int],
        iteration_id: Optional[int],
        sampled: bool,
        status: str = "ok",
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._event_counter += 1
        return {
            "schema": self.schema,
            "event_id": f"mapper-activity:{os.getpid()}:{self._event_counter}",
            "event_type": event_type,
            "timestamp_utc": _utc_now(),
            "pid": os.getpid(),
            "process_role": "mapper",
            "update_id": int(update_id) if update_id is not None else None,
            "iteration_id": (
                int(iteration_id) if iteration_id is not None else None
            ),
            "sampled": bool(sampled),
            "status": str(status),
            "reason": reason,
        }

    @staticmethod
    def _emit(event: Dict[str, Any]) -> None:
        print(
            _LOG_PREFIX
            + json.dumps(
                event,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            flush=True,
        )

    def begin_update(
        self,
        update_id: int,
        update_kind: str,
        configured_iteration_total: int,
        gaussian_count: int,
    ) -> None:
        if self._active_update is not None:
            raise RuntimeError("Mapping activity update is already active.")
        self._active_update = {
            "update_id": int(update_id),
            "update_kind": str(update_kind),
            "configured_iteration_total": int(configured_iteration_total),
            "global_count_start": int(gaussian_count),
            "iteration_total": 0,
            "records": [],
            "pending_cuda_events": [],
            "invariant_failures": [],
            "explicit_observer_sync_count": 0,
        }

    def begin_iteration(
        self,
        update_id: int,
        iteration_id: int,
        gaussian_count: int,
    ) -> Optional[Dict[str, Any]]:
        update = self._require_update(update_id)
        update["iteration_total"] += 1
        if iteration_id % self.sample_every != 0:
            return None

        record = self._common_event(
            event_type="iteration_activity",
            update_id=update_id,
            iteration_id=iteration_id,
            sampled=True,
        )
        record.update(
            {
                "camera_uids": [],
                "sample_every": self.sample_every,
                "gradient_eps": self.gradient_eps,
                "global_gaussian_count": int(gaussian_count),
                "global_gaussian_count_at_backward": None,
                "global_gaussian_count_after_iteration": None,
                "visible_count_per_camera": [],
                "visible_union_count": None,
                "visible_union_ratio": None,
                "touched_positive_count_per_camera": [],
                "touched_union_count": None,
                "touched_union_ratio": None,
                "touched_proxy_status": (
                    "pending"
                    if self.collect_touched_proxy
                    else "not_collected"
                ),
                "touched_proxy_reason": (
                    None
                    if self.collect_touched_proxy
                    else "collection_disabled"
                ),
                "gradient_active_any_count": None,
                "gradient_active_any_ratio": None,
                "gradient_active_render_proxy_count": None,
                "gradient_active_render_proxy_ratio": None,
                "scaling_grad_active_count": None,
                "scaling_grad_active_ratio": None,
                "grad_none_parameter_names": [],
                "densify_and_prune_executed": False,
                "cpu_wall_ms_by_stage": {
                    stage: 0.0 for stage in _CUDA_STAGE_NAMES
                },
                "cuda_ms_by_stage": {
                    stage: (
                        0.0
                        if self.collect_cuda_timing and self.cuda_available
                        else None
                    )
                    for stage in _CUDA_STAGE_NAMES
                },
                "memory_snapshots": {},
                "memory_boundary_sample_count": {},
                "memory_boundary_aggregation": (
                    "maximum_for_repeated_camera_boundaries"
                ),
                "invariant_failures": [],
                "_visible_union_mask": None,
                "_touched_union_mask": None,
                "_touched_visibility_violation_count": None,
            }
        )
        update["records"].append(record)
        self.memory_boundary(record, "before_forward")
        record["_iteration_total_token"] = self.begin_stage(
            record, "iteration_total"
        )
        return record

    def _require_update(self, update_id: int) -> Dict[str, Any]:
        if self._active_update is None:
            raise RuntimeError("No mapping activity update is active.")
        if int(update_id) != self._active_update["update_id"]:
            raise RuntimeError(
                "Mapping activity update mismatch: "
                f"active={self._active_update['update_id']}, got={update_id}."
            )
        return self._active_update

    def begin_stage(
        self, record: Optional[Dict[str, Any]], stage: str
    ) -> Optional[Dict[str, Any]]:
        if record is None:
            return None
        if stage not in _CUDA_STAGE_NAMES:
            raise ValueError(f"Unknown mapping activity stage: {stage!r}.")

        gpu_start = None
        if self.collect_cuda_timing and self.cuda_available:
            gpu_start = torch.cuda.Event(enable_timing=True)
            gpu_start.record()
        return {
            "stage": stage,
            "cpu_start_ns": time.perf_counter_ns(),
            "gpu_start": gpu_start,
        }

    def end_stage(
        self,
        record: Optional[Dict[str, Any]],
        token: Optional[Dict[str, Any]],
    ) -> None:
        if record is None or token is None:
            return
        stage = token["stage"]
        record["cpu_wall_ms_by_stage"][stage] += (
            time.perf_counter_ns() - token["cpu_start_ns"]
        ) / 1_000_000.0

        if token["gpu_start"] is not None:
            gpu_end = torch.cuda.Event(enable_timing=True)
            gpu_end.record()
            update = self._active_update
            if update is None:
                raise RuntimeError("CUDA stage ended without an active update.")
            update["pending_cuda_events"].append(
                (token["gpu_start"], gpu_end, record, stage)
            )

    def memory_boundary(
        self, record: Optional[Dict[str, Any]], boundary: str
    ) -> None:
        if (
            record is None
            or not self.collect_memory
            or not self.cuda_available
        ):
            return
        snapshot = {
            "cuda_device": str(self.device),
            "memory_allocated_bytes": int(
                torch.cuda.memory_allocated(device=self.device)
            ),
            "memory_reserved_bytes": int(
                torch.cuda.memory_reserved(device=self.device)
            ),
            "max_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device=self.device)
            ),
            "max_memory_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device=self.device)
            ),
            "max_fields_are_process_cumulative": True,
        }
        record["memory_boundary_sample_count"][boundary] = (
            record["memory_boundary_sample_count"].get(boundary, 0) + 1
        )
        previous = record["memory_snapshots"].get(boundary)
        if previous is None:
            record["memory_snapshots"][boundary] = snapshot
            return
        for key in _MEMORY_FIELDS:
            previous[key] = max(previous[key], snapshot[key])

    def observe_render(
        self,
        record: Optional[Dict[str, Any]],
        camera_uid: int,
        visibility_filter: Any,
        n_touched: Any,
    ) -> None:
        if record is None:
            return
        record["camera_uids"].append(int(camera_uid))
        n_gaussians = record["global_gaussian_count"]

        need_visibility = self.collect_visible or self.collect_touched_proxy
        visibility_mask = None
        if need_visibility:
            if (
                not isinstance(visibility_filter, torch.Tensor)
                or visibility_filter.ndim != 1
                or visibility_filter.shape[0] != n_gaussians
            ):
                if self.collect_visible:
                    record["visible_count_per_camera"].append(None)
                record["invariant_failures"].append(
                    "invalid_visibility_filter"
                )
            else:
                visibility_mask = visibility_filter.to(dtype=torch.bool)
                visible_count = torch.count_nonzero(visibility_mask)
                if self.collect_visible:
                    record["visible_count_per_camera"].append(visible_count)
                    if record["_visible_union_mask"] is None:
                        record["_visible_union_mask"] = torch.zeros_like(
                            visibility_mask, dtype=torch.bool
                        )
                    record["_visible_union_mask"].logical_or_(visibility_mask)

        if not self.collect_touched_proxy:
            return
        if visibility_mask is None:
            record["touched_positive_count_per_camera"].append(None)
            record["touched_proxy_status"] = "unavailable"
            record["touched_proxy_reason"] = "visibility_filter_invalid"
            return
        if not isinstance(n_touched, torch.Tensor):
            record["touched_positive_count_per_camera"].append(None)
            record["touched_proxy_status"] = "unavailable"
            record["touched_proxy_reason"] = "n_touched_missing_or_not_tensor"
            return
        if (
            n_touched.ndim != 1
            or n_touched.shape[0] != n_gaussians
            or n_touched.device != visibility_mask.device
        ):
            record["touched_positive_count_per_camera"].append(None)
            record["touched_proxy_status"] = "unavailable"
            record["touched_proxy_reason"] = (
                "n_touched_shape_or_device_mismatch"
            )
            return

        touched_mask = n_touched > 0
        touched_count = torch.count_nonzero(touched_mask)
        record["touched_positive_count_per_camera"].append(touched_count)
        if record["_touched_union_mask"] is None:
            record["_touched_union_mask"] = torch.zeros_like(
                touched_mask, dtype=torch.bool
            )
        record["_touched_union_mask"].logical_or_(touched_mask)
        violation_count = torch.count_nonzero(
            torch.logical_and(touched_mask, torch.logical_not(visibility_mask))
        )
        if record["_touched_visibility_violation_count"] is None:
            record["_touched_visibility_violation_count"] = violation_count
        else:
            record["_touched_visibility_violation_count"] = (
                record["_touched_visibility_violation_count"] + violation_count
            )
        if record["touched_proxy_status"] == "pending":
            record["touched_proxy_status"] = "ok"
            record["touched_proxy_reason"] = None
        elif record["touched_proxy_status"] == "unavailable":
            record["touched_proxy_status"] = "partial_unavailable"

    def finalize_activity_masks(
        self, record: Optional[Dict[str, Any]]
    ) -> None:
        if record is None:
            return
        visible_union = record.pop("_visible_union_mask", None)
        touched_union = record.pop("_touched_union_mask", None)
        if visible_union is not None and self.collect_visible:
            record["visible_union_count"] = torch.count_nonzero(visible_union)
        if touched_union is not None and self.collect_touched_proxy:
            record["touched_union_count"] = torch.count_nonzero(touched_union)
        del visible_union
        del touched_union

    @staticmethod
    def _row_active(
        parameter: torch.Tensor, n_gaussians: int, eps: float
    ) -> Optional[torch.Tensor]:
        grad = parameter.grad
        if grad is None or grad.numel() == 0:
            return None
        if grad.shape[0] != n_gaussians:
            raise ValueError(
                f"Gradient first dimension {grad.shape[0]} does not match N={n_gaussians}."
            )
        flattened = grad.reshape(n_gaussians, -1)
        return torch.any(torch.abs(flattened) > eps, dim=1)

    @classmethod
    def gradient_activity_counts(
        cls,
        parameters: Mapping[str, torch.Tensor],
        n_gaussians: int,
        gradient_eps: float,
    ) -> Dict[str, Any]:
        missing = [name for name in _GRADIENT_PARAMETERS if name not in parameters]
        if missing:
            raise ValueError(f"Missing Gaussian parameters: {missing}.")

        device = parameters["xyz"].device
        active_any = torch.zeros(
            n_gaussians, dtype=torch.bool, device=device
        )
        active_render_proxy = torch.zeros_like(active_any)
        scaling_active = torch.zeros_like(active_any)
        grad_none_names: List[str] = []

        with torch.no_grad():
            for name in _GRADIENT_PARAMETERS:
                parameter = parameters[name]
                if parameter.shape[0] != n_gaussians:
                    raise ValueError(
                        f"Parameter {name!r} first dimension "
                        f"{parameter.shape[0]} does not match N={n_gaussians}."
                    )
                if parameter.device != device:
                    raise ValueError(
                        f"Parameter {name!r} is on {parameter.device}, expected {device}."
                    )
                row_active = cls._row_active(
                    parameter, n_gaussians, gradient_eps
                )
                if row_active is None:
                    if parameter.grad is None:
                        grad_none_names.append(name)
                    continue
                active_any.logical_or_(row_active)
                if name in _RENDER_PROXY_PARAMETERS:
                    active_render_proxy.logical_or_(row_active)
                if name == "scaling":
                    scaling_active.logical_or_(row_active)
                del row_active

            result = {
                "gradient_active_any_count": torch.count_nonzero(active_any),
                "gradient_active_render_proxy_count": torch.count_nonzero(
                    active_render_proxy
                ),
                "scaling_grad_active_count": torch.count_nonzero(
                    scaling_active
                ),
                "grad_none_parameter_names": grad_none_names,
            }
        return result

    def observe_after_backward(
        self,
        record: Optional[Dict[str, Any]],
        parameters: Mapping[str, torch.Tensor],
    ) -> None:
        if record is None:
            return
        n_gaussians = int(parameters["xyz"].shape[0])
        if record["global_gaussian_count_at_backward"] is None:
            record["global_gaussian_count_at_backward"] = n_gaussians
        elif record["global_gaussian_count_at_backward"] != n_gaussians:
            record["invariant_failures"].append(
                "gaussian_count_changed_during_backward"
            )
        if (
            n_gaussians != record["global_gaussian_count"]
            and "gaussian_count_changed_before_backward"
            not in record["invariant_failures"]
        ):
            record["invariant_failures"].append(
                "gaussian_count_changed_before_backward"
            )
        if not self.collect_gradient_active:
            return
        result = self.gradient_activity_counts(
            parameters=parameters,
            n_gaussians=n_gaussians,
            gradient_eps=self.gradient_eps,
        )
        record.update(result)

    def observe_before_backward(
        self,
        record: Optional[Dict[str, Any]],
        gaussian_count: int,
    ) -> None:
        if record is None:
            return
        record["global_gaussian_count_at_backward"] = int(gaussian_count)
        if int(gaussian_count) != record["global_gaussian_count"]:
            record["invariant_failures"].append(
                "gaussian_count_changed_before_backward"
            )

    def mark_densify_and_prune(
        self, record: Optional[Dict[str, Any]], executed: bool
    ) -> None:
        if record is not None:
            record["densify_and_prune_executed"] = bool(executed)

    def end_iteration(
        self,
        record: Optional[Dict[str, Any]],
        gaussian_count: int,
    ) -> None:
        if record is None:
            return
        record["global_gaussian_count_after_iteration"] = int(gaussian_count)
        token = record.pop("_iteration_total_token", None)
        self.end_stage(record, token)

    def _has_pending_cuda_scalars(self, update: Dict[str, Any]) -> bool:
        for record in update["records"]:
            for value in self._walk_values(record):
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim == 0
                    and value.is_cuda
                ):
                    return True
        return False

    @staticmethod
    def _walk_values(value: Any):
        if isinstance(value, dict):
            for child in value.values():
                yield from MappingActivityObserver._walk_values(child)
        elif isinstance(value, list):
            for child in value:
                yield from MappingActivityObserver._walk_values(child)
        else:
            yield value

    @staticmethod
    def _collect_scalar_locations(
        value: Any,
        locations: List[Tuple[Any, Any, torch.Tensor]],
    ) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(child, torch.Tensor) and child.ndim == 0:
                    locations.append((value, key, child))
                else:
                    MappingActivityObserver._collect_scalar_locations(
                        child, locations
                    )
        elif isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, torch.Tensor) and child.ndim == 0:
                    locations.append((value, index, child))
                else:
                    MappingActivityObserver._collect_scalar_locations(
                        child, locations
                    )

    def _convert_pending_scalars(self, update: Dict[str, Any]) -> None:
        locations: List[Tuple[Any, Any, torch.Tensor]] = []
        for record in update["records"]:
            self._collect_scalar_locations(record, locations)

        groups: Dict[str, List[Tuple[Any, Any, torch.Tensor]]] = {}
        for location in locations:
            groups.setdefault(str(location[2].device), []).append(location)

        for group in groups.values():
            tensors = [entry[2].detach().reshape(()) for entry in group]
            values = torch.stack(tensors).cpu().tolist()
            for (container, key, _), converted in zip(group, values):
                container[key] = int(converted)

    def _finish_cuda_work(self, update: Dict[str, Any]) -> None:
        pending = update["pending_cuda_events"]
        has_scalars = self._has_pending_cuda_scalars(update)
        if pending:
            pending[-1][1].synchronize()
            update["explicit_observer_sync_count"] = 1
            for start, end, record, stage in pending:
                record["cuda_ms_by_stage"][stage] += float(
                    start.elapsed_time(end)
                )
        elif has_scalars:
            torch.cuda.synchronize(device=self.device)
            update["explicit_observer_sync_count"] = 1

        self._convert_pending_scalars(update)
        pending.clear()

    @staticmethod
    def _finalize_iteration_record(record: Dict[str, Any]) -> None:
        n_gaussians = record["global_gaussian_count"]
        record["visible_union_ratio"] = _ratio(
            record["visible_union_count"], n_gaussians
        )
        record["touched_union_ratio"] = _ratio(
            record["touched_union_count"], n_gaussians
        )
        record["gradient_active_any_ratio"] = _ratio(
            record["gradient_active_any_count"], n_gaussians
        )
        record["gradient_active_render_proxy_ratio"] = _ratio(
            record["gradient_active_render_proxy_count"], n_gaussians
        )
        record["scaling_grad_active_ratio"] = _ratio(
            record["scaling_grad_active_count"], n_gaussians
        )

        violation_count = record.pop(
            "_touched_visibility_violation_count", None
        )
        if violation_count not in (None, 0):
            record["invariant_failures"].append(
                f"touched_not_subset_of_visible:{violation_count}"
            )
        for key in (
            "visible_union_count",
            "touched_union_count",
            "gradient_active_any_count",
            "gradient_active_render_proxy_count",
            "scaling_grad_active_count",
        ):
            count = record[key]
            if count is not None and not 0 <= count <= n_gaussians:
                record["invariant_failures"].append(
                    f"{key}_out_of_range:{count}"
                )
        visible = record["visible_union_count"]
        touched = record["touched_union_count"]
        if visible is not None and touched is not None and touched > visible:
            record["invariant_failures"].append(
                "touched_union_exceeds_visible_union"
            )
        for touched_count, visible_count in zip(
            record["touched_positive_count_per_camera"],
            record["visible_count_per_camera"],
        ):
            if (
                touched_count is not None
                and visible_count is not None
                and touched_count > visible_count
            ):
                record["invariant_failures"].append(
                    "touched_per_camera_exceeds_visible_per_camera"
                )

        record["status"] = (
            "ok" if not record["invariant_failures"] else "error"
        )
        record["reason"] = (
            None
            if not record["invariant_failures"]
            else ";".join(record["invariant_failures"])
        )

    def _build_update_summary(
        self,
        update: Dict[str, Any],
        gaussian_count: int,
        status: str,
        reason: Optional[str],
    ) -> Dict[str, Any]:
        records = update["records"]
        failures = list(update["invariant_failures"])
        for record in records:
            failures.extend(record["invariant_failures"])

        globals_seen = [r["global_gaussian_count"] for r in records]
        visible_counts = [
            r["visible_union_count"]
            for r in records
            if r["visible_union_count"] is not None
        ]
        visible_ratios = [
            r["visible_union_ratio"]
            for r in records
            if r["visible_union_ratio"] is not None
        ]
        touched_counts = [
            r["touched_union_count"]
            for r in records
            if r["touched_union_count"] is not None
        ]
        touched_ratios = [
            r["touched_union_ratio"]
            for r in records
            if r["touched_union_ratio"] is not None
        ]
        gradient_counts = [
            r["gradient_active_any_count"]
            for r in records
            if r["gradient_active_any_count"] is not None
        ]
        gradient_ratios = [
            r["gradient_active_any_ratio"]
            for r in records
            if r["gradient_active_any_ratio"] is not None
        ]
        render_gradient_counts = [
            r["gradient_active_render_proxy_count"]
            for r in records
            if r["gradient_active_render_proxy_count"] is not None
        ]
        render_gradient_ratios = [
            r["gradient_active_render_proxy_ratio"]
            for r in records
            if r["gradient_active_render_proxy_ratio"] is not None
        ]
        scaling_counts = [
            r["scaling_grad_active_count"]
            for r in records
            if r["scaling_grad_active_count"] is not None
        ]

        memory_maxima = {key: None for key in _MEMORY_FIELDS}
        memory_boundary_maxima: Dict[str, Dict[str, int]] = {}
        for record in records:
            for boundary, snapshot in record["memory_snapshots"].items():
                boundary_max = memory_boundary_maxima.setdefault(
                    boundary, {key: 0 for key in _MEMORY_FIELDS}
                )
                for key in _MEMORY_FIELDS:
                    value = snapshot[key]
                    boundary_max[key] = max(boundary_max[key], value)
                    memory_maxima[key] = (
                        value
                        if memory_maxima[key] is None
                        else max(memory_maxima[key], value)
                    )

        summary = self._common_event(
            event_type="update_summary",
            update_id=update["update_id"],
            iteration_id=None,
            sampled=False,
            status=("error" if failures else status),
            reason=(
                ";".join(failures)
                if failures
                else reason
            ),
        )
        summary.update(
            {
                "update_kind": update["update_kind"],
                "sample_every": self.sample_every,
                "gradient_eps": self.gradient_eps,
                "configured_iteration_total": update[
                    "configured_iteration_total"
                ],
                "iteration_total": update["iteration_total"],
                "sampled_iteration_count": len(records),
                "global_count_start": update["global_count_start"],
                "global_count_end": int(gaussian_count),
                "global_count_min": (
                    int(min(globals_seen)) if globals_seen else None
                ),
                "global_count_max": (
                    int(max(globals_seen)) if globals_seen else None
                ),
                "visible_union_count_mean": _mean(visible_counts),
                "visible_union_count_min": _minimum(visible_counts),
                "visible_union_count_max": _maximum(visible_counts),
                "visible_union_ratio_mean": _mean(visible_ratios),
                "touched_union_count_mean": _mean(touched_counts),
                "touched_union_ratio_mean": _mean(touched_ratios),
                "gradient_active_any_count_mean": _mean(gradient_counts),
                "gradient_active_any_ratio_mean": _mean(gradient_ratios),
                "gradient_active_render_proxy_count_mean": _mean(
                    render_gradient_counts
                ),
                "gradient_active_render_proxy_ratio_mean": _mean(
                    render_gradient_ratios
                ),
                "scaling_grad_active_count_mean": _mean(scaling_counts),
                "render_forward_cuda_ms_total": self._stage_total(
                    records, "render_forward"
                ),
                "loss_cuda_ms_total": self._stage_total(
                    records, "loss_computation"
                ),
                "backward_cuda_ms_total": self._stage_total(
                    records, "backward"
                ),
                "densification_stats_cuda_ms_total": self._stage_total(
                    records, "densification_stats"
                ),
                "densify_and_prune_cuda_ms_total": self._stage_total(
                    records, "densify_and_prune"
                ),
                "optimizer_step_cuda_ms_total": self._stage_total(
                    records, "optimizer_step"
                ),
                "zero_grad_housekeeping_cuda_ms_total": self._stage_total(
                    records, "zero_grad_housekeeping"
                ),
                "sampled_iteration_cuda_ms_total": self._stage_total(
                    records, "iteration_total"
                ),
                "cpu_wall_ms_totals_by_stage": {
                    stage: float(
                        sum(
                            record["cpu_wall_ms_by_stage"][stage]
                            for record in records
                        )
                    )
                    for stage in _CUDA_STAGE_NAMES
                },
                "memory_maxima": memory_maxima,
                "memory_boundary_maxima": memory_boundary_maxima,
                "explicit_observer_sync_count": update[
                    "explicit_observer_sync_count"
                ],
                "invariant_failures": failures,
                "means_are_sampled_iterations_only": True,
                "global_count_extrema_are_sampled_iterations_only": True,
                "memory_maxima_are_sampled_boundaries_only": True,
            }
        )
        return summary

    @staticmethod
    def _stage_total(
        records: Sequence[Dict[str, Any]], stage: str
    ) -> Optional[float]:
        values = [
            record["cuda_ms_by_stage"][stage]
            for record in records
            if record["cuda_ms_by_stage"][stage] is not None
        ]
        return float(sum(values)) if values else None

    def end_update(
        self,
        gaussian_count: int,
        status: str = "ok",
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._active_update is None:
            raise RuntimeError("No mapping activity update is active.")
        update = self._active_update
        for record in update["records"]:
            self.finalize_activity_masks(record)
        self._finish_cuda_work(update)
        for record in update["records"]:
            self._finalize_iteration_record(record)
            for key in [
                key for key in record if str(key).startswith("_")
            ]:
                record.pop(key, None)
            if self.log_iteration_events:
                self._emit(record)
                self._iteration_events_emitted += 1

        summary = self._build_update_summary(
            update=update,
            gaussian_count=gaussian_count,
            status=status,
            reason=reason,
        )
        if self.log_update_summary:
            self._emit(summary)

        self._updates_completed += 1
        update["records"].clear()
        update["pending_cuda_events"].clear()
        self._active_update = None
        return summary

    def finalize(self, gaussian_count: int) -> Dict[str, Any]:
        if self._active_update is not None:
            self.end_update(
                gaussian_count=gaussian_count,
                status="error",
                reason="finalize_with_active_update",
            )
        event = self._common_event(
            event_type="finalize_summary",
            update_id=None,
            iteration_id=None,
            sampled=False,
            status=("error" if self._observer_errors else "ok"),
            reason=(
                "observer_errors_recorded"
                if self._observer_errors
                else None
            ),
        )
        event.update(
            {
                "update_count": self._updates_completed,
                "iteration_event_count": self._iteration_events_emitted,
                "final_gaussian_count": int(gaussian_count),
                "observer_errors": list(self._observer_errors),
                "pending_update": False,
                "pending_cuda_event_count": 0,
                "observed_scope": "gaussian_mapper_updates_only",
                "refinement_mapping_steps_observed": False,
            }
        )
        self._emit(event)
        self._observer_errors.clear()
        self._finalized = True
        return event
