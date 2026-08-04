"""Observe-only confidence evidence for Gaussian candidates.

The observer reconstructs candidate-to-pixel lineage by projecting the
already-created world-space candidate coordinates into their source Camera.
It reads a provenance-checked confidence snapshot and records aggregate
quality evidence.  It never returns candidates or selection indices.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from numbers import Real
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch


LOG_PREFIX = "[GaussianCandidateSelectorV1]"
SCHEMA_VERSION = 1
MODE = "observe"
ACTIVE_MODE = "active"
SUPPORTED_MODES = frozenset({"off", MODE, ACTIVE_MODE})
COVERAGE_METHOD = "voxel_occupancy"
K_VALUES = (1, 2, 4, 8)
COUNTERFACTUAL_SORT = (
    "(candidate_voxel_id, -confidence, original_candidate_index)"
)
CONFIDENCE_SAMPLING_METHOD = "lowres_cell_floor_v1"

_TOP_LEVEL_FIELDS = frozenset({"mode", "budget", "logging"})
_LOGGING_FIELDS = frozenset({"enabled"})


def _structured_error(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping, got {type(value).__name__}.")
    return value


def _reject_unknown_fields(
    config: Mapping[str, Any],
    allowed: frozenset[str],
    field: str,
) -> None:
    unknown = sorted(set(config.keys()) - allowed)
    if unknown:
        raise ValueError(f"{field} contains unknown fields: {unknown}.")


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be bool, got {value!r}.")
    return value


def _normalize_mode(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(
            "mapping.candidate_selector_v1.mode must be a string, "
            f"got {value!r}."
        )
    mode = value.strip().lower()
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            "mapping.candidate_selector_v1.mode must be one of "
            f"{sorted(SUPPORTED_MODES)}, got {mode!r}."
        )
    return mode


def _normalize_voxel_size(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            "candidate_observer.voxel_size must be a positive finite "
            f"number, got {value!r}."
        )
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(
            "candidate_observer.voxel_size must be a positive finite "
            f"number, got {value!r}."
        )
    return normalized


def build_gaussian_candidate_selector_v1(
    config: Optional[Mapping[str, Any]],
    *,
    candidate_observer: Any,
    resource_admission_mode: str,
    confidence_snapshot_getter: Callable[..., Any],
    device: Any,
) -> Optional[Any]:
    """Build the observer, returning None without runtime state when off."""

    if config is None:
        return None
    config = _require_mapping(config, "mapping.candidate_selector_v1")
    _reject_unknown_fields(
        config,
        _TOP_LEVEL_FIELDS,
        "mapping.candidate_selector_v1",
    )
    mode = _normalize_mode(config.get("mode", "off"))
    logging = _require_mapping(
        config.get("logging", {"enabled": True}),
        "mapping.candidate_selector_v1.logging",
    )
    _reject_unknown_fields(
        logging,
        _LOGGING_FIELDS,
        "mapping.candidate_selector_v1.logging",
    )
    logging_enabled = _require_bool(
        logging.get("enabled", True),
        "mapping.candidate_selector_v1.logging.enabled",
    )
    if mode == "off":
        return None
    if mode == ACTIVE_MODE:
        from .gaussian_candidate_active_topk_v1 import (
            build_gaussian_candidate_active_topk_v1,
        )

        return build_gaussian_candidate_active_topk_v1(
            config,
            candidate_observer=candidate_observer,
            resource_admission_mode=resource_admission_mode,
            confidence_snapshot_getter=confidence_snapshot_getter,
            device=device,
        )
    budget = config.get("budget", 600)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget != 600:
        raise ValueError(
            "mapping.candidate_selector_v1.budget is frozen to integer 600."
        )
    if not logging_enabled:
        raise ValueError(
            "mapping.candidate_selector_v1.logging.enabled must be true "
            "in observe mode."
        )
    normalized_admission_mode = str(resource_admission_mode).strip().lower()
    if normalized_admission_mode not in {"disabled", "observe"}:
        raise ValueError(
            "Gaussian Candidate Selector v1 observe mode requires M01 "
            "ResourceAdmission to be disabled, absent, or observe; got "
            f"{resource_admission_mode!r}."
        )
    if candidate_observer is None:
        raise ValueError(
            "Gaussian Candidate Selector v1 requires "
            "mapping.candidate_observer.mode=observe."
        )
    if not hasattr(candidate_observer, "voxel_size"):
        raise AttributeError(
            "Gaussian Candidate Observer must expose public voxel_size."
        )
    if not callable(confidence_snapshot_getter):
        raise TypeError("confidence_snapshot_getter must be callable.")
    return GaussianCandidateSelectorV1(
        voxel_size=_normalize_voxel_size(candidate_observer.voxel_size),
        confidence_snapshot_getter=confidence_snapshot_getter,
        logging_enabled=logging_enabled,
        device=device,
    )


@dataclass(frozen=True)
class CandidateQualityEvidenceToken:
    """Tensor-free state finalized after the unmodified extend path."""

    fields: dict[str, Any]


@dataclass(frozen=True)
class CandidateQualityEvidenceSummary:
    fields: dict[str, Any]

    def to_event(self) -> dict[str, Any]:
        return dict(self.fields)


class GaussianCandidateSelectorV1:
    """Collect confidence evidence without producing a selection."""

    schema = SCHEMA_VERSION

    def __init__(
        self,
        *,
        voxel_size: float,
        confidence_snapshot_getter: Callable[..., Any],
        logging_enabled: bool,
        device: Any,
    ) -> None:
        self.voxel_size = _normalize_voxel_size(voxel_size)
        if not callable(confidence_snapshot_getter):
            raise TypeError("confidence_snapshot_getter must be callable.")
        self._confidence_snapshot_getter = confidence_snapshot_getter
        self.logging_enabled = _require_bool(
            logging_enabled,
            "candidate_selector_v1.logging_enabled",
        )
        self.device = torch.device(device)
        self._event_sequence = 0

    def _next_event_sequence(self) -> int:
        sequence = self._event_sequence
        self._event_sequence += 1
        return sequence

    @staticmethod
    def _normalize_current_gaussian_xyz(
        current_gaussian_xyz: torch.Tensor,
    ) -> torch.Tensor:
        if (
            isinstance(current_gaussian_xyz, torch.Tensor)
            and current_gaussian_xyz.ndim == 1
            and current_gaussian_xyz.numel() == 0
        ):
            return current_gaussian_xyz.reshape(0, 3)
        if (
            not isinstance(current_gaussian_xyz, torch.Tensor)
            or current_gaussian_xyz.ndim != 2
            or current_gaussian_xyz.shape[1] != 3
        ):
            raise ValueError(
                "current_gaussian_xyz must have shape [G,3], got "
                f"{getattr(current_gaussian_xyz, 'shape', None)}."
            )
        return current_gaussian_xyz

    @staticmethod
    def _validate_candidates(
        *,
        xyz: torch.Tensor,
        features: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
    ) -> tuple[int, dict[str, torch.Tensor]]:
        tensors = {
            "xyz": xyz,
            "features": features,
            "scales": scales,
            "rotations": rotations,
            "opacities": opacities,
        }
        if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
            types = {name: type(value).__name__ for name, value in tensors.items()}
            raise TypeError(f"Candidate inputs must be tensors, got {types}.")
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"Candidate xyz must have shape [N,3], got {list(xyz.shape)}.")
        count = int(xyz.shape[0])
        shapes = {name: list(value.shape) for name, value in tensors.items()}
        if any(value.ndim < 1 or int(value.shape[0]) != count for value in tensors.values()):
            raise ValueError(
                "Candidate first-dimension mismatch: "
                f"candidate_count={count}, shapes={shapes}."
            )
        devices = {name: str(value.device) for name, value in tensors.items()}
        if any(value.device != xyz.device for value in tensors.values()):
            raise ValueError(f"Candidate device mismatch: devices={devices}.")
        return count, tensors

    @staticmethod
    def _tensor_state(tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        return {
            name: {
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "requires_grad": bool(value.requires_grad),
                "data_ptr": int(value.data_ptr()),
                "version": int(value._version),
            }
            for name, value in tensors.items()
        }

    @staticmethod
    def _fingerprint(value: Any) -> str:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def _index_fingerprint(cls, indices: torch.Tensor) -> str:
        indices = indices.to(dtype=torch.int64)
        if int(indices.shape[0]) == 0:
            values = [0, 0, 0]
        else:
            positions = torch.arange(
                1,
                int(indices.shape[0]) + 1,
                dtype=torch.int64,
                device=indices.device,
            )
            aggregates = torch.stack(
                (
                    torch.tensor(indices.shape[0], dtype=torch.int64, device=indices.device),
                    indices.sum(),
                    (indices * positions).sum(),
                )
            )
            values = [int(value) for value in aggregates.cpu().tolist()]
        return cls._fingerprint(values)

    @staticmethod
    def _stats(values: torch.Tensor) -> dict[str, Optional[float]]:
        if int(values.shape[0]) == 0:
            return {"count": 0, "min": None, "max": None, "mean": None}
        aggregate = torch.stack((values.amin(), values.amax(), values.mean()))
        minimum, maximum, mean = [float(value) for value in aggregate.cpu().tolist()]
        if not all(math.isfinite(value) for value in (minimum, maximum, mean)):
            raise ValueError("Observer statistics contain NaN or Inf.")
        return {
            "count": int(values.shape[0]),
            "min": minimum,
            "max": maximum,
            "mean": mean,
        }

    @staticmethod
    def _camera_scalar(value: Any, name: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        scalar = torch.as_tensor(value, device=device, dtype=dtype)
        if scalar.numel() != 1 or not bool(torch.isfinite(scalar).item()):
            raise ValueError(f"Camera {name} must be one finite scalar.")
        return scalar.reshape(())

    @staticmethod
    def _source_depth(camera: Any, depthmap: Any, depth_source: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if depth_source == "explicit_depthmap":
            source = depthmap
        elif depth_source == "estimated_clean_depth":
            source = getattr(camera, "depth", None)
        elif depth_source == "depth_prior":
            source = getattr(camera, "depth_prior", None)
        else:
            raise ValueError(
                "Source depth cannot be reconstructed exactly for "
                f"depth_source={depth_source!r}."
            )
        if source is None:
            raise ValueError(f"Source depth is missing for depth_source={depth_source!r}.")
        if isinstance(source, torch.Tensor):
            result = source.detach().to(device=device, dtype=dtype)
        else:
            result = torch.as_tensor(np.asarray(source), device=device, dtype=dtype)
        if result.ndim != 2:
            raise ValueError(f"Source depth must have shape [H,W], got {list(result.shape)}.")
        return result

    def project_world_to_source_pixels(
        self,
        xyz: torch.Tensor,
        camera: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project [N,3] world coordinates with the Camera's explicit W2C."""

        if not isinstance(xyz, torch.Tensor) or xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape [N,3].")
        pose_w2c = getattr(camera, "pose", None)
        if not isinstance(pose_w2c, torch.Tensor) or tuple(pose_w2c.shape) != (4, 4):
            raise ValueError("Camera pose must be a [4,4] W2C tensor.")
        pose_w2c = pose_w2c.detach().to(device=xyz.device, dtype=xyz.dtype)
        if not bool(torch.isfinite(pose_w2c).all().item()):
            raise ValueError("Camera W2C pose contains NaN or Inf.")
        camera_xyz = xyz @ pose_w2c[:3, :3].transpose(0, 1) + pose_w2c[:3, 3]
        z = camera_xyz[:, 2]
        projection_finite = torch.isfinite(camera_xyz).all(dim=1)
        positive_z = z > 0
        fx = self._camera_scalar(camera.fx, "fx", xyz.device, xyz.dtype)
        fy = self._camera_scalar(camera.fy, "fy", xyz.device, xyz.dtype)
        cx = self._camera_scalar(camera.cx, "cx", xyz.device, xyz.dtype)
        cy = self._camera_scalar(camera.cy, "cy", xyz.device, xyz.dtype)
        if bool((fx <= 0).item()) or bool((fy <= 0).item()):
            raise ValueError("Camera fx and fy must be positive.")
        safe_z = torch.where(positive_z, z, torch.ones_like(z))
        u_float = fx * camera_xyz[:, 0] / safe_z + cx
        v_float = fy * camera_xyz[:, 1] / safe_z + cy
        valid = (
            projection_finite
            & positive_z
            & torch.isfinite(u_float)
            & torch.isfinite(v_float)
        )
        return (
            torch.round(u_float).to(dtype=torch.long),
            torch.round(v_float).to(dtype=torch.long),
            z,
            valid,
        )

    @staticmethod
    def _classify_snapshot_error(error: BaseException) -> str:
        message = str(error).lower()
        if "stale" in message:
            return "stale_confidence"
        if "timestamp mismatch" in message:
            return "confidence_timestamp_mismatch"
        if "identity mismatch" in message or "source_frame_id" in message:
            return "confidence_identity_mismatch"
        if message.startswith("confidence/candidate device mismatch"):
            return "confidence_device_mismatch"
        if message.startswith("source depth/camera image shape mismatch"):
            return "source_depth_camera_shape_mismatch"
        if message.startswith(
            (
                "candidate inputs must be tensors",
                "candidate xyz must have shape",
                "candidate first-dimension mismatch",
                "candidate device mismatch",
            )
        ):
            return "candidate_contract_error"
        if (
            message.startswith("confidence")
            and "shape" in message
            and "mismatch" in message
        ):
            return "confidence_shape_mismatch"
        if "nan or inf" in message or "non-finite" in message:
            return "nonfinite_confidence"
        if "not valid" in message or "no immutable" in message:
            return "missing_confidence"
        return "confidence_snapshot_unavailable"

    @staticmethod
    def map_source_pixels_to_confidence(
        u: torch.Tensor,
        v: torch.Tensor,
        *,
        source_resolution: tuple[int, int],
        confidence_resolution: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """Map source pixels to confidence cells without clamping."""
        if u.shape != v.shape:
            raise ValueError("Source pixel coordinate shape mismatch.")
        source_height, source_width = source_resolution
        confidence_height, confidence_width = confidence_resolution
        if min(
            source_height,
            source_width,
            confidence_height,
            confidence_width,
        ) <= 0:
            raise ValueError("Confidence/source resolution must be positive.")
        source_in_bounds = (
            (u >= 0)
            & (u < source_width)
            & (v >= 0)
            & (v < source_height)
        )
        if not bool(torch.all(source_in_bounds).item()):
            raise ValueError("Source pixel coordinate is out of bounds.")

        confidence_u = torch.div(
            u * confidence_width,
            source_width,
            rounding_mode="floor",
        )
        confidence_v = torch.div(
            v * confidence_height,
            source_height,
            rounding_mode="floor",
        )
        confidence_in_bounds = (
            (confidence_u >= 0)
            & (confidence_u < confidence_width)
            & (confidence_v >= 0)
            & (confidence_v < confidence_height)
        )
        if not bool(torch.all(confidence_in_bounds).item()):
            raise ValueError("Mapped confidence coordinate is out of bounds.")
        return (
            confidence_u,
            confidence_v,
            source_width / confidence_width,
            source_height / confidence_height,
        )

    def _base_fields(
        self,
        *,
        sequence: int,
        mapper_update_id: int,
        camera: Any,
        init: bool,
        candidate_count: int,
        gaussian_before: int,
    ) -> dict[str, Any]:
        source_camera_id = getattr(camera, "uid", None)
        mapper_id = mapper_update_id
        try:
            source_camera_id = int(source_camera_id)
        except (TypeError, ValueError, OverflowError):
            source_camera_id = None
        try:
            mapper_id = int(mapper_id)
        except (TypeError, ValueError, OverflowError):
            mapper_id = None
        source_timestamp = getattr(camera, "source_timestamp", None)
        try:
            source_timestamp = float(source_timestamp)
            if not math.isfinite(source_timestamp):
                source_timestamp = None
        except (TypeError, ValueError, OverflowError):
            source_timestamp = None
        return {
            "schema": SCHEMA_VERSION,
            "event_type": "candidate_quality_evidence",
            "event_id": (
                f"gcs-v1:{sequence}:{mapper_id}:"
                f"{source_camera_id}"
            ),
            "event_sequence": sequence,
            "status": "ok",
            "reason": "quality_evidence_observed",
            "mode": MODE,
            "observe_only": True,
            "active_topk_applied": False,
            "selected_indices_created": False,
            "mapper_update_id": mapper_id,
            "source_camera_id": source_camera_id,
            "buffer_index": getattr(camera, "buffer_index", None),
            "source_frame_id": getattr(camera, "source_frame_id", None),
            "source_timestamp": source_timestamp,
            "init": bool(init),
            "protected_init": bool(init),
            "coverage_method": COVERAGE_METHOD,
            "coverage_voxel_size": self.voxel_size,
            "raw_candidate_count": int(candidate_count),
            "observed_candidate_count": 0,
            "candidate_finite_count": 0,
            "candidate_nonfinite_count": int(candidate_count),
            "projection_finite_count": 0,
            "positive_camera_z_count": 0,
            "in_bounds_count": 0,
            "source_depth_valid_count": 0,
            "depth_consistent_count": 0,
            "depth_abs_error": self._stats(torch.empty(0)),
            "valid_confidence_count": 0,
            "invalid_confidence_count": int(candidate_count),
            "invalid_confidence_reasons": {},
            "confidence_zero_count": 0,
            "confidence_zero_ratio": None,
            "confidence": self._stats(torch.empty(0)),
            "novel_candidate_count": 0,
            "occupied_candidate_count": 0,
            "candidate_unique_voxel_count": 0,
            "candidate_intra_voxel_duplicate_count": 0,
            "novel_valid_confidence": self._stats(torch.empty(0)),
            "occupied_valid_confidence": self._stats(torch.empty(0)),
            "occupied_voxel_confidence_span": self._stats(torch.empty(0)),
            "counterfactual_sort": COUNTERFACTUAL_SORT,
            "counterfactual_topk": [],
            "candidate_input_state": None,
            "candidate_output_state": None,
            "candidate_input_fingerprint": None,
            "candidate_output_fingerprint": None,
            "candidate_ordering_fingerprint": None,
            "observer_no_mutation": False,
            "all_candidates_forwarded": False,
            "diagnostic_gpu_to_cpu_sync": True,
            "dry_run_wall_ms": None,
            "confidence_source_frame_id": None,
            "confidence_version": None,
            "confidence_up_version": None,
            "confidence_is_current": None,
            "confidence_is_stale": None,
            "confidence_shape": None,
            "confidence_sampling_method": CONFIDENCE_SAMPLING_METHOD,
            "source_resolution": None,
            "confidence_resolution": None,
            "scale_x": None,
            "scale_y": None,
            "confidence_dtype": None,
            "confidence_device": None,
            "confidence_requires_grad": None,
            "gaussian_before": int(gaussian_before),
            "gaussian_after_extend": None,
            "actual_admitted_count": None,
            "actual_dropped_count": None,
            "actual_conservation_pass": None,
            "error": None,
        }

    def _counterfactual_topk(
        self,
        *,
        voxel_ids: torch.Tensor,
        confidence: torch.Tensor,
        original_indices: torch.Tensor,
        novel_count: int,
        protected_count: int,
        raw_count: int,
    ) -> list[dict[str, Any]]:
        if int(original_indices.shape[0]) == 0:
            return [
                {
                    "k": k,
                    "eligible_occupied_count": 0,
                    "counterfactual_selected_occupied_count": 0,
                    "protected_novel_count": novel_count,
                    "protected_invalid_count": protected_count,
                    "estimated_forwarded_count": raw_count,
                    "selected_confidence": self._stats(torch.empty(0)),
                    "selected_original_index_fingerprint": self._index_fingerprint(
                        original_indices
                    ),
                }
                for k in K_VALUES
            ]

        order = torch.arange(
            int(original_indices.shape[0]),
            dtype=torch.long,
            device=original_indices.device,
        )
        order = order[torch.argsort(original_indices[order], stable=True)]
        order = order[torch.argsort(-confidence[order], stable=True)]
        order = order[torch.argsort(voxel_ids[order], stable=True)]
        sorted_voxels = voxel_ids[order]
        group_start = torch.ones_like(sorted_voxels, dtype=torch.bool)
        group_start[1:] = sorted_voxels[1:] != sorted_voxels[:-1]
        group_ids = torch.cumsum(group_start.to(dtype=torch.long), dim=0) - 1
        start_positions = torch.nonzero(group_start, as_tuple=False).flatten()
        ranks = torch.arange(
            int(order.shape[0]),
            dtype=torch.long,
            device=order.device,
        ) - start_positions[group_ids]

        results = []
        for k in K_VALUES:
            counterfactual_mask = ranks < k
            selected_count = int(torch.count_nonzero(counterfactual_mask).item())
            results.append(
                {
                    "k": k,
                    "eligible_occupied_count": int(original_indices.shape[0]),
                    "counterfactual_selected_occupied_count": selected_count,
                    "protected_novel_count": int(novel_count),
                    "protected_invalid_count": int(protected_count),
                    "estimated_forwarded_count": (
                        int(novel_count) + int(protected_count) + selected_count
                    ),
                    "selected_confidence": self._stats(
                        confidence[order][counterfactual_mask]
                    ),
                    "selected_original_index_fingerprint": self._index_fingerprint(
                        original_indices[order][counterfactual_mask]
                    ),
                }
            )
        return results

    def observe_before_extend(
        self,
        *,
        xyz: torch.Tensor,
        features: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        current_gaussian_xyz: torch.Tensor,
        camera: Any,
        depthmap: Any,
        depth_source: str,
        mapper_update_id: int,
        init: bool,
    ) -> CandidateQualityEvidenceToken:
        """Collect evidence and return no candidate tensors or indices."""

        started_ns = time.perf_counter_ns()
        sequence = self._next_event_sequence()
        raw_count = int(xyz.shape[0]) if isinstance(xyz, torch.Tensor) and xyz.ndim >= 1 else 0
        gaussian_before = (
            int(current_gaussian_xyz.shape[0])
            if isinstance(current_gaussian_xyz, torch.Tensor)
            and current_gaussian_xyz.ndim >= 1
            else 0
        )
        fields = self._base_fields(
            sequence=sequence,
            mapper_update_id=mapper_update_id,
            camera=camera,
            init=init,
            candidate_count=raw_count,
            gaussian_before=gaussian_before,
        )
        tensors: Optional[dict[str, torch.Tensor]] = None
        input_state = None
        error: Optional[BaseException] = None
        try:
            raw_count, tensors = self._validate_candidates(
                xyz=xyz,
                features=features,
                scales=scales,
                rotations=rotations,
                opacities=opacities,
            )
            current_gaussian_xyz = self._normalize_current_gaussian_xyz(
                current_gaussian_xyz
            )
            if current_gaussian_xyz.device != xyz.device:
                raise ValueError(
                    "Candidate/current-map device mismatch: "
                    f"candidate={xyz.device}, map={current_gaussian_xyz.device}."
                )
            input_state = self._tensor_state(tensors)
            fields["candidate_input_state"] = input_state
            fields["candidate_input_fingerprint"] = self._fingerprint(input_state)

            if raw_count == 0:
                output_state = self._tensor_state(tensors)
                fields.update(
                    {
                        "candidate_finite_count": 0,
                        "candidate_nonfinite_count": 0,
                        "invalid_confidence_count": 0,
                        "candidate_output_state": output_state,
                        "candidate_output_fingerprint": self._fingerprint(
                            output_state
                        ),
                        "candidate_ordering_fingerprint": self._index_fingerprint(
                            torch.empty(0, dtype=torch.long, device=xyz.device)
                        ),
                        "observer_no_mutation": input_state == output_state,
                        "diagnostic_gpu_to_cpu_sync": False,
                    }
                )
                elapsed_ms = (
                    time.perf_counter_ns() - started_ns
                ) / 1_000_000.0
                fields["dry_run_wall_ms"] = (
                    elapsed_ms if math.isfinite(elapsed_ms) else None
                )
                return CandidateQualityEvidenceToken(fields=fields)

            with torch.no_grad():
                finite_mask = torch.isfinite(xyz).all(dim=1)
                finite_indices = torch.nonzero(finite_mask, as_tuple=False).flatten()
                finite_count = int(finite_indices.shape[0])
                fields["candidate_finite_count"] = finite_count
                fields["candidate_nonfinite_count"] = raw_count - finite_count
                if finite_count != raw_count:
                    fields["invalid_confidence_reasons"]["nonfinite_xyz"] = raw_count - finite_count
                    raise ValueError("Candidate xyz contains NaN or Inf.")

                finite_xyz = xyz[finite_indices]
                current_finite = current_gaussian_xyz[
                    torch.isfinite(current_gaussian_xyz).all(dim=1)
                ]
                candidate_voxels = torch.floor(
                    finite_xyz / self.voxel_size
                ).to(dtype=torch.int64)
                unique_candidate_voxels, candidate_inverse = torch.unique(
                    candidate_voxels,
                    dim=0,
                    return_inverse=True,
                )
                unique_count = int(unique_candidate_voxels.shape[0])
                fields["candidate_unique_voxel_count"] = unique_count
                fields["candidate_intra_voxel_duplicate_count"] = finite_count - unique_count

                if int(current_finite.shape[0]) == 0:
                    occupied_mask = torch.zeros(
                        finite_count,
                        dtype=torch.bool,
                        device=xyz.device,
                    )
                else:
                    existing_voxels = torch.floor(
                        current_finite / self.voxel_size
                    ).to(dtype=torch.int64)
                    combined = torch.cat((existing_voxels, candidate_voxels), dim=0)
                    unique_voxels, inverse = torch.unique(
                        combined,
                        dim=0,
                        return_inverse=True,
                    )
                    occupied_flags = torch.zeros(
                        int(unique_voxels.shape[0]),
                        dtype=torch.bool,
                        device=xyz.device,
                    )
                    occupied_flags[inverse[: int(existing_voxels.shape[0])]] = True
                    occupied_mask = occupied_flags[inverse[int(existing_voxels.shape[0]) :]]
                occupied_count = int(torch.count_nonzero(occupied_mask).item())
                novel_count = finite_count - occupied_count
                fields["occupied_candidate_count"] = occupied_count
                fields["novel_candidate_count"] = novel_count

                u, v, z, projection_valid = self.project_world_to_source_pixels(
                    finite_xyz,
                    camera,
                )
                projection_finite = torch.isfinite(finite_xyz).all(dim=1)
                positive_z = z > 0
                fields["projection_finite_count"] = int(
                    torch.count_nonzero(projection_finite).item()
                )
                fields["positive_camera_z_count"] = int(
                    torch.count_nonzero(projection_finite & positive_z).item()
                )

                source_depth = self._source_depth(
                    camera,
                    depthmap,
                    depth_source,
                    xyz.device,
                    xyz.dtype,
                )
                height, width = (int(source_depth.shape[0]), int(source_depth.shape[1]))
                fields["source_resolution"] = [height, width]
                if height != int(camera.image_height) or width != int(camera.image_width):
                    raise ValueError(
                        "Source depth/Camera image shape mismatch: "
                        f"depth={(height, width)}, camera="
                        f"{(int(camera.image_height), int(camera.image_width))}."
                    )
                in_bounds = (
                    projection_valid
                    & (u >= 0)
                    & (u < width)
                    & (v >= 0)
                    & (v < height)
                )
                in_bounds_count = int(torch.count_nonzero(in_bounds).item())
                fields["in_bounds_count"] = in_bounds_count
                sampled_depth = torch.zeros_like(z)
                if in_bounds_count > 0:
                    sampled_depth[in_bounds] = source_depth[v[in_bounds], u[in_bounds]]
                source_depth_valid = in_bounds & torch.isfinite(sampled_depth) & (sampled_depth > 0)
                fields["source_depth_valid_count"] = int(torch.count_nonzero(source_depth_valid).item())
                depth_error = torch.abs(sampled_depth - z)
                depth_tolerance = 1.0e-3 + 1.0e-3 * torch.abs(z)
                depth_consistent = source_depth_valid & (depth_error <= depth_tolerance)
                depth_consistent_count = int(torch.count_nonzero(depth_consistent).item())
                fields["depth_consistent_count"] = depth_consistent_count
                fields["observed_candidate_count"] = depth_consistent_count
                fields["depth_abs_error"] = self._stats(depth_error[source_depth_valid])
                if depth_consistent_count != raw_count:
                    fields["invalid_confidence_reasons"]["invalid_projection_or_depth_lineage"] = raw_count - depth_consistent_count
                    raise ValueError(
                        "Candidate reprojection did not recover a valid, depth-consistent source pixel for every candidate."
                    )

                snapshot = self._confidence_snapshot_getter(
                    camera,
                    require_current=True,
                    upsampled=False,
                )
                camera_buffer_index = getattr(camera, "buffer_index", None)
                camera_source_frame_id = getattr(camera, "source_frame_id", None)
                camera_timestamp = getattr(camera, "source_timestamp", None)
                if (
                    camera_buffer_index is None
                    or camera_source_frame_id is None
                    or camera_timestamp is None
                ):
                    raise ValueError(
                        "Camera has no complete immutable confidence identity."
                    )
                if int(snapshot.buffer_index) != int(camera_buffer_index):
                    raise ValueError("Confidence buffer identity mismatch.")
                if int(snapshot.source_frame_id) != int(camera_source_frame_id):
                    raise ValueError("Confidence source_frame_id identity mismatch.")
                if float(snapshot.source_timestamp) != float(camera_timestamp):
                    raise ValueError("Confidence timestamp mismatch.")
                if int(snapshot.confidence_source_frame_id) != int(
                    camera_source_frame_id
                ):
                    raise ValueError("Confidence source identity mismatch.")
                if (
                    not bool(snapshot.is_current)
                    or bool(snapshot.is_stale)
                    or int(snapshot.confidence_version) <= 0
                ):
                    raise ValueError("Confidence snapshot is stale.")
                confidence_map = getattr(snapshot, "confidence", None)
                if not isinstance(confidence_map, torch.Tensor):
                    raise ValueError("Confidence snapshot contains no tensor.")
                if confidence_map.ndim != 2:
                    raise ValueError(
                        "Confidence shape mismatch: expected 2 dimensions, "
                        f"got {tuple(confidence_map.shape)}."
                    )
                confidence_height, confidence_width = (
                    int(confidence_map.shape[0]),
                    int(confidence_map.shape[1]),
                )
                fields["confidence_resolution"] = [
                    confidence_height,
                    confidence_width,
                ]
                if confidence_map.device != xyz.device:
                    raise ValueError(
                        "Confidence/candidate device mismatch: "
                        f"confidence={confidence_map.device}, candidate={xyz.device}."
                    )
                if tuple(snapshot.shape) != tuple(confidence_map.shape):
                    raise ValueError("Confidence snapshot shape metadata mismatch.")
                if str(snapshot.dtype) != str(confidence_map.dtype):
                    raise ValueError("Confidence snapshot dtype metadata mismatch.")
                if str(snapshot.device) != str(confidence_map.device):
                    raise ValueError("Confidence snapshot device metadata mismatch.")
                if bool(snapshot.requires_grad) != bool(confidence_map.requires_grad):
                    raise ValueError(
                        "Confidence snapshot requires_grad metadata mismatch."
                    )
                if not bool(torch.isfinite(confidence_map).all().item()):
                    raise ValueError("Confidence snapshot contains NaN or Inf.")
                confidence_u, confidence_v, scale_x, scale_y = (
                    self.map_source_pixels_to_confidence(
                        u,
                        v,
                        source_resolution=(height, width),
                        confidence_resolution=(
                            confidence_height,
                            confidence_width,
                        ),
                    )
                )
                fields["scale_x"] = scale_x
                fields["scale_y"] = scale_y
                sampled_confidence = confidence_map[confidence_v, confidence_u]
                confidence_finite = torch.isfinite(sampled_confidence)
                valid_confidence = depth_consistent & confidence_finite
                valid_count = int(torch.count_nonzero(valid_confidence).item())
                fields["valid_confidence_count"] = valid_count
                fields["invalid_confidence_count"] = raw_count - valid_count
                fields["confidence"] = self._stats(sampled_confidence[valid_confidence])
                zero_count = int(torch.count_nonzero(valid_confidence & (sampled_confidence == 0)).item())
                fields["confidence_zero_count"] = zero_count
                fields["confidence_zero_ratio"] = zero_count / valid_count if valid_count > 0 else None
                fields["confidence_source_frame_id"] = int(snapshot.confidence_source_frame_id)
                fields["confidence_version"] = int(snapshot.confidence_version)
                fields["confidence_up_version"] = int(snapshot.confidence_up_version)
                fields["confidence_is_current"] = bool(snapshot.is_current)
                fields["confidence_is_stale"] = bool(snapshot.is_stale)
                fields["confidence_shape"] = list(snapshot.shape)
                fields["confidence_dtype"] = str(snapshot.dtype)
                fields["confidence_device"] = str(snapshot.device)
                fields["confidence_requires_grad"] = bool(snapshot.requires_grad)

                fields["novel_valid_confidence"] = self._stats(
                    sampled_confidence[valid_confidence & ~occupied_mask]
                )
                fields["occupied_valid_confidence"] = self._stats(
                    sampled_confidence[valid_confidence & occupied_mask]
                )

                eligible = valid_confidence & occupied_mask
                eligible_voxels = candidate_inverse[eligible]
                eligible_confidence = sampled_confidence[eligible]
                eligible_indices = finite_indices[eligible]
                if int(eligible_indices.shape[0]) > 0:
                    unique_eligible, eligible_inverse = torch.unique(
                        eligible_voxels,
                        return_inverse=True,
                    )
                    span_max = torch.full(
                        (int(unique_eligible.shape[0]),),
                        -torch.inf,
                        dtype=eligible_confidence.dtype,
                        device=eligible_confidence.device,
                    )
                    span_min = torch.full(
                        (int(unique_eligible.shape[0]),),
                        torch.inf,
                        dtype=eligible_confidence.dtype,
                        device=eligible_confidence.device,
                    )
                    span_max.scatter_reduce_(0, eligible_inverse, eligible_confidence, reduce="amax", include_self=True)
                    span_min.scatter_reduce_(0, eligible_inverse, eligible_confidence, reduce="amin", include_self=True)
                    fields["occupied_voxel_confidence_span"] = self._stats(span_max - span_min)

                protected_invalid = raw_count - novel_count - int(eligible_indices.shape[0])
                fields["counterfactual_topk"] = self._counterfactual_topk(
                    voxel_ids=eligible_voxels,
                    confidence=eligible_confidence,
                    original_indices=eligible_indices,
                    novel_count=novel_count,
                    protected_count=protected_invalid,
                    raw_count=raw_count,
                )
                ordering_components = {
                    "raw_count": raw_count,
                    "finite_indices": self._index_fingerprint(finite_indices),
                    "eligible_indices": self._index_fingerprint(eligible_indices),
                    "u_index": self._index_fingerprint(u),
                    "v_index": self._index_fingerprint(v),
                    "confidence_u_index": self._index_fingerprint(confidence_u),
                    "confidence_v_index": self._index_fingerprint(confidence_v),
                    "voxel_inverse": self._index_fingerprint(candidate_inverse),
                }
                fields["candidate_ordering_fingerprint"] = self._fingerprint(ordering_components)
                if valid_count != raw_count:
                    fields["invalid_confidence_reasons"]["invalid_confidence_value"] = raw_count - valid_count
                    raise ValueError("Not every candidate has finite current confidence.")
        except Exception as caught:
            error = caught
            if not fields["invalid_confidence_reasons"]:
                reason = self._classify_snapshot_error(caught)
                fields["invalid_confidence_reasons"][reason] = raw_count
            fields.update(
                {
                    "status": "error",
                    "reason": next(iter(fields["invalid_confidence_reasons"])),
                    "error": _structured_error(caught),
                }
            )
        finally:
            if tensors is not None:
                output_state = self._tensor_state(tensors)
                fields["candidate_output_state"] = output_state
                fields["candidate_output_fingerprint"] = self._fingerprint(output_state)
                fields["observer_no_mutation"] = input_state == output_state
                if not fields["observer_no_mutation"] and error is None:
                    fields.update(
                        {
                            "status": "error",
                            "reason": "observer_mutated_candidate_input",
                            "error": {
                                "type": "CandidateMutationError",
                                "message": "Candidate tensor identity or version changed inside observer.",
                            },
                        }
                    )
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
            if math.isfinite(elapsed_ms):
                fields["dry_run_wall_ms"] = elapsed_ms
            else:
                fields.update(
                    {
                        "status": "error",
                        "reason": "nonfinite_dry_run_wall_time",
                        "dry_run_wall_ms": None,
                        "error": {
                            "type": "NonFiniteDryRunWallTimeError",
                            "message": "dry_run_wall_ms was not finite.",
                        },
                    }
                )
        return CandidateQualityEvidenceToken(fields=fields)

    def observe_empty(
        self,
        *,
        current_gaussian_xyz: torch.Tensor,
        camera: Any,
        mapper_update_id: int,
        init: bool,
    ) -> CandidateQualityEvidenceToken:
        sequence = self._next_event_sequence()
        gaussian_before = (
            int(current_gaussian_xyz.shape[0])
            if isinstance(current_gaussian_xyz, torch.Tensor)
            and current_gaussian_xyz.ndim >= 1
            else 0
        )
        fields = self._base_fields(
            sequence=sequence,
            mapper_update_id=mapper_update_id,
            camera=camera,
            init=init,
            candidate_count=0,
            gaussian_before=gaussian_before,
        )
        try:
            self._normalize_current_gaussian_xyz(current_gaussian_xyz)
            fields.update(
                {
                    "reason": "point_cloud_not_emitted_leq5",
                    "observer_no_mutation": True,
                    "dry_run_wall_ms": 0.0,
                    "invalid_confidence_count": 0,
                    "diagnostic_gpu_to_cpu_sync": False,
                }
            )
        except Exception as error:
            fields.update(
                {
                    "status": "error",
                    "reason": "empty_candidate_evidence_error",
                    "dry_run_wall_ms": 0.0,
                    "invalid_confidence_count": 0,
                    "diagnostic_gpu_to_cpu_sync": False,
                    "error": _structured_error(error),
                }
            )
        return CandidateQualityEvidenceToken(fields=fields)

    def record_after_extend(
        self,
        token: CandidateQualityEvidenceToken,
        *,
        admitted_candidate_count: int,
        dropped_candidate_count: int,
        gaussian_after_extend: int,
    ) -> CandidateQualityEvidenceSummary:
        fields = dict(token.fields)
        raw_count = int(fields["raw_candidate_count"])
        admitted = int(admitted_candidate_count)
        dropped = int(dropped_candidate_count)
        gaussian_before = int(fields["gaussian_before"])
        gaussian_after = int(gaussian_after_extend)
        all_forwarded = raw_count == admitted and dropped == 0
        conservation = gaussian_after == gaussian_before + admitted
        fields.update(
            {
                "actual_admitted_count": admitted,
                "actual_dropped_count": dropped,
                "gaussian_after_extend": gaussian_after,
                "all_candidates_forwarded": all_forwarded,
                "actual_conservation_pass": conservation,
            }
        )
        if (not all_forwarded or not conservation) and fields["status"] == "ok":
            fields.update(
                {
                    "status": "error",
                    "reason": "observe_only_forwarding_contract_failed",
                    "error": {
                        "type": "ForwardingContractError",
                        "message": (
                            "GCS-v1 observe mode requires every candidate to "
                            "continue through the unchanged path."
                        ),
                    },
                }
            )
        summary = CandidateQualityEvidenceSummary(fields=fields)
        if self.logging_enabled:
            self._emit(summary.to_event())
        return summary

    @staticmethod
    def _emit(event: dict[str, Any]) -> None:
        try:
            payload = json.dumps(
                event,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except Exception as error:
            fallback = {
                "schema": SCHEMA_VERSION,
                "event_type": "candidate_quality_evidence",
                "event_id": event.get("event_id", "gcs-v1:fallback"),
                "event_sequence": event.get("event_sequence"),
                "status": "error",
                "reason": "event_serialization_failed",
                "mode": MODE,
                "observe_only": True,
                "active_topk_applied": False,
                "selected_indices_created": False,
                "mapper_update_id": event.get("mapper_update_id"),
                "source_camera_id": event.get("source_camera_id"),
                "raw_candidate_count": event.get("raw_candidate_count", 0),
                "actual_admitted_count": event.get("actual_admitted_count", 0),
                "actual_dropped_count": event.get("actual_dropped_count", 0),
                "gaussian_before": event.get("gaussian_before", 0),
                "gaussian_after_extend": event.get("gaussian_after_extend", 0),
                "all_candidates_forwarded": event.get("all_candidates_forwarded", False),
                "observer_no_mutation": event.get("observer_no_mutation", False),
                "error": _structured_error(error),
            }
            payload = json.dumps(
                fallback,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        print(f"{LOG_PREFIX} {payload}", flush=True)
