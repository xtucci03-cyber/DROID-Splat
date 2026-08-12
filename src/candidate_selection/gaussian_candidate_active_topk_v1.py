"""Active fixed-budget quality selection for Gaussian candidates.

This module is intentionally separate from the observe-only GCS-v1 module.
It applies the frozen Quality Top-k600 policy or an exact deterministic M01
fallback, then records an independent active-selection event.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch

from ..resource_management.m01_resource_admission.resource_admission import (
    deterministic_uniform_indices,
)


LOG_PREFIX = "[GaussianCandidateSelectorV1]"
ACTIVE_MODE = "active"
ACTIVE_SCHEMA_VERSION = 1
ACTIVE_BUDGET = 600
CONFIDENCE_SAMPLING_METHOD = "lowres_cell_floor_v1"
_TOP_LEVEL_FIELDS = frozenset({"mode", "budget", "logging"})
_LOGGING_FIELDS = frozenset({"enabled"})

ACTIVE_EVENT_FIELDS = frozenset(
    {
        "schema",
        "event_type",
        "event_id",
        "event_sequence",
        "mapper_update_id",
        "status",
        "reason",
        "mode",
        "source_camera_id",
        "source_frame_id",
        "buffer_index",
        "source_timestamp",
        "evidence_frame_id",
        "confidence_status",
        "confidence_version",
        "confidence_source_frame_id",
        "confidence_sampling_method",
        "source_resolution",
        "confidence_resolution",
        "scale_x",
        "scale_y",
        "input_candidate_count",
        "retained_candidate_count",
        "requested_k",
        "effective_k",
        "selector",
        "init_bypass",
        "fallback",
        "fallback_reason",
        "confidence_min",
        "confidence_max",
        "selected_indices_created",
        "selected_indices_count",
        "selected_indices_fingerprint",
        "conservation_check",
        "m01_second_selection_applied",
        "gaussian_before",
        "gaussian_after_extend",
        "actual_admitted_count",
        "actual_dropped_count",
        "selection_wall_ms",
        "diagnostic_gpu_to_cpu_sync",
        "error",
    }
)


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


def _normalize_active_budget(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            "mapping.candidate_selector_v1.budget must be integer 600 and "
            f"must not be bool, got {value!r}."
        )
    if value != ACTIVE_BUDGET:
        raise ValueError(
            "mapping.candidate_selector_v1.budget is frozen to 600 in v1, "
            f"got {value!r}."
        )
    return int(value)


class ActiveConfidenceEvidenceError(ValueError):
    """A defined evidence/provenance failure eligible for fixed600 fallback."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _read_boolean_statuses(*checks: torch.Tensor) -> tuple[bool, ...]:
    """Read a small, ordered set of scalar CUDA predicates once."""

    if not checks:
        return ()
    for check in checks:
        if not isinstance(check, torch.Tensor) or check.numel() != 1:
            raise RuntimeError("Active status checks must be scalar tensors.")
        if check.dtype != torch.bool:
            raise RuntimeError("Active status checks must have bool dtype.")
    devices = {check.device for check in checks}
    if len(devices) != 1:
        raise RuntimeError("Active status checks must share one device.")
    values = torch.stack(tuple(check.reshape(()) for check in checks))
    return tuple(bool(value) for value in values.detach().cpu().tolist())


def _known_snapshot_failure(error: BaseException) -> Optional[str]:
    """Classify only known snapshot/provenance failures; unknowns propagate."""

    message = str(error).lower()
    if "stale" in message:
        return "stale_confidence"
    if "timestamp mismatch" in message:
        return "confidence_timestamp_mismatch"
    if "identity mismatch" in message or "source_frame_id" in message:
        return "confidence_identity_mismatch"
    if "shape mismatch" in message:
        return "confidence_shape_mismatch"
    if "device mismatch" in message:
        return "confidence_device_mismatch"
    if "dtype mismatch" in message:
        return "confidence_dtype_mismatch"
    if "nan or inf" in message or "non-finite" in message:
        return "nonfinite_confidence"
    if "not valid" in message or "no tensor" in message:
        return "missing_confidence"
    return None


def stable_quality_topk_indices(
    confidence: torch.Tensor,
    *,
    candidate_count: int,
    requested_k: int,
    device: torch.device,
) -> torch.Tensor:
    """Rank by quality stably, then restore original candidate order."""

    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int):
        raise TypeError("candidate_count must be an integer and must not be bool.")
    if candidate_count < 0:
        raise ValueError("candidate_count must be greater than or equal to 0.")
    if isinstance(requested_k, bool) or not isinstance(requested_k, int):
        raise TypeError("requested_k must be an integer and must not be bool.")
    if requested_k < 1:
        raise ValueError("requested_k must be greater than or equal to 1.")
    if not isinstance(confidence, torch.Tensor):
        raise ActiveConfidenceEvidenceError(
            "missing_confidence",
            "Confidence evidence is missing.",
        )
    if confidence.ndim != 1:
        raise ActiveConfidenceEvidenceError(
            "confidence_shape_mismatch",
            "Confidence shape mismatch: expected [N], got "
            f"{tuple(confidence.shape)}.",
        )
    if int(confidence.shape[0]) != candidate_count:
        raise ActiveConfidenceEvidenceError(
            "confidence_length_mismatch",
            "Confidence length mismatch: expected "
            f"{candidate_count}, got {int(confidence.shape[0])}.",
        )
    normalized_device = torch.device(device)
    if confidence.device != normalized_device:
        raise ActiveConfidenceEvidenceError(
            "confidence_device_mismatch",
            "Confidence/candidate device mismatch: confidence="
            f"{confidence.device}, candidate={normalized_device}.",
        )
    if not confidence.dtype.is_floating_point:
        raise ActiveConfidenceEvidenceError(
            "confidence_dtype_mismatch",
            "Confidence dtype mismatch: expected floating point, got "
            f"{confidence.dtype}.",
        )
    (confidence_is_finite,) = _read_boolean_statuses(
        torch.isfinite(confidence).all()
    )
    if not confidence_is_finite:
        raise ActiveConfidenceEvidenceError(
            "nonfinite_confidence",
            "Confidence contains NaN or Inf.",
        )

    effective_k = min(candidate_count, requested_k)
    if effective_k == candidate_count:
        return torch.arange(
            candidate_count,
            dtype=torch.long,
            device=normalized_device,
        )
    with torch.no_grad():
        ranked = torch.argsort(-confidence.detach(), stable=True)
        winners = ranked[:effective_k]
        return torch.sort(winners).values


def build_gaussian_candidate_active_topk_v1(
    config: Mapping[str, Any],
    *,
    candidate_observer: Any,
    resource_admission_mode: str,
    confidence_snapshot_getter: Callable[..., Any],
    device: Any,
) -> "GaussianCandidateActiveTopKV1":
    config = _require_mapping(config, "mapping.candidate_selector_v1")
    _reject_unknown_fields(config, _TOP_LEVEL_FIELDS, "mapping.candidate_selector_v1")
    if str(config.get("mode", "off")).strip().lower() != ACTIVE_MODE:
        raise ValueError("Active Top-k builder requires mode='active'.")
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
    if not logging_enabled:
        raise ValueError(
            "mapping.candidate_selector_v1.logging.enabled must be true "
            "in active mode."
        )
    if str(resource_admission_mode).strip().lower() != "disabled":
        raise ValueError(
            "Gaussian Candidate Selector v1 active mode requires M01 "
            "ResourceAdmission to be disabled so no second budget selection "
            f"can occur; got {resource_admission_mode!r}."
        )
    if candidate_observer is not None:
        raise ValueError(
            "Gaussian Candidate Selector v1 active mode requires "
            "mapping.candidate_observer.mode=off so the GCO observe-only "
            "forwarding contract is not polluted by active selection."
        )
    if not callable(confidence_snapshot_getter):
        raise TypeError("confidence_snapshot_getter must be callable.")
    return GaussianCandidateActiveTopKV1(
        confidence_snapshot_getter=confidence_snapshot_getter,
        logging_enabled=logging_enabled,
        device=device,
        active_budget=_normalize_active_budget(
            config.get("budget", ACTIVE_BUDGET)
        ),
    )


@dataclass(frozen=True)
class CandidateActiveSelectionToken:
    fields: dict[str, Any]


@dataclass(frozen=True)
class CandidateActiveSelectionResult:
    xyz: torch.Tensor
    features: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor
    selected_indices: Optional[torch.Tensor]
    token: CandidateActiveSelectionToken

    @property
    def retained_candidate_count(self) -> int:
        return int(self.xyz.shape[0])


@dataclass(frozen=True)
class CandidateActiveSelectionSummary:
    fields: dict[str, Any]

    def to_event(self) -> dict[str, Any]:
        return dict(self.fields)


class GaussianCandidateActiveTopKV1:
    """Apply one isolated active selection before persistent insertion."""

    def __init__(
        self,
        *,
        confidence_snapshot_getter: Callable[..., Any],
        logging_enabled: bool,
        device: Any,
        active_budget: int = ACTIVE_BUDGET,
    ) -> None:
        if not callable(confidence_snapshot_getter):
            raise TypeError("confidence_snapshot_getter must be callable.")
        self._confidence_snapshot_getter = confidence_snapshot_getter
        self.logging_enabled = _require_bool(
            logging_enabled,
            "mapping.candidate_selector_v1.logging.enabled",
        )
        self.device = torch.device(device)
        self.active_budget = _normalize_active_budget(active_budget)
        self._event_sequence = 0

    @property
    def is_active(self) -> bool:
        return True

    def _next_event_sequence(self) -> int:
        sequence = self._event_sequence
        self._event_sequence += 1
        return sequence

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
            raise ValueError(
                f"Candidate xyz must have shape [N,3], got {list(xyz.shape)}."
            )
        count = int(xyz.shape[0])
        shapes = {name: list(value.shape) for name, value in tensors.items()}
        if any(
            value.ndim < 1 or int(value.shape[0]) != count
            for value in tensors.values()
        ):
            raise ValueError(
                "Candidate first-dimension mismatch: "
                f"candidate_count={count}, shapes={shapes}."
            )
        devices = {name: str(value.device) for name, value in tensors.items()}
        if any(value.device != xyz.device for value in tensors.values()):
            raise ValueError(f"Candidate device mismatch: devices={devices}.")
        return count, tensors

    @classmethod
    def _validate_active_candidate_contract(
        cls,
        *,
        xyz: torch.Tensor,
        features: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
    ) -> tuple[int, dict[str, torch.Tensor]]:
        count, tensors = cls._validate_candidates(
            xyz=xyz,
            features=features,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
        )
        expected = {
            "features": features.ndim == 3 and int(features.shape[1]) == 3,
            "scales": scales.ndim == 2 and int(scales.shape[1]) in {1, 3},
            "rotations": rotations.ndim == 2 and int(rotations.shape[1]) == 4,
            "opacities": opacities.ndim == 2 and int(opacities.shape[1]) == 1,
        }
        invalid = [name for name, valid in expected.items() if not valid]
        if invalid:
            shapes = {name: list(value.shape) for name, value in tensors.items()}
            raise ValueError(
                "Active candidate field shape contract violation: "
                f"invalid={invalid}, shapes={shapes}."
            )
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
                    torch.tensor(
                        indices.shape[0],
                        dtype=torch.int64,
                        device=indices.device,
                    ),
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
        minimum, maximum, mean = [
            float(value) for value in aggregate.cpu().tolist()
        ]
        if not all(math.isfinite(value) for value in (minimum, maximum, mean)):
            raise RuntimeError("Active confidence statistics are non-finite.")
        return {
            "count": int(values.shape[0]),
            "min": minimum,
            "max": maximum,
            "mean": mean,
        }

    @staticmethod
    def _camera_intrinsics(
        camera: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, Optional[tuple[float, float, float, float]]]:
        values = (camera.fx, camera.fy, camera.cx, camera.cy)
        names = ("fx", "fy", "cx", "cy")
        has_non_cpu_tensor = any(
            isinstance(value, torch.Tensor) and value.device.type != "cpu"
            for value in values
        )

        if not has_non_cpu_tensor:
            cpu_scalars = []
            for value, name in zip(values, names):
                if isinstance(value, torch.Tensor):
                    if value.numel() != 1:
                        raise ActiveConfidenceEvidenceError(
                            "invalid_camera_projection",
                            f"Camera {name} must be one finite scalar.",
                        )
                    scalar = torch.as_tensor(
                        value.detach(),
                        device="cpu",
                        dtype=dtype,
                    ).reshape(())
                else:
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError, OverflowError) as error:
                        raise ActiveConfidenceEvidenceError(
                            "invalid_camera_projection",
                            f"Camera {name} must be one finite scalar.",
                        ) from error
                    scalar = torch.tensor(numeric, device="cpu", dtype=dtype)
                cpu_scalars.append(scalar)
            cpu_intrinsics = torch.stack(cpu_scalars)
            normalized_values = tuple(
                float(value) for value in cpu_intrinsics.tolist()
            )
            for value, name in zip(normalized_values, names):
                if not math.isfinite(value):
                    raise ActiveConfidenceEvidenceError(
                        "invalid_camera_projection",
                        f"Camera {name} must be one finite scalar.",
                    )
            if normalized_values[0] <= 0 or normalized_values[1] <= 0:
                raise ActiveConfidenceEvidenceError(
                    "invalid_camera_projection",
                    "Camera fx and fy must be positive.",
                )
            return (
                cpu_intrinsics.to(device=device),
                normalized_values,
            )

        scalars = []
        for value, name in zip(values, names):
            scalar = torch.as_tensor(value, device=device, dtype=dtype)
            if scalar.numel() != 1:
                raise ActiveConfidenceEvidenceError(
                    "invalid_camera_projection",
                    f"Camera {name} must be one finite scalar.",
                )
            scalars.append(scalar.reshape(()))
        return torch.stack(scalars), None

    @staticmethod
    def _source_depth(
        camera: Any,
        depthmap: Any,
        depth_source: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if depth_source == "explicit_depthmap":
            source = depthmap
        elif depth_source == "estimated_clean_depth":
            source = getattr(camera, "depth", None)
        elif depth_source == "depth_prior":
            source = getattr(camera, "depth_prior", None)
        else:
            raise ActiveConfidenceEvidenceError(
                "source_depth_unavailable",
                "Source depth cannot be reconstructed exactly for "
                f"depth_source={depth_source!r}.",
            )
        if source is None:
            raise ActiveConfidenceEvidenceError(
                "source_depth_unavailable",
                f"Source depth is missing for depth_source={depth_source!r}.",
            )
        if isinstance(source, torch.Tensor):
            result = source.detach().to(device=device, dtype=dtype)
        else:
            result = torch.as_tensor(
                np.asarray(source),
                device=device,
                dtype=dtype,
            )
        if result.ndim != 2:
            raise ActiveConfidenceEvidenceError(
                "source_depth_shape_mismatch",
                f"Source depth must have shape [H,W], got {list(result.shape)}.",
            )
        return result

    def project_world_to_source_pixels(
        self,
        xyz: torch.Tensor,
        camera: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(xyz, torch.Tensor) or xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape [N,3].")
        pose_w2c = getattr(camera, "pose", None)
        if not isinstance(pose_w2c, torch.Tensor) or tuple(pose_w2c.shape) != (4, 4):
            raise ActiveConfidenceEvidenceError(
                "invalid_camera_projection",
                "Camera pose must be a [4,4] W2C tensor.",
            )
        pose_w2c = pose_w2c.detach().to(device=xyz.device, dtype=xyz.dtype)
        (pose_is_finite,) = _read_boolean_statuses(
            torch.isfinite(pose_w2c).all()
        )
        if not pose_is_finite:
            raise ActiveConfidenceEvidenceError(
                "invalid_camera_projection",
                "Camera W2C pose contains NaN or Inf.",
            )
        intrinsics, host_intrinsics = self._camera_intrinsics(
            camera,
            xyz.device,
            xyz.dtype,
        )
        if host_intrinsics is not None:
            intrinsic_finite_values = (True, True, True, True)
            focal_lengths_are_positive = True
        else:
            statuses = _read_boolean_statuses(
                *(torch.isfinite(value) for value in intrinsics.unbind()),
                (intrinsics[:2] > 0).all(),
            )
            intrinsic_finite_values = statuses[:4]
            focal_lengths_are_positive = statuses[4]
        for is_finite, name in zip(
            intrinsic_finite_values,
            ("fx", "fy", "cx", "cy"),
        ):
            if not is_finite:
                raise ActiveConfidenceEvidenceError(
                    "invalid_camera_projection",
                    f"Camera {name} must be one finite scalar.",
                )
        if not focal_lengths_are_positive:
            raise ActiveConfidenceEvidenceError(
                "invalid_camera_projection",
                "Camera fx and fy must be positive.",
            )
        camera_xyz = xyz @ pose_w2c[:3, :3].transpose(0, 1) + pose_w2c[:3, 3]
        z = camera_xyz[:, 2]
        projection_finite = torch.isfinite(camera_xyz).all(dim=1)
        positive_z = z > 0
        fx, fy, cx, cy = intrinsics.unbind()
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
    def map_source_pixels_to_confidence(
        u: torch.Tensor,
        v: torch.Tensor,
        *,
        source_resolution: tuple[int, int],
        confidence_resolution: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        if u.shape != v.shape:
            raise RuntimeError("Internal source pixel coordinate shape mismatch.")
        source_height, source_width = source_resolution
        confidence_height, confidence_width = confidence_resolution
        if min(
            source_height,
            source_width,
            confidence_height,
            confidence_width,
        ) <= 0:
            raise ActiveConfidenceEvidenceError(
                "confidence_shape_mismatch",
                "Confidence/source resolution must be positive.",
            )
        source_in_bounds = (
            (u >= 0)
            & (u < source_width)
            & (v >= 0)
            & (v < source_height)
        )
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
        source_bounds_valid, confidence_bounds_valid = _read_boolean_statuses(
            torch.all(source_in_bounds),
            torch.all(confidence_in_bounds),
        )
        if not source_bounds_valid:
            raise ActiveConfidenceEvidenceError(
                "invalid_candidate_projection",
                "Source pixel coordinate is out of bounds.",
            )
        if not confidence_bounds_valid:
            raise ActiveConfidenceEvidenceError(
                "confidence_shape_mismatch",
                "Mapped confidence coordinate is out of bounds.",
            )
        return (
            confidence_u,
            confidence_v,
            source_width / confidence_width,
            source_height / confidence_height,
        )

    def _read_active_confidence(
        self,
        *,
        xyz: torch.Tensor,
        camera: Any,
        depthmap: Any,
        depth_source: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        source_frame_id = getattr(camera, "source_frame_id", None)
        source_timestamp = getattr(camera, "source_timestamp", None)
        buffer_index = getattr(camera, "buffer_index", None)
        if source_frame_id is None or source_timestamp is None or buffer_index is None:
            raise ActiveConfidenceEvidenceError(
                "missing_confidence",
                "Camera has no complete immutable confidence identity.",
            )

        u, v, z, projection_valid = self.project_world_to_source_pixels(xyz, camera)
        source_depth = self._source_depth(
            camera,
            depthmap,
            depth_source,
            xyz.device,
            xyz.dtype,
        )
        height, width = int(source_depth.shape[0]), int(source_depth.shape[1])
        if height != int(camera.image_height) or width != int(camera.image_width):
            raise ActiveConfidenceEvidenceError(
                "source_depth_camera_shape_mismatch",
                "Source depth/Camera image shape mismatch: "
                f"depth={(height, width)}, camera="
                f"{(int(camera.image_height), int(camera.image_width))}.",
            )
        in_bounds = (
            projection_valid
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )
        (candidate_projection_is_valid,) = _read_boolean_statuses(
            torch.all(in_bounds)
        )
        if not candidate_projection_is_valid:
            raise ActiveConfidenceEvidenceError(
                "invalid_candidate_projection",
                "Candidate source projection is invalid or out of bounds.",
            )
        sampled_depth = source_depth[v, u]
        depth_valid = torch.isfinite(sampled_depth) & (sampled_depth > 0)
        depth_error = torch.abs(sampled_depth - z)
        depth_tolerance = 1.0e-3 + 1.0e-3 * torch.abs(z)
        (depth_lineage_is_valid,) = _read_boolean_statuses(
            torch.all(depth_valid & (depth_error <= depth_tolerance))
        )
        if not depth_lineage_is_valid:
            raise ActiveConfidenceEvidenceError(
                "source_depth_lineage_mismatch",
                "Candidate source depth lineage is invalid or inconsistent.",
            )

        try:
            snapshot = self._confidence_snapshot_getter(
                camera,
                require_current=True,
                upsampled=False,
            )
        except Exception as error:
            reason = _known_snapshot_failure(error)
            if reason is None:
                raise
            raise ActiveConfidenceEvidenceError(reason, str(error)) from error

        if int(snapshot.buffer_index) != int(buffer_index):
            raise ActiveConfidenceEvidenceError(
                "confidence_identity_mismatch",
                "Confidence buffer identity mismatch.",
            )
        if int(snapshot.source_frame_id) != int(source_frame_id):
            raise ActiveConfidenceEvidenceError(
                "confidence_identity_mismatch",
                "Confidence source_frame_id identity mismatch.",
            )
        if float(snapshot.source_timestamp) != float(source_timestamp):
            raise ActiveConfidenceEvidenceError(
                "confidence_timestamp_mismatch",
                "Confidence timestamp mismatch.",
            )
        if int(snapshot.confidence_source_frame_id) != int(source_frame_id):
            raise ActiveConfidenceEvidenceError(
                "confidence_identity_mismatch",
                "Confidence source identity mismatch.",
            )
        if (
            not bool(snapshot.is_current)
            or bool(snapshot.is_stale)
            or int(snapshot.confidence_version) <= 0
        ):
            raise ActiveConfidenceEvidenceError(
                "stale_confidence",
                "Confidence snapshot is stale.",
            )

        confidence_map = getattr(snapshot, "confidence", None)
        if not isinstance(confidence_map, torch.Tensor):
            raise ActiveConfidenceEvidenceError(
                "missing_confidence",
                "Confidence snapshot contains no tensor.",
            )
        if confidence_map.ndim != 2:
            raise ActiveConfidenceEvidenceError(
                "confidence_shape_mismatch",
                "Confidence shape mismatch: expected 2 dimensions, got "
                f"{tuple(confidence_map.shape)}.",
            )
        if confidence_map.device != xyz.device:
            raise ActiveConfidenceEvidenceError(
                "confidence_device_mismatch",
                "Confidence/candidate device mismatch: confidence="
                f"{confidence_map.device}, candidate={xyz.device}.",
            )
        if not confidence_map.dtype.is_floating_point:
            raise ActiveConfidenceEvidenceError(
                "confidence_dtype_mismatch",
                "Confidence dtype mismatch: expected floating point, got "
                f"{confidence_map.dtype}.",
            )
        if tuple(snapshot.shape) != tuple(confidence_map.shape):
            raise ActiveConfidenceEvidenceError(
                "confidence_shape_mismatch",
                "Confidence snapshot shape metadata mismatch.",
            )
        if str(snapshot.dtype) != str(confidence_map.dtype):
            raise ActiveConfidenceEvidenceError(
                "confidence_dtype_mismatch",
                "Confidence snapshot dtype metadata mismatch.",
            )
        if str(snapshot.device) != str(confidence_map.device):
            raise ActiveConfidenceEvidenceError(
                "confidence_device_mismatch",
                "Confidence snapshot device metadata mismatch.",
            )
        if bool(snapshot.requires_grad) != bool(confidence_map.requires_grad):
            raise ActiveConfidenceEvidenceError(
                "confidence_requires_grad_mismatch",
                "Confidence snapshot requires_grad metadata mismatch.",
            )
        (confidence_map_is_finite,) = _read_boolean_statuses(
            torch.isfinite(confidence_map).all()
        )
        if not confidence_map_is_finite:
            raise ActiveConfidenceEvidenceError(
                "nonfinite_confidence",
                "Confidence snapshot contains NaN or Inf.",
            )

        confidence_height = int(confidence_map.shape[0])
        confidence_width = int(confidence_map.shape[1])
        confidence_u, confidence_v, scale_x, scale_y = (
            self.map_source_pixels_to_confidence(
                u,
                v,
                source_resolution=(height, width),
                confidence_resolution=(confidence_height, confidence_width),
            )
        )
        sampled = confidence_map[confidence_v, confidence_u]
        return sampled, {
            "evidence_frame_id": int(snapshot.confidence_source_frame_id),
            "confidence_status": "current",
            "confidence_version": int(snapshot.confidence_version),
            "confidence_source_frame_id": int(snapshot.confidence_source_frame_id),
            "confidence_sampling_method": CONFIDENCE_SAMPLING_METHOD,
            "source_resolution": [height, width],
            "confidence_resolution": [confidence_height, confidence_width],
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
        }

    def _active_base_fields(
        self,
        *,
        sequence: int,
        camera: Any,
        mapper_update_id: int,
        init: bool,
        candidate_count: int,
        gaussian_before: int,
    ) -> dict[str, Any]:
        source_frame_id = getattr(camera, "source_frame_id", None)
        source_camera_id = getattr(camera, "uid", None)
        buffer_index = getattr(camera, "buffer_index", None)
        source_timestamp = getattr(camera, "source_timestamp", None)
        try:
            source_timestamp = float(source_timestamp)
            if not math.isfinite(source_timestamp):
                source_timestamp = None
        except (TypeError, ValueError, OverflowError):
            source_timestamp = None
        fields = {
            "schema": ACTIVE_SCHEMA_VERSION,
            "event_type": "candidate_active_selection",
            "event_id": (
                f"gcs-v1-active:{sequence}:{mapper_update_id}:"
                f"{source_camera_id}"
            ),
            "event_sequence": int(sequence),
            "mapper_update_id": int(mapper_update_id),
            "status": "ok",
            "reason": None,
            "mode": ACTIVE_MODE,
            "source_camera_id": (
                int(source_camera_id) if source_camera_id is not None else None
            ),
            "source_frame_id": (
                int(source_frame_id) if source_frame_id is not None else None
            ),
            "buffer_index": int(buffer_index) if buffer_index is not None else None,
            "source_timestamp": source_timestamp,
            "evidence_frame_id": None,
            "confidence_status": "not_read",
            "confidence_version": None,
            "confidence_source_frame_id": None,
            "confidence_sampling_method": None,
            "source_resolution": None,
            "confidence_resolution": None,
            "scale_x": None,
            "scale_y": None,
            "input_candidate_count": int(candidate_count),
            "retained_candidate_count": int(candidate_count),
            "requested_k": int(self.active_budget),
            "effective_k": min(int(candidate_count), int(self.active_budget)),
            "selector": None,
            "init_bypass": bool(init),
            "fallback": False,
            "fallback_reason": None,
            "confidence_min": None,
            "confidence_max": None,
            "selected_indices_created": False,
            "selected_indices_count": 0,
            "selected_indices_fingerprint": None,
            "conservation_check": True,
            "m01_second_selection_applied": False,
            "gaussian_before": int(gaussian_before),
            "gaussian_after_extend": None,
            "actual_admitted_count": None,
            "actual_dropped_count": None,
            "selection_wall_ms": None,
            "diagnostic_gpu_to_cpu_sync": False,
            "error": None,
        }
        if frozenset(fields) != ACTIVE_EVENT_FIELDS:
            raise RuntimeError("Internal active event schema is incomplete.")
        return fields

    @staticmethod
    def _index_select_candidates(
        tensors: Mapping[str, torch.Tensor],
        selected_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            name: torch.index_select(value, 0, selected_indices)
            for name, value in tensors.items()
        }

    def select_before_extend(
        self,
        *,
        xyz: torch.Tensor,
        features: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        camera: Any,
        depthmap: Any,
        depth_source: str,
        mapper_update_id: int,
        init: bool,
        gaussian_before: int,
        preinsert_render_evidence: Any = None,
    ) -> CandidateActiveSelectionResult:
        # Reserved for GCS-v2.  Active v1 deliberately does not read render
        # values, so confidence ranking and selected indices remain frozen.
        started_ns = time.perf_counter_ns()
        candidate_count, tensors = self._validate_active_candidate_contract(
            xyz=xyz,
            features=features,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
        )
        input_state = self._tensor_state(tensors)
        fields = self._active_base_fields(
            sequence=self._next_event_sequence(),
            camera=camera,
            mapper_update_id=mapper_update_id,
            init=init,
            candidate_count=candidate_count,
            gaussian_before=gaussian_before,
        )
        selected_indices: Optional[torch.Tensor] = None
        output_tensors = tensors

        if init:
            fields.update(
                {
                    "reason": "init_bypass",
                    "selector": "init_bypass",
                    "effective_k": candidate_count,
                    "confidence_status": "not_read_init_bypass",
                }
            )
        elif candidate_count <= self.active_budget:
            fields.update(
                {
                    "reason": "keep_all_n_le_k",
                    "selector": "keep_all_n_le_k",
                    "effective_k": candidate_count,
                    "confidence_status": "not_read_n_le_k",
                }
            )
        else:
            try:
                confidence, metadata = self._read_active_confidence(
                    xyz=xyz,
                    camera=camera,
                    depthmap=depthmap,
                    depth_source=depth_source,
                )
                selected_indices = stable_quality_topk_indices(
                    confidence,
                    candidate_count=candidate_count,
                    requested_k=self.active_budget,
                    device=xyz.device,
                )
            except ActiveConfidenceEvidenceError as error:
                selected_indices = deterministic_uniform_indices(
                    candidate_count=candidate_count,
                    fixed_budget=self.active_budget,
                    device=xyz.device,
                )
                fields.update(
                    {
                        "reason": "fixed600_fallback_applied",
                        "selector": "fixed600_fallback",
                        "fallback": True,
                        "fallback_reason": {
                            "code": error.reason,
                            "error": _structured_error(error),
                        },
                        "confidence_status": "invalid_fallback",
                    }
                )
            else:
                confidence_stats = self._stats(confidence)
                fields.update(metadata)
                fields.update(
                    {
                        "reason": "quality_topk_applied",
                        "selector": "quality_topk",
                        "confidence_min": confidence_stats["min"],
                        "confidence_max": confidence_stats["max"],
                        "diagnostic_gpu_to_cpu_sync": True,
                    }
                )

            output_tensors = self._index_select_candidates(
                tensors,
                selected_indices,
            )
            fields.update(
                {
                    "retained_candidate_count": int(selected_indices.shape[0]),
                    "effective_k": int(selected_indices.shape[0]),
                    "selected_indices_created": True,
                    "selected_indices_count": int(selected_indices.shape[0]),
                    "selected_indices_fingerprint": self._index_fingerprint(
                        selected_indices
                    ),
                    "diagnostic_gpu_to_cpu_sync": True,
                }
            )

        if self._tensor_state(tensors) != input_state:
            raise RuntimeError("Active selector mutated an input candidate tensor.")
        retained_counts = {
            name: int(value.shape[0]) for name, value in output_tensors.items()
        }
        retained = int(output_tensors["xyz"].shape[0])
        if any(count != retained for count in retained_counts.values()):
            raise RuntimeError(
                "Active selected candidate fields are misaligned: "
                f"{retained_counts}."
            )
        fields["retained_candidate_count"] = retained
        fields["conservation_check"] = (
            candidate_count - retained >= 0
            and candidate_count == retained + (candidate_count - retained)
        )
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        fields["selection_wall_ms"] = (
            elapsed_ms if math.isfinite(elapsed_ms) else None
        )
        return CandidateActiveSelectionResult(
            xyz=output_tensors["xyz"],
            features=output_tensors["features"],
            scales=output_tensors["scales"],
            rotations=output_tensors["rotations"],
            opacities=output_tensors["opacities"],
            selected_indices=selected_indices,
            token=CandidateActiveSelectionToken(fields=fields),
        )

    def observe_active_empty(
        self,
        *,
        camera: Any,
        mapper_update_id: int,
        init: bool,
        gaussian_before: int,
    ) -> CandidateActiveSelectionToken:
        fields = self._active_base_fields(
            sequence=self._next_event_sequence(),
            camera=camera,
            mapper_update_id=mapper_update_id,
            init=init,
            candidate_count=0,
            gaussian_before=gaussian_before,
        )
        fields.update(
            {
                "reason": "empty_candidate_bypass",
                "selector": "empty_candidate_bypass",
                "effective_k": 0,
                "confidence_status": "not_read_empty",
                "selection_wall_ms": 0.0,
            }
        )
        return CandidateActiveSelectionToken(fields=fields)

    def record_active_after_extend(
        self,
        token: CandidateActiveSelectionToken,
        *,
        admitted_candidate_count: int,
        dropped_candidate_count: int,
        gaussian_after_extend: int,
        m01_second_selection_applied: bool,
    ) -> CandidateActiveSelectionSummary:
        fields = dict(token.fields)
        retained = int(fields["retained_candidate_count"])
        admitted = int(admitted_candidate_count)
        gaussian_before = int(fields["gaussian_before"])
        gaussian_after = int(gaussian_after_extend)
        second_selection = bool(m01_second_selection_applied)
        conservation = (
            bool(fields["conservation_check"])
            and admitted == retained
            and int(dropped_candidate_count) == 0
            and gaussian_after == gaussian_before + admitted
            and not second_selection
        )
        fields.update(
            {
                "actual_admitted_count": admitted,
                "actual_dropped_count": (
                    int(fields["input_candidate_count"]) - admitted
                ),
                "gaussian_after_extend": gaussian_after,
                "m01_second_selection_applied": second_selection,
                "conservation_check": conservation,
            }
        )
        if frozenset(fields) != ACTIVE_EVENT_FIELDS:
            raise RuntimeError("Final active event schema is inconsistent.")
        if not conservation:
            fields.update(
                {
                    "status": "error",
                    "reason": "active_selection_conservation_failed",
                    "error": {
                        "type": "ActiveSelectionConservationError",
                        "message": (
                            "Active selection, M01 bypass, or Gaussian count "
                            "conservation failed."
                        ),
                    },
                }
            )
        summary = CandidateActiveSelectionSummary(fields=fields)
        if self.logging_enabled:
            self._emit_active(summary.to_event())
        return summary

    @staticmethod
    def _emit_active(event: dict[str, Any]) -> None:
        try:
            payload = json.dumps(
                event,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except Exception as error:
            payload = json.dumps(
                {
                    "schema": ACTIVE_SCHEMA_VERSION,
                    "event_type": "candidate_active_selection",
                    "event_id": event.get("event_id", "gcs-v1-active:fallback"),
                    "status": "error",
                    "reason": "active_event_serialization_failed",
                    "mode": ACTIVE_MODE,
                    "input_candidate_count": event.get(
                        "input_candidate_count", 0
                    ),
                    "retained_candidate_count": event.get(
                        "retained_candidate_count", 0
                    ),
                    "requested_k": event.get("requested_k", ACTIVE_BUDGET),
                    "fallback": event.get("fallback", False),
                    "error": _structured_error(error),
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        print(f"{LOG_PREFIX} {payload}", flush=True)
