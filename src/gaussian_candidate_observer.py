"""Observe Gaussian candidates without changing admission or map insertion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from numbers import Real
import time
from typing import Any, Mapping, Optional

import numpy as np
import torch


LOG_PREFIX = "[GaussianCandidateObserver]"
SCHEMA_VERSION = 1
COVERAGE_METHOD = "voxel_occupancy"
SUPPORTED_MODES = frozenset({"off", "observe"})
DEPTH_SOURCES = frozenset(
    {
        "explicit_depthmap",
        "estimated_clean_depth",
        "depth_prior",
        "random_initialization",
        "neighbor_scale_fallback",
        "unknown",
    }
)

_TOP_LEVEL_FIELDS = frozenset(
    {
        "mode",
        "coverage",
        "timing",
        "memory",
        "logging",
    }
)
_COVERAGE_FIELDS = frozenset({"method", "voxel_size"})
_TIMING_FIELDS = frozenset({"gpu"})
_MEMORY_FIELDS = frozenset({"enabled"})
_LOGGING_FIELDS = frozenset({"enabled"})
_EVENT_FLOAT_FIELDS = (
    "valid_depth_ratio",
    "coverage_voxel_size",
    "occupied_ratio",
    "novel_ratio",
    "spatial_extent_x",
    "spatial_extent_y",
    "spatial_extent_z",
    "observer_cpu_ms",
    "observer_gpu_ms",
)
_NULLABLE_EVENT_FLOAT_FIELDS = frozenset(
    {
        "valid_depth_ratio",
        "occupied_ratio",
        "novel_ratio",
        "spatial_extent_x",
        "spatial_extent_y",
        "spatial_extent_z",
        "observer_gpu_ms",
    }
)


def _structured_error(error: BaseException) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message": str(error),
    }


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{field} must be a mapping, got {type(value).__name__}."
        )
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


def _normalize_voxel_size(value: Any, *, required: bool) -> Optional[float]:
    if value is None:
        if required:
            raise ValueError(
                "mapping.candidate_observer.coverage.voxel_size must be "
                "explicitly provided in observe mode."
            )
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            "mapping.candidate_observer.coverage.voxel_size must be a "
            f"positive finite number and bool is not accepted, got {value!r}."
        )
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(
            "mapping.candidate_observer.coverage.voxel_size must be a "
            f"positive finite number, got {value!r}."
        )
    return normalized


@dataclass(frozen=True)
class CandidateGenerationMetadata:
    """Small CPU-side facts collected from the already-materialized depth/PCD."""

    depth_source: str
    depth_pixel_count: int
    valid_depth_count: int
    pre_downsample_point_count: int
    post_downsample_point_count: int
    metadata_cpu_ns: int = 0
    error: Optional[dict[str, str]] = None


@dataclass(frozen=True)
class CandidateGenerationResult:
    """Observe-only sidecar around the original optional five-tensor result."""

    candidates: Optional[tuple[torch.Tensor, ...]]
    metadata: CandidateGenerationMetadata


@dataclass(frozen=True)
class CandidateObservationToken:
    """Internal immutable state carried across the original map extension."""

    fields: dict[str, Any]


@dataclass(frozen=True)
class CandidateEvidenceSummary:
    """One Camera-level evidence record; it never contains selection decisions."""

    schema: int
    event_type: str
    event_id: str
    status: str
    reason: str
    mapper_update_id: int
    source_camera_id: int
    init: bool
    observer_mode: str
    gpu_timing_enabled: bool
    memory_observation_enabled: bool
    depth_source: str
    depth_pixel_count: int
    valid_depth_count: int
    valid_depth_ratio: Optional[float]
    pre_downsample_point_count: int
    post_downsample_point_count: int
    candidate_3d_count: int
    candidate_finite_count: int
    candidate_nonfinite_count: int
    existing_gaussian_count: int
    coverage_method: str
    coverage_voxel_size: float
    candidate_unique_voxel_count: int
    candidate_intra_voxel_duplicate_count: int
    occupied_candidate_count: int
    novel_candidate_count: int
    occupied_ratio: Optional[float]
    novel_ratio: Optional[float]
    spatial_extent_x: Optional[float]
    spatial_extent_y: Optional[float]
    spatial_extent_z: Optional[float]
    observer_cpu_ms: float
    observer_gpu_ms: Optional[float]
    allocated_before_bytes: Optional[int]
    allocated_after_bytes: Optional[int]
    reserved_before_bytes: Optional[int]
    reserved_after_bytes: Optional[int]
    gaussian_before: int
    gaussian_after_extend: int
    admitted_candidate_count: int
    dropped_candidate_count: int
    empty_candidate: bool
    all_candidates_admitted: bool
    conservation_pass: bool
    error: Optional[dict[str, str]]

    def to_event(self) -> dict[str, Any]:
        return asdict(self)


def collect_candidate_generation_metadata(
    *,
    depth: Any,
    depth_source: str,
    pre_downsample_point_count: int,
    post_downsample_point_count: int,
    depth_trunc: float,
) -> CandidateGenerationMetadata:
    """Collect metadata from the CPU depth already consumed by Open3D.

    This helper must only be called when observe mode is active.
    """

    started_ns = time.perf_counter_ns()
    try:
        normalized_source = (
            depth_source if depth_source in DEPTH_SOURCES else "unknown"
        )
        depth_array = np.asarray(depth)
        depth_pixel_count = int(depth_array.size)
        valid_depth = (
            np.isfinite(depth_array)
            & (depth_array > 0.0)
            & (depth_array <= float(depth_trunc))
        )
        valid_depth_count = int(np.count_nonzero(valid_depth))
        if (
            isinstance(pre_downsample_point_count, bool)
            or int(pre_downsample_point_count) < 0
            or isinstance(post_downsample_point_count, bool)
            or int(post_downsample_point_count) < 0
        ):
            raise ValueError("Point counts must be non-negative integers.")
        finished_ns = time.perf_counter_ns()
        return CandidateGenerationMetadata(
            depth_source=normalized_source,
            depth_pixel_count=depth_pixel_count,
            valid_depth_count=valid_depth_count,
            pre_downsample_point_count=int(pre_downsample_point_count),
            post_downsample_point_count=int(post_downsample_point_count),
            metadata_cpu_ns=finished_ns - started_ns,
        )
    except Exception as error:
        finished_ns = time.perf_counter_ns()
        return CandidateGenerationMetadata(
            depth_source=(
                depth_source if depth_source in DEPTH_SOURCES else "unknown"
            ),
            depth_pixel_count=0,
            valid_depth_count=0,
            pre_downsample_point_count=max(
                0,
                int(pre_downsample_point_count)
                if not isinstance(pre_downsample_point_count, bool)
                else 0,
            ),
            post_downsample_point_count=max(
                0,
                int(post_downsample_point_count)
                if not isinstance(post_downsample_point_count, bool)
                else 0,
            ),
            metadata_cpu_ns=finished_ns - started_ns,
            error=_structured_error(error),
        )


def build_gaussian_candidate_observer(
    config: Optional[Mapping[str, Any]],
    *,
    resource_admission_mode: str,
    device: Any,
) -> Optional["GaussianCandidateObserver"]:
    """Validate local configuration and return None for the exact off path."""

    if config is None:
        return None
    config = _require_mapping(config, "mapping.candidate_observer")
    _reject_unknown_fields(
        config,
        _TOP_LEVEL_FIELDS,
        "mapping.candidate_observer",
    )

    mode = config.get("mode", "off")
    if not isinstance(mode, str):
        raise TypeError(
            "mapping.candidate_observer.mode must be a string, "
            f"got {mode!r}."
        )
    mode = mode.strip().lower()
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            "mapping.candidate_observer.mode must be one of "
            f"{sorted(SUPPORTED_MODES)}, got {mode!r}."
        )

    coverage = _require_mapping(
        config.get(
            "coverage",
            {"method": COVERAGE_METHOD, "voxel_size": None},
        ),
        "mapping.candidate_observer.coverage",
    )
    _reject_unknown_fields(
        coverage,
        _COVERAGE_FIELDS,
        "mapping.candidate_observer.coverage",
    )
    coverage_method = coverage.get("method", COVERAGE_METHOD)
    if not isinstance(coverage_method, str):
        raise TypeError(
            "mapping.candidate_observer.coverage.method must be a string, "
            f"got {coverage_method!r}."
        )
    coverage_method = coverage_method.strip().lower()
    if coverage_method != COVERAGE_METHOD:
        raise ValueError(
            "mapping.candidate_observer.coverage.method must be "
            f"{COVERAGE_METHOD!r}, got {coverage_method!r}."
        )
    voxel_size = _normalize_voxel_size(
        coverage.get("voxel_size", None),
        required=mode == "observe",
    )

    timing = _require_mapping(
        config.get("timing", {"gpu": False}),
        "mapping.candidate_observer.timing",
    )
    _reject_unknown_fields(
        timing,
        _TIMING_FIELDS,
        "mapping.candidate_observer.timing",
    )
    gpu_timing = _require_bool(
        timing.get("gpu", False),
        "mapping.candidate_observer.timing.gpu",
    )

    memory = _require_mapping(
        config.get("memory", {"enabled": False}),
        "mapping.candidate_observer.memory",
    )
    _reject_unknown_fields(
        memory,
        _MEMORY_FIELDS,
        "mapping.candidate_observer.memory",
    )
    memory_enabled = _require_bool(
        memory.get("enabled", False),
        "mapping.candidate_observer.memory.enabled",
    )

    logging = _require_mapping(
        config.get("logging", {"enabled": True}),
        "mapping.candidate_observer.logging",
    )
    _reject_unknown_fields(
        logging,
        _LOGGING_FIELDS,
        "mapping.candidate_observer.logging",
    )
    logging_enabled = _require_bool(
        logging.get("enabled", True),
        "mapping.candidate_observer.logging.enabled",
    )

    if mode == "off":
        return None
    if not logging_enabled:
        raise ValueError(
            "mapping.candidate_observer.logging.enabled must be true in "
            "observe mode so every Camera has an auditable event."
        )

    normalized_admission_mode = str(resource_admission_mode).strip().lower()
    if normalized_admission_mode not in {"disabled", "observe"}:
        raise ValueError(
            "Gaussian Candidate Observer observe mode requires M01 "
            "ResourceAdmission to be disabled, absent, or observe; got "
            f"{resource_admission_mode!r}."
        )

    assert voxel_size is not None
    return GaussianCandidateObserver(
        voxel_size=voxel_size,
        gpu_timing=gpu_timing,
        memory_enabled=memory_enabled,
        logging_enabled=logging_enabled,
        device=device,
    )


class GaussianCandidateObserver:
    """Observe candidate evidence without returning or mutating candidates."""

    schema = SCHEMA_VERSION

    def __init__(
        self,
        *,
        voxel_size: float,
        gpu_timing: bool,
        memory_enabled: bool,
        logging_enabled: bool,
        device: Any,
    ) -> None:
        normalized_voxel_size = _normalize_voxel_size(
            voxel_size,
            required=True,
        )
        assert normalized_voxel_size is not None
        self.voxel_size = normalized_voxel_size
        self.gpu_timing = _require_bool(
            gpu_timing,
            "candidate_observer.gpu_timing",
        )
        self.memory_enabled = _require_bool(
            memory_enabled,
            "candidate_observer.memory_enabled",
        )
        self.logging_enabled = _require_bool(
            logging_enabled,
            "candidate_observer.logging_enabled",
        )
        self.device = torch.device(device)

    @staticmethod
    def _validate_candidate_contract(
        *,
        xyz: torch.Tensor,
        features: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
    ) -> int:
        tensors = {
            "xyz": xyz,
            "features": features,
            "scales": scales,
            "rotations": rotations,
            "opacities": opacities,
        }
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors.values()):
            types = {
                name: type(tensor).__name__
                for name, tensor in tensors.items()
            }
            raise TypeError(f"Candidate inputs must be tensors, got {types}.")
        if xyz.ndim < 1:
            raise ValueError(f"Candidate xyz must have a leading dimension, got {list(xyz.shape)}.")
        candidate_count = int(xyz.shape[0])
        shapes = {name: list(tensor.shape) for name, tensor in tensors.items()}
        devices = {name: str(tensor.device) for name, tensor in tensors.items()}
        if any(
            tensor.ndim < 1 or int(tensor.shape[0]) != candidate_count
            for tensor in tensors.values()
        ):
            raise ValueError(
                "Candidate first-dimension mismatch: "
                f"candidate_count={candidate_count}, shapes={shapes}."
            )
        if any(tensor.device != xyz.device for tensor in tensors.values()):
            raise ValueError(
                "Candidate device mismatch: "
                f"devices={devices}."
            )
        return candidate_count

    def _memory_snapshot(
        self,
        reference: torch.Tensor,
    ) -> tuple[Optional[int], Optional[int]]:
        if (
            not self.memory_enabled
            or not reference.is_cuda
            or not torch.cuda.is_available()
        ):
            return None, None
        return (
            int(torch.cuda.memory_allocated(device=reference.device)),
            int(torch.cuda.memory_reserved(device=reference.device)),
        )

    @staticmethod
    def _valid_depth_ratio(
        metadata: CandidateGenerationMetadata,
    ) -> Optional[float]:
        if metadata.depth_pixel_count == 0:
            return None
        return metadata.valid_depth_count / metadata.depth_pixel_count

    def _base_fields(
        self,
        *,
        metadata: CandidateGenerationMetadata,
        mapper_update_id: int,
        source_camera_id: int,
        init: bool,
        candidate_count: int,
        existing_gaussian_count: int,
    ) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "event_type": "candidate_observation",
            "event_id": (
                f"gco-v0:{int(mapper_update_id)}:{int(source_camera_id)}"
            ),
            "status": "ok",
            "reason": "observed",
            "mapper_update_id": int(mapper_update_id),
            "source_camera_id": int(source_camera_id),
            "init": bool(init),
            "observer_mode": "observe",
            "gpu_timing_enabled": self.gpu_timing,
            "memory_observation_enabled": self.memory_enabled,
            "depth_source": metadata.depth_source,
            "depth_pixel_count": int(metadata.depth_pixel_count),
            "valid_depth_count": int(metadata.valid_depth_count),
            "valid_depth_ratio": self._valid_depth_ratio(metadata),
            "pre_downsample_point_count": int(
                metadata.pre_downsample_point_count
            ),
            "post_downsample_point_count": int(
                metadata.post_downsample_point_count
            ),
            "candidate_3d_count": int(candidate_count),
            "candidate_finite_count": 0,
            "candidate_nonfinite_count": int(candidate_count),
            "existing_gaussian_count": int(existing_gaussian_count),
            "coverage_method": COVERAGE_METHOD,
            "coverage_voxel_size": self.voxel_size,
            "candidate_unique_voxel_count": 0,
            "candidate_intra_voxel_duplicate_count": 0,
            "occupied_candidate_count": 0,
            "novel_candidate_count": 0,
            "occupied_ratio": None,
            "novel_ratio": None,
            "spatial_extent_x": None,
            "spatial_extent_y": None,
            "spatial_extent_z": None,
            "observer_cpu_ms": 0.0,
            "observer_gpu_ms": None,
            "allocated_before_bytes": None,
            "allocated_after_bytes": None,
            "reserved_before_bytes": None,
            "reserved_after_bytes": None,
            "gaussian_before": int(existing_gaussian_count),
            "empty_candidate": candidate_count == 0,
            "error": metadata.error,
        }

    def _error_token(
        self,
        *,
        metadata: CandidateGenerationMetadata,
        mapper_update_id: int,
        source_camera_id: int,
        init: bool,
        candidate_count: int,
        existing_gaussian_count: int,
        started_ns: int,
        error: BaseException,
    ) -> CandidateObservationToken:
        fields = self._base_fields(
            metadata=metadata,
            mapper_update_id=mapper_update_id,
            source_camera_id=source_camera_id,
            init=init,
            candidate_count=candidate_count,
            existing_gaussian_count=existing_gaussian_count,
        )
        fields.update(
            {
                "status": "error",
                "reason": "observer_evidence_error",
                "observer_cpu_ms": (
                    metadata.metadata_cpu_ns
                    + time.perf_counter_ns()
                    - started_ns
                )
                / 1_000_000.0,
                "error": _structured_error(error),
            }
        )
        return CandidateObservationToken(fields=fields)

    def observe_before_extend(
        self,
        *,
        xyz: torch.Tensor,
        features: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        current_gaussian_xyz: torch.Tensor,
        metadata: CandidateGenerationMetadata,
        mapper_update_id: int,
        source_camera_id: int,
        init: bool,
    ) -> CandidateObservationToken:
        """Compute read-only evidence and return no candidate tensors."""

        started_ns = time.perf_counter_ns()
        candidate_count = (
            int(xyz.shape[0])
            if isinstance(xyz, torch.Tensor) and xyz.ndim >= 1
            else 0
        )
        existing_count = (
            int(current_gaussian_xyz.shape[0])
            if isinstance(current_gaussian_xyz, torch.Tensor)
            and current_gaussian_xyz.ndim >= 1
            else 0
        )
        try:
            if metadata.error is not None:
                raise RuntimeError(
                    "Candidate generation metadata collection failed: "
                    f"{metadata.error}."
                )
            if (
                metadata.depth_pixel_count < 0
                or metadata.valid_depth_count < 0
                or metadata.valid_depth_count > metadata.depth_pixel_count
                or metadata.pre_downsample_point_count < 0
                or metadata.post_downsample_point_count < 0
                or metadata.post_downsample_point_count
                > metadata.pre_downsample_point_count
            ):
                raise ValueError(
                    "Candidate generation metadata count contract failed."
                )
            candidate_count = self._validate_candidate_contract(
                xyz=xyz,
                features=features,
                scales=scales,
                rotations=rotations,
                opacities=opacities,
            )
            if (
                not isinstance(current_gaussian_xyz, torch.Tensor)
                or current_gaussian_xyz.ndim != 2
                or current_gaussian_xyz.shape[1] != 3
            ):
                raise ValueError(
                    "current_gaussian_xyz must have shape [G,3], got "
                    f"{getattr(current_gaussian_xyz, 'shape', None)}."
                )
            if current_gaussian_xyz.device != xyz.device:
                raise ValueError(
                    "Candidate/current-map device mismatch: "
                    f"candidate={xyz.device}, map={current_gaussian_xyz.device}."
                )
            if metadata.post_downsample_point_count != candidate_count:
                raise ValueError(
                    "Post-downsample/candidate count mismatch: "
                    f"post_downsample={metadata.post_downsample_point_count}, "
                    f"candidate={candidate_count}."
                )

            fields = self._base_fields(
                metadata=metadata,
                mapper_update_id=mapper_update_id,
                source_camera_id=source_camera_id,
                init=init,
                candidate_count=candidate_count,
                existing_gaussian_count=existing_count,
            )
            allocated_before, reserved_before = self._memory_snapshot(xyz)

            start_event = None
            end_event = None
            if self.gpu_timing:
                if not xyz.is_cuda or not torch.cuda.is_available():
                    raise RuntimeError(
                        "GPU timing was requested but candidate tensors are "
                        "not on an available CUDA device."
                    )
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            with torch.no_grad():
                candidate_finite_mask = torch.isfinite(xyz).all(dim=1)
                finite_candidate_xyz = xyz[candidate_finite_mask]
                current_finite_mask = torch.isfinite(
                    current_gaussian_xyz
                ).all(dim=1)
                finite_current_xyz = current_gaussian_xyz[current_finite_mask]

                finite_count = int(finite_candidate_xyz.shape[0])
                if finite_count == 0:
                    unique_count = 0
                    occupied_count = 0
                    extent_values: list[Optional[float]] = [None, None, None]
                else:
                    candidate_voxels = torch.floor(
                        finite_candidate_xyz / self.voxel_size
                    ).to(dtype=torch.int64)
                    unique_candidate_voxels = torch.unique(
                        candidate_voxels,
                        dim=0,
                    )
                    unique_count = int(unique_candidate_voxels.shape[0])

                    if int(finite_current_xyz.shape[0]) == 0:
                        occupied_count_tensor = torch.zeros(
                            (),
                            dtype=torch.long,
                            device=xyz.device,
                        )
                    else:
                        existing_voxels = torch.floor(
                            finite_current_xyz / self.voxel_size
                        ).to(dtype=torch.int64)
                        combined_voxels = torch.cat(
                            (existing_voxels, candidate_voxels),
                            dim=0,
                        )
                        unique_voxels, inverse = torch.unique(
                            combined_voxels,
                            dim=0,
                            return_inverse=True,
                        )
                        existing_count_finite = int(existing_voxels.shape[0])
                        occupied_voxel_flags = torch.zeros(
                            (unique_voxels.shape[0],),
                            dtype=torch.bool,
                            device=xyz.device,
                        )
                        occupied_voxel_flags[
                            inverse[:existing_count_finite]
                        ] = True
                        candidate_occupied_mask = occupied_voxel_flags[
                            inverse[existing_count_finite:]
                        ]
                        occupied_count_tensor = torch.count_nonzero(
                            candidate_occupied_mask
                        )

                    extent_tensor = (
                        finite_candidate_xyz.amax(dim=0)
                        - finite_candidate_xyz.amin(dim=0)
                    )
                    scalar_values = torch.cat(
                        (
                            occupied_count_tensor.reshape(1).to(
                                dtype=torch.float64
                            ),
                            extent_tensor.reshape(3).to(dtype=torch.float64),
                        )
                    )

                    if end_event is not None:
                        end_event.record()
                        end_event.synchronize()

                    copied_values = scalar_values.cpu().tolist()
                    occupied_count = int(copied_values[0])
                    extent_values = [
                        float(copied_values[1]),
                        float(copied_values[2]),
                        float(copied_values[3]),
                    ]

                if end_event is not None and finite_count == 0:
                    end_event.record()
                    end_event.synchronize()

            gpu_elapsed_ms = (
                float(start_event.elapsed_time(end_event))
                if start_event is not None and end_event is not None
                else None
            )
            novel_count = finite_count - occupied_count
            allocated_after, reserved_after = self._memory_snapshot(xyz)
            fields.update(
                {
                    "candidate_finite_count": finite_count,
                    "candidate_nonfinite_count": (
                        candidate_count - finite_count
                    ),
                    "candidate_unique_voxel_count": unique_count,
                    "candidate_intra_voxel_duplicate_count": (
                        finite_count - unique_count
                    ),
                    "occupied_candidate_count": occupied_count,
                    "novel_candidate_count": novel_count,
                    "occupied_ratio": (
                        occupied_count / finite_count
                        if finite_count > 0
                        else None
                    ),
                    "novel_ratio": (
                        novel_count / finite_count
                        if finite_count > 0
                        else None
                    ),
                    "spatial_extent_x": extent_values[0],
                    "spatial_extent_y": extent_values[1],
                    "spatial_extent_z": extent_values[2],
                    "observer_gpu_ms": gpu_elapsed_ms,
                    "allocated_before_bytes": allocated_before,
                    "allocated_after_bytes": allocated_after,
                    "reserved_before_bytes": reserved_before,
                    "reserved_after_bytes": reserved_after,
                    "observer_cpu_ms": (
                        metadata.metadata_cpu_ns
                        + time.perf_counter_ns()
                        - started_ns
                    )
                    / 1_000_000.0,
                }
            )
            return CandidateObservationToken(fields=fields)
        except Exception as error:
            return self._error_token(
                metadata=metadata,
                mapper_update_id=mapper_update_id,
                source_camera_id=source_camera_id,
                init=init,
                candidate_count=candidate_count,
                existing_gaussian_count=existing_count,
                started_ns=started_ns,
                error=error,
            )

    def observe_empty(
        self,
        *,
        current_gaussian_xyz: torch.Tensor,
        metadata: CandidateGenerationMetadata,
        mapper_update_id: int,
        source_camera_id: int,
        init: bool,
    ) -> CandidateObservationToken:
        """Record a PCD that produced at most five points and was not emitted."""

        started_ns = time.perf_counter_ns()
        existing_count = (
            int(current_gaussian_xyz.shape[0])
            if isinstance(current_gaussian_xyz, torch.Tensor)
            and current_gaussian_xyz.ndim >= 1
            else 0
        )
        try:
            if metadata.error is not None:
                raise RuntimeError(
                    "Candidate generation metadata collection failed: "
                    f"{metadata.error}."
                )
            if not 0 <= metadata.post_downsample_point_count <= 5:
                raise ValueError(
                    "Empty candidate events require post-downsample point "
                    "count in 0..5."
                )
            if (
                metadata.pre_downsample_point_count
                < metadata.post_downsample_point_count
            ):
                raise ValueError(
                    "Pre-downsample point count must not be smaller than "
                    "post-downsample point count."
                )
            fields = self._base_fields(
                metadata=metadata,
                mapper_update_id=mapper_update_id,
                source_camera_id=source_camera_id,
                init=init,
                candidate_count=0,
                existing_gaussian_count=existing_count,
            )
            allocated_before, reserved_before = self._memory_snapshot(
                current_gaussian_xyz
            )
            allocated_after, reserved_after = self._memory_snapshot(
                current_gaussian_xyz
            )
            fields.update(
                {
                    "reason": "point_cloud_not_emitted_leq5",
                    "allocated_before_bytes": allocated_before,
                    "allocated_after_bytes": allocated_after,
                    "reserved_before_bytes": reserved_before,
                    "reserved_after_bytes": reserved_after,
                    "observer_gpu_ms": (
                        0.0
                        if self.gpu_timing
                        and current_gaussian_xyz.is_cuda
                        and torch.cuda.is_available()
                        else None
                    ),
                    "observer_cpu_ms": (
                        metadata.metadata_cpu_ns
                        + time.perf_counter_ns()
                        - started_ns
                    )
                    / 1_000_000.0,
                }
            )
            return CandidateObservationToken(fields=fields)
        except Exception as error:
            return self._error_token(
                metadata=metadata,
                mapper_update_id=mapper_update_id,
                source_camera_id=source_camera_id,
                init=init,
                candidate_count=0,
                existing_gaussian_count=existing_count,
                started_ns=started_ns,
                error=error,
            )

    def _sanitize_summary_float_fields(
        self,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        unsafe_fields: list[str] = []
        for field in _EVENT_FLOAT_FIELDS:
            value = fields.get(field)
            if value is None and field in _NULLABLE_EVENT_FLOAT_FIELDS:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                unsafe_fields.append(field)
                if field in _NULLABLE_EVENT_FLOAT_FIELDS:
                    fields[field] = None
                elif field == "observer_cpu_ms":
                    fields[field] = 0.0
                else:
                    fields[field] = self.voxel_size
            else:
                fields[field] = float(value)

        if unsafe_fields:
            fields.update(
                {
                    "status": "error",
                    "reason": "nonfinite_observer_summary",
                    "error": {
                        "type": "NonFiniteObserverSummaryError",
                        "message": (
                            "Observer summary contained unsafe float fields: "
                            + ", ".join(unsafe_fields)
                            + "."
                        ),
                    },
                }
            )
        return fields

    def _serialization_fallback_summary(
        self,
        summary: CandidateEvidenceSummary,
        error: BaseException,
    ) -> CandidateEvidenceSummary:
        fields = self._sanitize_summary_float_fields(asdict(summary))
        fields.update(
            {
                "status": "error",
                "reason": "observer_json_serialization_failed",
                "error": _structured_error(error),
            }
        )
        return CandidateEvidenceSummary(**fields)

    def record_after_extend(
        self,
        token: CandidateObservationToken,
        *,
        admitted_candidate_count: int,
        dropped_candidate_count: int,
        gaussian_after_extend: int,
    ) -> CandidateEvidenceSummary:
        """Finalize conservation facts after the unmodified extend path."""

        fields = dict(token.fields)
        candidate_count = int(fields["candidate_3d_count"])
        admitted_count = int(admitted_candidate_count)
        dropped_count = int(dropped_candidate_count)
        gaussian_before = int(fields["gaussian_before"])
        gaussian_after = int(gaussian_after_extend)
        finite_conservation = (
            int(fields["occupied_candidate_count"])
            + int(fields["novel_candidate_count"])
            == int(fields["candidate_finite_count"])
        )
        all_admitted = (
            candidate_count == admitted_count and dropped_count == 0
        )
        conservation_pass = (
            finite_conservation
            and all_admitted
            and gaussian_after == gaussian_before + admitted_count
        )
        fields.update(
            {
                "gaussian_after_extend": gaussian_after,
                "admitted_candidate_count": admitted_count,
                "dropped_candidate_count": dropped_count,
                "all_candidates_admitted": all_admitted,
                "conservation_pass": conservation_pass,
            }
        )
        if not conservation_pass and fields["status"] == "ok":
            fields.update(
                {
                    "status": "error",
                    "reason": "candidate_conservation_failed",
                    "error": {
                        "type": "ConservationError",
                        "message": (
                            "Candidate or Gaussian count conservation failed."
                        ),
                    },
                }
            )

        fields = self._sanitize_summary_float_fields(fields)
        try:
            summary = CandidateEvidenceSummary(**fields)
        except Exception as error:
            fields.update(
                {
                    "status": "error",
                    "reason": "summary_construction_failed",
                    "error": _structured_error(error),
                }
            )
            fields = self._sanitize_summary_float_fields(fields)
            summary = CandidateEvidenceSummary(**fields)

        if self.logging_enabled:
            try:
                payload = json.dumps(
                    summary.to_event(),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            except Exception as error:
                summary = self._serialization_fallback_summary(
                    summary,
                    error,
                )
                try:
                    payload = json.dumps(
                        summary.to_event(),
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                except Exception:
                    # A broken encoder must not undo the completed map extend.
                    return summary
            try:
                print(LOG_PREFIX + " " + payload, flush=True)
            except Exception:
                # A physically unwritable stdout is outside the JSON contract.
                return summary
        return summary
