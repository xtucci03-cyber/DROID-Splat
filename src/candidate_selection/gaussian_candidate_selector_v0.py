"""Dry-run estimates for occupied-voxel Gaussian candidate caps.

This module never selects or returns candidates.  It observes the candidate
batch immediately before M01 admission and records counterfactual K-cap
statistics while the original tensors continue through the existing path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Real
import time
from typing import Any, Mapping, Optional

import torch


LOG_PREFIX = "[GaussianCandidateSelectorDryRun]"
SCHEMA_VERSION = 1
MODE = "dry_run"
ALGORITHM = "occupied_voxel_cap_k_v0"
COVERAGE_METHOD = "voxel_occupancy"
K_VALUES = (1, 2, 4, 8)
SUPPORTED_MODES = frozenset({"off", MODE})

_TOP_LEVEL_FIELDS = frozenset({"mode", "algorithm", "logging"})
_LOGGING_FIELDS = frozenset({"enabled"})


def _structured_error(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


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


def _normalize_mode(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(
            "mapping.candidate_selector.mode must be a string, "
            f"got {value!r}."
        )
    mode = value.strip().lower()
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            "mapping.candidate_selector.mode must be one of "
            f"{sorted(SUPPORTED_MODES)}, got {mode!r}."
        )
    return mode


def _normalize_algorithm(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(
            "mapping.candidate_selector.algorithm must be a string, "
            f"got {value!r}."
        )
    algorithm = value.strip().lower()
    if algorithm != ALGORITHM:
        raise ValueError(
            "mapping.candidate_selector.algorithm must be "
            f"{ALGORITHM!r}, got {algorithm!r}."
        )
    return algorithm


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


def build_gaussian_candidate_selector(
    config: Optional[Mapping[str, Any]],
    *,
    candidate_observer: Any,
    resource_admission_mode: str,
    device: Any,
) -> Optional["GaussianCandidateSelectorDryRun"]:
    """Validate configuration and return None for the exact off path."""

    if config is None:
        return None
    config = _require_mapping(config, "mapping.candidate_selector")
    _reject_unknown_fields(
        config,
        _TOP_LEVEL_FIELDS,
        "mapping.candidate_selector",
    )
    mode = _normalize_mode(config.get("mode", "off"))

    algorithm_value = config.get("algorithm", ALGORITHM)
    if mode == MODE:
        algorithm = _normalize_algorithm(algorithm_value)
    else:
        # Off remains tolerant of the documented default but fail-closes on
        # an explicit malformed value or unknown algorithm.
        algorithm = _normalize_algorithm(algorithm_value)

    logging = _require_mapping(
        config.get("logging", {"enabled": True}),
        "mapping.candidate_selector.logging",
    )
    _reject_unknown_fields(
        logging,
        _LOGGING_FIELDS,
        "mapping.candidate_selector.logging",
    )
    logging_enabled = _require_bool(
        logging.get("enabled", True),
        "mapping.candidate_selector.logging.enabled",
    )

    if mode == "off":
        return None
    if not logging_enabled:
        raise ValueError(
            "mapping.candidate_selector.logging.enabled must be true in "
            "dry_run mode so every Camera has an auditable event."
        )

    normalized_admission_mode = str(resource_admission_mode).strip().lower()
    if normalized_admission_mode not in {"disabled", "observe"}:
        raise ValueError(
            "Gaussian Candidate Selector dry_run requires M01 "
            "ResourceAdmission to be disabled, absent, or observe; got "
            f"{resource_admission_mode!r}."
        )
    if candidate_observer is None:
        raise ValueError(
            "Gaussian Candidate Selector dry_run requires "
            "mapping.candidate_observer.mode=observe."
        )
    if not hasattr(candidate_observer, "voxel_size"):
        raise AttributeError(
            "Gaussian Candidate Observer must expose public voxel_size."
        )
    voxel_size = _normalize_voxel_size(candidate_observer.voxel_size)

    return GaussianCandidateSelectorDryRun(
        voxel_size=voxel_size,
        algorithm=algorithm,
        logging_enabled=logging_enabled,
        device=device,
    )


@dataclass(frozen=True)
class CandidateSelectorDryRunToken:
    """Small immutable state carried across the original extend call."""

    fields: dict[str, Any]


@dataclass(frozen=True)
class CandidateSelectorDryRunSummary:
    """One Camera-level dry-run result with no selection output."""

    fields: dict[str, Any]

    def to_event(self) -> dict[str, Any]:
        return dict(self.fields)


class GaussianCandidateSelectorDryRun:
    """Estimate occupied-voxel K caps without modifying candidates."""

    schema = SCHEMA_VERSION

    def __init__(
        self,
        *,
        voxel_size: float,
        algorithm: str,
        logging_enabled: bool,
        device: Any,
    ) -> None:
        self.voxel_size = _normalize_voxel_size(voxel_size)
        self.algorithm = _normalize_algorithm(algorithm)
        self.logging_enabled = _require_bool(
            logging_enabled,
            "candidate_selector.logging_enabled",
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
        if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
            types = {
                name: type(value).__name__
                for name, value in tensors.items()
            }
            raise TypeError(f"Candidate inputs must be tensors, got {types}.")
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(
                f"Candidate xyz must have shape [N,3], got {list(xyz.shape)}."
            )
        candidate_count = int(xyz.shape[0])
        shapes = {name: list(value.shape) for name, value in tensors.items()}
        devices = {name: str(value.device) for name, value in tensors.items()}
        if any(
            value.ndim < 1 or int(value.shape[0]) != candidate_count
            for value in tensors.values()
        ):
            raise ValueError(
                "Candidate first-dimension mismatch: "
                f"candidate_count={candidate_count}, shapes={shapes}."
            )
        if any(value.device != xyz.device for value in tensors.values()):
            raise ValueError(f"Candidate device mismatch: devices={devices}.")
        return candidate_count

    def _base_fields(
        self,
        *,
        sequence: int,
        mapper_update_id: int,
        source_camera_id: int,
        init: bool,
        candidate_count: int,
        existing_gaussian_count: int,
    ) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "event_type": "candidate_selector_dry_run",
            "event_id": (
                "gcs-v0:"
                f"{sequence}:{int(mapper_update_id)}:{int(source_camera_id)}"
            ),
            "event_sequence": sequence,
            "status": "ok",
            "reason": "dry_run_observed",
            "mode": MODE,
            "algorithm": self.algorithm,
            "mapper_update_id": int(mapper_update_id),
            "source_camera_id": int(source_camera_id),
            "init": bool(init),
            "protected_init": bool(init),
            "coverage_method": COVERAGE_METHOD,
            "coverage_voxel_size": self.voxel_size,
            "k_values": list(K_VALUES),
            "candidate_count": int(candidate_count),
            "candidate_finite_count": 0,
            "candidate_nonfinite_count": int(candidate_count),
            "existing_gaussian_count": int(existing_gaussian_count),
            "candidate_unique_voxel_count": 0,
            "occupied_candidate_count": 0,
            "novel_candidate_count": 0,
            "occupied_unique_voxel_count": 0,
            "novel_unique_voxel_count": 0,
            "occupied_multiplicity_histogram": [],
            "occupied_multiplicity_mean": None,
            "occupied_multiplicity_max": 0,
            "occupied_multiplicity_p50": None,
            "occupied_multiplicity_p90": None,
            "occupied_multiplicity_p95": None,
            "occupied_multiplicity_p99": None,
            "k_scan": [],
            "selection_applied": False,
            "selected_indices": None,
            "all_candidates_forwarded": True,
            "gaussian_before": int(existing_gaussian_count),
            "actual_admitted_count": 0,
            "actual_dropped_count": 0,
            "gaussian_after_extend": int(existing_gaussian_count),
            "actual_conservation_pass": False,
            "empty_candidate": candidate_count == 0,
            "dry_run_wall_ms": 0.0,
            "error": None,
        }

    @staticmethod
    def _nearest_rank_values(
        sorted_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        count = int(sorted_counts.shape[0])
        if count == 0:
            zero = torch.zeros((), dtype=torch.long, device=sorted_counts.device)
            return zero, zero, zero, zero
        indices = [
            max(0, math.ceil(percentile * count) - 1)
            for percentile in (0.50, 0.90, 0.95, 0.99)
        ]
        return tuple(sorted_counts[index] for index in indices)

    @staticmethod
    def _k_scan(
        *,
        candidate_count: int,
        protected_init: bool,
        novel_count: int,
        nonfinite_count: int,
        occupied_multiplicity_histogram: list[dict[str, int]],
    ) -> list[dict[str, Any]]:
        results = []
        for k_value in K_VALUES:
            if protected_init:
                admitted = candidate_count
                dropped = 0
            else:
                occupied_admitted = sum(
                    min(bucket["multiplicity"], k_value)
                    * bucket["voxel_count"]
                    for bucket in occupied_multiplicity_histogram
                )
                admitted = novel_count + nonfinite_count + occupied_admitted
                dropped = candidate_count - admitted
            results.append(
                {
                    "k": k_value,
                    "estimated_admitted_count": admitted,
                    "estimated_dropped_count": dropped,
                    "admitted_ratio": (
                        admitted / candidate_count
                        if candidate_count > 0
                        else None
                    ),
                    "dropped_ratio": (
                        dropped / candidate_count
                        if candidate_count > 0
                        else None
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
        mapper_update_id: int,
        source_camera_id: int,
        init: bool,
    ) -> CandidateSelectorDryRunToken:
        """Compute aggregate counterfactual evidence without candidates out."""

        started_ns = time.perf_counter_ns()
        sequence = self._next_event_sequence()
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
        fields = self._base_fields(
            sequence=sequence,
            mapper_update_id=mapper_update_id,
            source_camera_id=source_camera_id,
            init=init,
            candidate_count=candidate_count,
            existing_gaussian_count=existing_count,
        )

        try:
            candidate_count = self._validate_candidate_contract(
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

            with torch.no_grad():
                candidate_finite_mask = torch.isfinite(xyz).all(dim=1)
                finite_candidate_xyz = xyz[candidate_finite_mask]
                finite_count = int(finite_candidate_xyz.shape[0])
                nonfinite_count = candidate_count - finite_count

                current_finite_mask = torch.isfinite(
                    current_gaussian_xyz
                ).all(dim=1)
                finite_current_xyz = current_gaussian_xyz[current_finite_mask]

                occupied_count = 0
                novel_count = 0
                candidate_unique_count = 0
                occupied_unique_count = 0
                novel_unique_count = 0
                histogram: list[dict[str, int]] = []
                multiplicity_mean = None
                multiplicity_max = 0
                quantiles: list[Optional[int]] = [None, None, None, None]

                if finite_count > 0:
                    candidate_voxels = torch.floor(
                        finite_candidate_xyz / self.voxel_size
                    ).to(dtype=torch.int64)
                    unique_candidate_voxels, candidate_inverse = torch.unique(
                        candidate_voxels,
                        dim=0,
                        return_inverse=True,
                    )
                    candidate_unique_count = int(
                        unique_candidate_voxels.shape[0]
                    )

                    if int(finite_current_xyz.shape[0]) == 0:
                        candidate_occupied_mask = torch.zeros(
                            (finite_count,),
                            dtype=torch.bool,
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
                        finite_existing_count = int(existing_voxels.shape[0])
                        occupied_flags = torch.zeros(
                            (unique_voxels.shape[0],),
                            dtype=torch.bool,
                            device=xyz.device,
                        )
                        occupied_flags[inverse[:finite_existing_count]] = True
                        candidate_occupied_mask = occupied_flags[
                            inverse[finite_existing_count:]
                        ]

                    occupied_count_tensor = torch.count_nonzero(
                        candidate_occupied_mask
                    )
                    occupied_candidate_voxel_ids = candidate_inverse[
                        candidate_occupied_mask
                    ]
                    novel_candidate_voxel_ids = candidate_inverse[
                        ~candidate_occupied_mask
                    ]
                    occupied_unique_ids, occupied_counts = torch.unique(
                        occupied_candidate_voxel_ids,
                        return_counts=True,
                    )
                    novel_unique_ids = torch.unique(novel_candidate_voxel_ids)
                    occupied_unique_count = int(occupied_unique_ids.shape[0])
                    novel_unique_count = int(novel_unique_ids.shape[0])

                    if occupied_unique_count > 0:
                        histogram_multiplicity, histogram_voxel_count = (
                            torch.unique(occupied_counts, return_counts=True)
                        )
                        sorted_counts = torch.sort(occupied_counts).values
                        q50, q90, q95, q99 = self._nearest_rank_values(
                            sorted_counts
                        )
                        aggregate = torch.cat(
                            (
                                occupied_count_tensor.reshape(1),
                                occupied_counts.sum().reshape(1),
                                occupied_counts.max().reshape(1),
                                q50.reshape(1),
                                q90.reshape(1),
                                q95.reshape(1),
                                q99.reshape(1),
                            )
                        ).to(dtype=torch.int64)
                        aggregate_values = aggregate.cpu().tolist()
                        occupied_count = int(aggregate_values[0])
                        multiplicity_sum = int(aggregate_values[1])
                        multiplicity_max = int(aggregate_values[2])
                        quantiles = [int(value) for value in aggregate_values[3:]]
                        multiplicity_mean = (
                            multiplicity_sum / occupied_unique_count
                        )
                        histogram_rows = torch.stack(
                            (
                                histogram_multiplicity,
                                histogram_voxel_count,
                            ),
                            dim=1,
                        ).cpu().tolist()
                        histogram = [
                            {
                                "multiplicity": int(multiplicity),
                                "voxel_count": int(voxel_count),
                            }
                            for multiplicity, voxel_count in histogram_rows
                        ]
                    else:
                        occupied_count = 0

                    novel_count = finite_count - occupied_count

                fields.update(
                    {
                        "candidate_count": candidate_count,
                        "candidate_finite_count": finite_count,
                        "candidate_nonfinite_count": nonfinite_count,
                        "candidate_unique_voxel_count": candidate_unique_count,
                        "occupied_candidate_count": occupied_count,
                        "novel_candidate_count": novel_count,
                        "occupied_unique_voxel_count": occupied_unique_count,
                        "novel_unique_voxel_count": novel_unique_count,
                        "occupied_multiplicity_histogram": histogram,
                        "occupied_multiplicity_mean": multiplicity_mean,
                        "occupied_multiplicity_max": multiplicity_max,
                        "occupied_multiplicity_p50": quantiles[0],
                        "occupied_multiplicity_p90": quantiles[1],
                        "occupied_multiplicity_p95": quantiles[2],
                        "occupied_multiplicity_p99": quantiles[3],
                        "k_scan": self._k_scan(
                            candidate_count=candidate_count,
                            protected_init=bool(init),
                            novel_count=novel_count,
                            nonfinite_count=nonfinite_count,
                            occupied_multiplicity_histogram=histogram,
                        ),
                    }
                )

            if fields["candidate_nonfinite_count"] > 0:
                fields.update(
                    {
                        "status": "error",
                        "reason": "nonfinite_candidate_coordinates",
                        "error": {
                            "type": "NonFiniteCandidateError",
                            "message": (
                                "Candidate batch contains non-finite xyz; "
                                "the original path remains unchanged."
                            ),
                        },
                    }
                )
        except Exception as error:
            fields.update(
                {
                    "status": "error",
                    "reason": "dry_run_evidence_error",
                    "error": _structured_error(error),
                    "k_scan": self._k_scan(
                        candidate_count=candidate_count,
                        protected_init=bool(init),
                        novel_count=0,
                        nonfinite_count=candidate_count,
                        occupied_multiplicity_histogram=[],
                    ),
                }
            )

        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        if not math.isfinite(elapsed_ms):
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
        else:
            fields["dry_run_wall_ms"] = elapsed_ms
        return CandidateSelectorDryRunToken(fields=fields)

    def observe_empty(
        self,
        *,
        current_gaussian_xyz: torch.Tensor,
        mapper_update_id: int,
        source_camera_id: int,
        init: bool,
    ) -> CandidateSelectorDryRunToken:
        """Record the original no-point path without creating candidates."""

        started_ns = time.perf_counter_ns()
        sequence = self._next_event_sequence()
        existing_count = (
            int(current_gaussian_xyz.shape[0])
            if isinstance(current_gaussian_xyz, torch.Tensor)
            and current_gaussian_xyz.ndim >= 1
            else 0
        )
        fields = self._base_fields(
            sequence=sequence,
            mapper_update_id=mapper_update_id,
            source_camera_id=source_camera_id,
            init=init,
            candidate_count=0,
            existing_gaussian_count=existing_count,
        )
        try:
            self._normalize_current_gaussian_xyz(current_gaussian_xyz)
            fields.update(
                {
                    "reason": "point_cloud_not_emitted_leq5",
                    "k_scan": self._k_scan(
                        candidate_count=0,
                        protected_init=bool(init),
                        novel_count=0,
                        nonfinite_count=0,
                        occupied_multiplicity_histogram=[],
                    ),
                }
            )
        except Exception as error:
            fields.update(
                {
                    "status": "error",
                    "reason": "dry_run_evidence_error",
                    "error": _structured_error(error),
                    "k_scan": self._k_scan(
                        candidate_count=0,
                        protected_init=bool(init),
                        novel_count=0,
                        nonfinite_count=0,
                        occupied_multiplicity_histogram=[],
                    ),
                }
            )
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        fields["dry_run_wall_ms"] = elapsed_ms if math.isfinite(elapsed_ms) else None
        return CandidateSelectorDryRunToken(fields=fields)

    def record_after_extend(
        self,
        token: CandidateSelectorDryRunToken,
        *,
        admitted_candidate_count: int,
        dropped_candidate_count: int,
        gaussian_after_extend: int,
    ) -> CandidateSelectorDryRunSummary:
        """Finalize actual-path conservation and emit one independent event."""

        fields = dict(token.fields)
        candidate_count = int(fields["candidate_count"])
        admitted_count = int(admitted_candidate_count)
        dropped_count = int(dropped_candidate_count)
        gaussian_before = int(fields["gaussian_before"])
        gaussian_after = int(gaussian_after_extend)
        actual_conservation = (
            candidate_count == admitted_count
            and dropped_count == 0
            and gaussian_after == gaussian_before + admitted_count
        )
        fields.update(
            {
                "actual_admitted_count": admitted_count,
                "actual_dropped_count": dropped_count,
                "gaussian_after_extend": gaussian_after,
                "all_candidates_forwarded": (
                    candidate_count == admitted_count and dropped_count == 0
                ),
                "actual_conservation_pass": actual_conservation,
            }
        )
        if not actual_conservation and fields["status"] == "ok":
            fields.update(
                {
                    "status": "error",
                    "reason": "actual_path_conservation_failed",
                    "error": {
                        "type": "ConservationError",
                        "message": (
                            "Dry-run requires the original candidate batch "
                            "to be forwarded without admission control."
                        ),
                    },
                }
            )

        summary = CandidateSelectorDryRunSummary(fields=fields)
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
                fields.update(
                    {
                        "status": "error",
                        "reason": "dry_run_json_serialization_failed",
                        "dry_run_wall_ms": (
                            fields.get("dry_run_wall_ms")
                            if isinstance(fields.get("dry_run_wall_ms"), Real)
                            and math.isfinite(float(fields["dry_run_wall_ms"]))
                            else None
                        ),
                        "error": _structured_error(error),
                    }
                )
                summary = CandidateSelectorDryRunSummary(fields=fields)
                try:
                    payload = json.dumps(
                        summary.to_event(),
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                except Exception:
                    return summary
            try:
                print(LOG_PREFIX + " " + payload, flush=True)
            except Exception:
                return summary
        return summary
