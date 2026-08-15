"""Marginal-utility diagnostics and fixed-budget experiments for GCS-v2.

This module consumes the already-captured pre-insertion render evidence.  It
never renders.  The original V2-A observer remains pass-through; the separate
active class ranks the same candidate batch under a fixed budget without
changing the frozen GCS-v1 implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Integral, Real
import time
from typing import Any, Callable, Mapping, Optional

import torch

from .gaussian_candidate_dynamic_budget_v1 import (
    DynamicBudgetObserveConfigV1,
    GaussianCandidateDynamicBudgetObserverV1,
    normalize_dynamic_budget_observe_config_v1,
)
from .gaussian_candidate_active_topk_v1 import (
    ActiveConfidenceEvidenceError,
    CONFIDENCE_SAMPLING_METHOD,
    GaussianCandidateActiveTopKV1,
    _known_snapshot_failure,
)
from .preinsert_render_evidence_v1 import (
    PreinsertRenderEvidenceError,
    PreinsertRenderEvidenceV1,
)
from ..resource_management.m01_resource_admission.resource_admission import (
    deterministic_uniform_indices,
)


LOG_PREFIX = "[GaussianCandidateSelectorV2]"
SCHEMA_VERSION = 1
MODE = "observe"
ACTIVE_FIXED_K_MODE = "active_fixed_k"
DYNAMIC_K_OBSERVE_MODE = "dynamic_k_observe"
SUPPORTED_MODES = frozenset(
    {"off", MODE, ACTIVE_FIXED_K_MODE, DYNAMIC_K_OBSERVE_MODE}
)
ACTIVE_STRATEGIES = frozenset(
    {"coverage", "evidence", "evidence_projected_cell"}
)
DEFAULT_FIXED_K = 600
DEFAULT_DIVERSITY_STRENGTH = 0.1
NUMERICAL_EPSILON = 1.0e-6
CONFIDENCE_NORMALIZER = math.sqrt(2.0)

_TOP_LEVEL_FIELDS = frozenset(
    {
        "mode",
        "strategy",
        "fixed_k",
        "diversity_strength",
        "dynamic_budget",
        "logging",
    }
)
_LOGGING_FIELDS = frozenset({"enabled"})
_COMPONENTS = (
    "confidence",
    "alpha",
    "coverage_gap",
    "rgb_l1",
    "depth_relative_error",
    "multiplicity",
    "cell_redundancy",
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


def _normalize_mode(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("mapping.candidate_selector_v2.mode must be a string.")
    mode = value.strip().lower()
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            "mapping.candidate_selector_v2.mode must be one of "
            f"{sorted(SUPPORTED_MODES)}, got {mode!r}."
        )
    return mode


def _normalize_strategy(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("mapping.candidate_selector_v2.strategy must be a string.")
    strategy = value.strip().lower()
    if strategy not in ACTIVE_STRATEGIES:
        raise ValueError(
            "mapping.candidate_selector_v2.strategy must be one of "
            f"{sorted(ACTIVE_STRATEGIES)}, got {strategy!r}."
        )
    return strategy


def _normalize_fixed_k(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            "mapping.candidate_selector_v2.fixed_k must be a positive integer."
        )
    value = int(value)
    if value < 1:
        raise ValueError(
            "mapping.candidate_selector_v2.fixed_k must be greater than zero."
        )
    return value


def _normalize_diversity_strength(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            "mapping.candidate_selector_v2.diversity_strength must be a "
            "finite non-negative number."
        )
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "mapping.candidate_selector_v2.diversity_strength must be a "
            "finite non-negative number."
        )
    return value


def build_gaussian_candidate_selector_v2(
    config: Optional[Mapping[str, Any]],
    *,
    candidate_selector_v1: Any,
    resource_admission_mode: str,
    preinsert_render_evidence_v1: Any,
    confidence_snapshot_getter: Callable[..., Any],
    device: Any,
) -> Optional["GaussianCandidateSelectorV2"]:
    """Build V2 diagnostics/selection; exact off has no runtime state."""

    if config is None:
        return None
    config = _require_mapping(config, "mapping.candidate_selector_v2")
    _reject_unknown_fields(config, _TOP_LEVEL_FIELDS, "mapping.candidate_selector_v2")
    mode = _normalize_mode(config.get("mode", "off"))
    logging = _require_mapping(
        config.get("logging", {"enabled": True}),
        "mapping.candidate_selector_v2.logging",
    )
    _reject_unknown_fields(
        logging,
        _LOGGING_FIELDS,
        "mapping.candidate_selector_v2.logging",
    )
    logging_enabled = _require_bool(
        logging.get("enabled", True),
        "mapping.candidate_selector_v2.logging.enabled",
    )
    dynamic_budget_config = normalize_dynamic_budget_observe_config_v1(
        config.get("dynamic_budget", None)
    )
    if mode == "off":
        return None
    if not logging_enabled:
        raise ValueError(
            "mapping.candidate_selector_v2.logging.enabled must be true "
            "outside off mode."
        )
    if str(resource_admission_mode).strip().lower() != "disabled":
        raise ValueError(
            "GCS-v2 requires mapping.resource_admission.mode=disabled."
        )
    if candidate_selector_v1 is not None:
        raise ValueError(
            "GCS-v2 requires mapping.candidate_selector_v1.mode=off."
        )
    if (
        preinsert_render_evidence_v1 is None
        or getattr(preinsert_render_evidence_v1, "mode", None) != "observe"
    ):
        raise ValueError(
            "GCS-v2 requires "
            "mapping.preinsert_render_evidence_v1.mode=observe."
        )
    if not callable(confidence_snapshot_getter):
        raise TypeError("confidence_snapshot_getter must be callable.")
    common = {
        "confidence_snapshot_getter": confidence_snapshot_getter,
        "logging_enabled": logging_enabled,
        "device": device,
    }
    if mode == MODE:
        return GaussianCandidateSelectorV2(**common)
    if mode == DYNAMIC_K_OBSERVE_MODE:
        strategy = _normalize_strategy(config.get("strategy", "evidence"))
        if strategy != "evidence":
            raise ValueError(
                "mapping.candidate_selector_v2.strategy must be 'evidence' "
                "in dynamic_k_observe mode."
            )
        return GaussianCandidateDynamicKObserveV2(
            **common,
            dynamic_budget_config=dynamic_budget_config,
        )
    return GaussianCandidateActiveFixedKV2(
        **common,
        strategy=_normalize_strategy(config.get("strategy", "evidence")),
        fixed_k=_normalize_fixed_k(config.get("fixed_k", DEFAULT_FIXED_K)),
        diversity_strength=_normalize_diversity_strength(
            config.get("diversity_strength", DEFAULT_DIVERSITY_STRENGTH)
        ),
    )


@dataclass(frozen=True)
class CandidateMarginalUtilityTokenV2:
    """Tensor-free event state finalized after the unchanged extend path."""

    fields: dict[str, Any]


@dataclass(frozen=True)
class CandidateMarginalUtilitySummaryV2:
    fields: dict[str, Any]

    def to_event(self) -> dict[str, Any]:
        return dict(self.fields)


@dataclass(frozen=True)
class CandidateActiveSelectionTokenV2:
    """Tensor-free fixed-budget event state finalized after extension."""

    fields: dict[str, Any]


@dataclass(frozen=True)
class CandidateActiveSelectionResultV2:
    xyz: torch.Tensor
    features: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor
    selected_indices: Optional[torch.Tensor]
    token: CandidateActiveSelectionTokenV2


@dataclass(frozen=True)
class CandidateActiveSelectionSummaryV2:
    fields: dict[str, Any]

    def to_event(self) -> dict[str, Any]:
        return dict(self.fields)


class GaussianCandidateSelectorV2:
    """Observe V2-A components without changing candidate admission."""

    mode = MODE
    is_active = False

    def __init__(
        self,
        *,
        confidence_snapshot_getter: Callable[..., Any],
        logging_enabled: bool,
        device: Any,
    ) -> None:
        if not callable(confidence_snapshot_getter):
            raise TypeError("confidence_snapshot_getter must be callable.")
        self.logging_enabled = _require_bool(
            logging_enabled,
            "mapping.candidate_selector_v2.logging.enabled",
        )
        self.device = torch.device(device)
        self._event_sequence = 0
        self._confidence_snapshot_getter = confidence_snapshot_getter

    # Invoke the frozen v1 methods as read-only unbound helpers.  This avoids
    # constructing an active selector and cannot enter its selection API.
    _source_depth = staticmethod(GaussianCandidateActiveTopKV1._source_depth)
    _camera_intrinsics = staticmethod(
        GaussianCandidateActiveTopKV1._camera_intrinsics
    )

    def project_world_to_source_pixels(
        self,
        xyz: torch.Tensor,
        camera: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return GaussianCandidateActiveTopKV1.project_world_to_source_pixels(
            self,
            xyz,
            camera,
        )
    map_source_pixels_to_confidence = staticmethod(
        GaussianCandidateActiveTopKV1.map_source_pixels_to_confidence
    )

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
        return GaussianCandidateActiveTopKV1._validate_candidates(
            xyz=xyz,
            features=features,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
        )

    @staticmethod
    def _tensor_state(tensors: Mapping[str, torch.Tensor]) -> dict[str, tuple[Any, ...]]:
        return {
            name: (
                int(value.data_ptr()),
                int(value._version),
                tuple(value.shape),
                value.dtype,
                value.device,
                bool(value.requires_grad),
            )
            for name, value in tensors.items()
        }

    @staticmethod
    def _count(mask: torch.Tensor) -> int:
        return int(torch.count_nonzero(mask).detach().cpu())

    @staticmethod
    def _stats(values: torch.Tensor) -> dict[str, Optional[float]]:
        if values.ndim != 1:
            values = values.reshape(-1)
        count = int(values.shape[0])
        if count == 0:
            return {
                "count": 0,
                "min": None,
                "mean": None,
                "p50": None,
                "p90": None,
                "p95": None,
                "max": None,
            }
        values = values.to(dtype=torch.float64)
        quantiles = torch.tensor(
            [0.5, 0.9, 0.95],
            dtype=values.dtype,
            device=values.device,
        )
        aggregate = torch.cat(
            (
                torch.stack((values.amin(), values.mean())),
                torch.quantile(values, quantiles),
                values.amax().reshape(1),
            )
        )
        minimum, mean, p50, p90, p95, maximum = (
            float(value) for value in aggregate.detach().cpu().tolist()
        )
        if not all(
            math.isfinite(value)
            for value in (minimum, mean, p50, p90, p95, maximum)
        ):
            raise RuntimeError("GCS-v2 aggregate statistics are non-finite.")
        return {
            "count": count,
            "min": minimum,
            "mean": mean,
            "p50": p50,
            "p90": p90,
            "p95": p95,
            "max": maximum,
        }

    @staticmethod
    def _camera_identity_fields(camera: Any) -> dict[str, Any]:
        timestamp = getattr(camera, "source_timestamp", None)
        try:
            timestamp = float(timestamp)
            if not math.isfinite(timestamp):
                timestamp = None
        except (TypeError, ValueError, OverflowError):
            timestamp = None
        return {
            "source_camera_id": getattr(camera, "uid", None),
            "source_frame_id": getattr(camera, "source_frame_id", None),
            "source_buffer_index": getattr(camera, "buffer_index", None),
            "source_timestamp": timestamp,
        }

    @classmethod
    def _base_fields(
        cls,
        *,
        sequence: int,
        camera: Any,
        mapper_update_id: int,
        candidate_count: int,
        gaussian_before: int,
        init: bool,
    ) -> dict[str, Any]:
        identity = cls._camera_identity_fields(camera)
        fields: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "event_type": "candidate_marginal_utility_evidence",
            "event_id": (
                f"gcs-v2a:{sequence}:{mapper_update_id}:"
                f"{identity['source_camera_id']}"
            ),
            "event_sequence": sequence,
            "mode": MODE,
            "observe_only": True,
            "status": "ok",
            "reason": "observed",
            "error": None,
            "mapper_update_id": int(mapper_update_id),
            "candidate_count": int(candidate_count),
            "gaussian_before": int(gaussian_before),
            "init_bypass": bool(init),
            "empty_candidate": candidate_count == 0,
            "evidence_available": False,
            "evidence_reason": None,
            "evidence_error": None,
            "confidence_available": False,
            "confidence_reason": None,
            "confidence_error": None,
            "confidence_sampling_method": None,
            "confidence_version": None,
            "confidence_source_frame_id": None,
            "source_resolution": None,
            "confidence_resolution": None,
            "scale_x": None,
            "scale_y": None,
            "projection_valid_count": 0,
            "projection_invalid_count": int(candidate_count),
            "mask_valid_count": 0,
            "selected_indices_created": False,
            "active_selection_applied": False,
            "all_candidates_forwarded": None,
            "actual_admitted_count": None,
            "actual_dropped_count": None,
            "gaussian_after_extend": None,
            "observer_no_mutation": True,
            "host_wall_elapsed_ms": 0.0,
            "diagnostic_gpu_to_cpu_sync_expected": False,
            "numerical_epsilon": NUMERICAL_EPSILON,
            **identity,
        }
        empty_stats = cls._stats(torch.empty(0, dtype=torch.float32))
        for component in _COMPONENTS:
            fields[f"{component}_valid_count"] = 0
            fields[f"{component}_invalid_count"] = int(candidate_count)
            fields[component] = dict(empty_stats)
        fields["confidence_raw"] = dict(empty_stats)
        fields["confidence_norm"] = dict(empty_stats)
        fields["candidate_camera_z"] = dict(empty_stats)
        return fields

    @staticmethod
    def _diagnostic_gpu_to_cpu_sync_expected(
        *,
        device_type: str,
        init: bool,
        candidate_count: int,
    ) -> bool:
        """Report a conservative diagnostic-path property, not an observation.

        Non-empty, non-initialization CUDA observations aggregate tensor values
        on the host.  The field deliberately says ``expected`` because this
        helper neither profiles synchronization nor adds an explicit CUDA
        synchronization point.  It remains true when render evidence is
        unavailable because confidence diagnostics are still attempted.
        """

        return (
            str(device_type).strip().lower() == "cuda"
            and not bool(init)
            and int(candidate_count) > 0
        )

    @staticmethod
    def _mask(camera: Any, *, device: torch.device, height: int, width: int) -> torch.Tensor:
        value = getattr(camera, "mask", None)
        if value is None:
            return torch.ones((height, width), dtype=torch.bool, device=device)
        if not isinstance(value, torch.Tensor):
            raise ValueError("Camera mask must be a Tensor when present.")
        if value.device != device:
            raise ValueError("Camera mask/candidate device mismatch.")
        if tuple(value.shape) == (1, height, width):
            value = value[0]
        if tuple(value.shape) != (height, width):
            raise ValueError(
                "Camera mask shape mismatch: expected "
                f"{(height, width)}, got {tuple(value.shape)}."
            )
        return value.to(dtype=torch.bool)

    def _observe_render_components(
        self,
        *,
        fields: dict[str, Any],
        xyz: torch.Tensor,
        camera: Any,
        gaussian_before: int,
        mapper_update_id: int,
        evidence: Optional[PreinsertRenderEvidenceV1],
        u: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        projection_valid: Optional[torch.Tensor],
    ) -> None:
        count = int(xyz.shape[0])
        if evidence is None:
            fields["evidence_reason"] = "missing_preinsert_render_evidence"
            return
        try:
            evidence.validate_for_event(
                camera=camera,
                gaussian_count_current=gaussian_before,
                mapper_update_id=mapper_update_id,
            )
        except PreinsertRenderEvidenceError as error:
            fields["evidence_reason"] = "invalid_preinsert_render_evidence"
            fields["evidence_error"] = _structured_error(error)
            return
        if not evidence.available:
            fields["evidence_reason"] = evidence.unavailable_reason
            return
        fields["evidence_available"] = True
        fields["evidence_reason"] = "available"
        if u is None or v is None or z is None or projection_valid is None:
            fields["evidence_available"] = False
            fields["evidence_reason"] = "invalid_candidate_projection"
            return

        height, width = int(evidence.height), int(evidence.width)
        in_bounds = (
            projection_valid
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )
        fields["projection_valid_count"] = self._count(in_bounds)
        fields["projection_invalid_count"] = count - fields["projection_valid_count"]
        indices = torch.nonzero(in_bounds, as_tuple=False).reshape(-1)
        if int(indices.shape[0]) == 0:
            return
        sample_u = u[indices]
        sample_v = v[indices]

        mask = self._mask(camera, device=xyz.device, height=height, width=width)
        mask_samples = mask[sample_v, sample_u]
        fields["mask_valid_count"] = self._count(mask_samples)

        alpha_raw = evidence.alpha_accum[0, sample_v, sample_u]
        alpha_valid = torch.isfinite(alpha_raw)
        alpha = torch.clamp(alpha_raw, 0.0, 1.0)
        alpha_values = alpha[alpha_valid]
        alpha_count = int(alpha_values.shape[0])
        fields["alpha_valid_count"] = alpha_count
        fields["alpha_invalid_count"] = count - alpha_count
        fields["coverage_gap_valid_count"] = alpha_count
        fields["coverage_gap_invalid_count"] = count - alpha_count
        fields["alpha"] = self._stats(alpha_values)
        fields["coverage_gap"] = self._stats(1.0 - alpha_values)

        image = getattr(camera, "original_image", None)
        if (
            isinstance(image, torch.Tensor)
            and image.device == xyz.device
            and tuple(image.shape) == (3, height, width)
        ):
            render_rgb = evidence.render_rgb[:, sample_v, sample_u].transpose(0, 1)
            observed_rgb = (
                image[:, sample_v, sample_u]
                .transpose(0, 1)
                .to(dtype=render_rgb.dtype)
                / 255.0
            )
            rgb_finite = torch.isfinite(render_rgb).all(1) & torch.isfinite(
                observed_rgb
            ).all(1)
            rgb_valid = mask_samples & rgb_finite
            rgb_l1 = torch.mean(torch.abs(render_rgb - observed_rgb), dim=1)
            rgb_values = rgb_l1[rgb_valid]
            rgb_count = int(rgb_values.shape[0])
            fields["rgb_l1_valid_count"] = rgb_count
            fields["rgb_l1_invalid_count"] = count - rgb_count
            fields["rgb_l1"] = self._stats(rgb_values)
        else:
            fields["evidence_error"] = {
                "type": "ImageContractError",
                "message": "Camera original_image is missing or incompatible.",
            }

        depth_accum = evidence.depth_accum[0, sample_v, sample_u]
        candidate_z = z[indices]
        old_depth = depth_accum / torch.clamp(alpha, min=NUMERICAL_EPSILON)
        depth_valid = (
            mask_samples
            & alpha_valid
            & (alpha > NUMERICAL_EPSILON)
            & torch.isfinite(depth_accum)
            & torch.isfinite(candidate_z)
            & (candidate_z > 0)
            & torch.isfinite(old_depth)
            & (old_depth > 0)
        )
        relative_error = torch.abs(old_depth - candidate_z) / torch.clamp(
            candidate_z,
            min=NUMERICAL_EPSILON,
        )
        depth_values = relative_error[depth_valid]
        depth_count = int(depth_values.shape[0])
        fields["depth_relative_error_valid_count"] = depth_count
        fields["depth_relative_error_invalid_count"] = count - depth_count
        fields["depth_relative_error"] = self._stats(depth_values)
        fields["candidate_camera_z"] = self._stats(candidate_z[depth_valid])

    def observe_before_extend(
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
        preinsert_render_evidence: Optional[PreinsertRenderEvidenceV1],
    ) -> CandidateMarginalUtilityTokenV2:
        started_ns = time.perf_counter_ns()
        count, tensors = self._validate_candidates(
            xyz=xyz,
            features=features,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
        )
        input_state = self._tensor_state(tensors)
        sequence = self._next_event_sequence()
        fields = self._base_fields(
            sequence=sequence,
            camera=camera,
            mapper_update_id=mapper_update_id,
            candidate_count=count,
            gaussian_before=gaussian_before,
            init=init,
        )
        if init:
            fields.update(
                {
                    "reason": "init_bypass",
                    "evidence_reason": "init_bypass",
                    "confidence_reason": "init_bypass",
                }
            )
        elif count == 0:
            fields.update(
                {
                    "reason": "empty_candidate",
                    "evidence_reason": "empty_candidate",
                    "confidence_reason": "empty_candidate",
                }
            )
        else:
            u = v = z = projection_valid = None
            try:
                u, v, z, projection_valid = (
                    self.project_world_to_source_pixels(xyz, camera)
                )
            except Exception as error:
                fields["reason"] = "invalid_candidate_projection"
                fields["evidence_reason"] = "invalid_candidate_projection"
                fields["evidence_error"] = _structured_error(error)

            self._observe_render_components(
                fields=fields,
                xyz=xyz,
                camera=camera,
                gaussian_before=gaussian_before,
                mapper_update_id=mapper_update_id,
                evidence=preinsert_render_evidence,
                u=u,
                v=v,
                z=z,
                projection_valid=projection_valid,
            )

            try:
                sampled_confidence, confidence_metadata = (
                    GaussianCandidateActiveTopKV1._read_active_confidence(
                        self,
                        xyz=xyz,
                        camera=camera,
                        depthmap=depthmap,
                        depth_source=depth_source,
                    )
                )
                confidence_norm = torch.clamp(
                    sampled_confidence / CONFIDENCE_NORMALIZER,
                    0.0,
                    1.0,
                )
                confidence_count = int(sampled_confidence.shape[0])
                fields.update(confidence_metadata)
                fields["confidence_available"] = True
                fields["confidence_reason"] = "current"
                fields["confidence_valid_count"] = confidence_count
                fields["confidence_invalid_count"] = count - confidence_count
                fields["confidence_raw"] = self._stats(sampled_confidence)
                fields["confidence_norm"] = self._stats(confidence_norm)
                fields["confidence"] = dict(fields["confidence_norm"])

                if u is None or v is None:
                    raise RuntimeError(
                        "Confidence succeeded without projected source coordinates."
                    )
                confidence_height, confidence_width = confidence_metadata[
                    "confidence_resolution"
                ]
                source_height, source_width = confidence_metadata[
                    "source_resolution"
                ]
                confidence_u, confidence_v, _, _ = (
                    self.map_source_pixels_to_confidence(
                        u,
                        v,
                        source_resolution=(source_height, source_width),
                        confidence_resolution=(
                            confidence_height,
                            confidence_width,
                        ),
                    )
                )
                cell_ids = confidence_v * confidence_width + confidence_u
                _, inverse, counts = torch.unique(
                    cell_ids,
                    return_inverse=True,
                    return_counts=True,
                )
                multiplicity = counts[inverse].to(dtype=xyz.dtype)
                redundancy = 1.0 - torch.reciprocal(multiplicity)
                fields["multiplicity_valid_count"] = count
                fields["multiplicity_invalid_count"] = 0
                fields["cell_redundancy_valid_count"] = count
                fields["cell_redundancy_invalid_count"] = 0
                fields["multiplicity"] = self._stats(multiplicity)
                fields["cell_redundancy"] = self._stats(redundancy)
            except Exception as error:
                reason = (
                    error.reason
                    if isinstance(error, ActiveConfidenceEvidenceError)
                    else "confidence_observation_error"
                )
                fields["confidence_available"] = False
                fields["confidence_reason"] = reason
                fields["confidence_error"] = _structured_error(error)
                if fields["reason"] == "observed":
                    fields["reason"] = "confidence_unavailable"

            if fields["reason"] == "observed":
                if not fields["evidence_available"] and not fields[
                    "confidence_available"
                ]:
                    fields["reason"] = "evidence_and_confidence_unavailable"
                elif not fields["evidence_available"]:
                    fields["reason"] = "evidence_unavailable"
                elif not fields["confidence_available"]:
                    fields["reason"] = "confidence_unavailable"

        output_state = self._tensor_state(tensors)
        fields["observer_no_mutation"] = input_state == output_state
        if not fields["observer_no_mutation"]:
            raise RuntimeError("GCS-v2 observer mutated candidate tensor state.")
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        fields["host_wall_elapsed_ms"] = (
            elapsed_ms if math.isfinite(elapsed_ms) else None
        )
        fields["diagnostic_gpu_to_cpu_sync_expected"] = (
            self._diagnostic_gpu_to_cpu_sync_expected(
                device_type=xyz.device.type,
                init=init,
                candidate_count=count,
            )
        )
        return CandidateMarginalUtilityTokenV2(fields=fields)

    def observe_empty(
        self,
        *,
        camera: Any,
        mapper_update_id: int,
        init: bool,
        gaussian_before: int,
        preinsert_render_evidence: Optional[PreinsertRenderEvidenceV1],
    ) -> CandidateMarginalUtilityTokenV2:
        started_ns = time.perf_counter_ns()
        del preinsert_render_evidence
        sequence = self._next_event_sequence()
        fields = self._base_fields(
            sequence=sequence,
            camera=camera,
            mapper_update_id=mapper_update_id,
            candidate_count=0,
            gaussian_before=gaussian_before,
            init=init,
        )
        fields.update(
            {
                "reason": "init_bypass" if init else "empty_candidate",
                "evidence_reason": "init_bypass" if init else "empty_candidate",
                "confidence_reason": "init_bypass" if init else "empty_candidate",
            }
        )
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        fields["host_wall_elapsed_ms"] = (
            elapsed_ms if math.isfinite(elapsed_ms) else None
        )
        return CandidateMarginalUtilityTokenV2(fields=fields)

    def record_after_extend(
        self,
        token: CandidateMarginalUtilityTokenV2,
        *,
        admitted_candidate_count: int,
        dropped_candidate_count: int,
        gaussian_after_extend: int,
    ) -> CandidateMarginalUtilitySummaryV2:
        fields = dict(token.fields)
        candidate_count = int(fields["candidate_count"])
        admitted = int(admitted_candidate_count)
        dropped = int(dropped_candidate_count)
        gaussian_before = int(fields["gaussian_before"])
        gaussian_after = int(gaussian_after_extend)
        forwarded = candidate_count == admitted and dropped == 0
        conservation = gaussian_after == gaussian_before + admitted
        fields.update(
            {
                "actual_admitted_count": admitted,
                "actual_dropped_count": dropped,
                "gaussian_after_extend": gaussian_after,
                "all_candidates_forwarded": forwarded,
                "conservation_check": conservation,
            }
        )
        if not forwarded or not conservation:
            fields.update(
                {
                    "status": "error",
                    "reason": "observe_only_forwarding_contract_failed",
                    "error": {
                        "type": "ForwardingContractError",
                        "message": "GCS-v2 observe must preserve every candidate.",
                    },
                }
            )
        summary = CandidateMarginalUtilitySummaryV2(fields=fields)
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
            payload = json.dumps(
                {
                    "schema": SCHEMA_VERSION,
                    "event_type": "candidate_marginal_utility_evidence",
                    "event_id": event.get("event_id", "gcs-v2a:fallback"),
                    "event_sequence": event.get("event_sequence"),
                    "mode": MODE,
                    "observe_only": True,
                    "status": "error",
                    "reason": "event_serialization_failed",
                    "selected_indices_created": False,
                    "error": _structured_error(error),
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        print(f"{LOG_PREFIX} {payload}", flush=True)


class GaussianCandidateActiveFixedKV2(GaussianCandidateSelectorV2):
    """Select a fixed number of candidates from existing event evidence.

    This class deliberately does not call a renderer and does not reuse the
    observe-only quantile path.  Projected-cell diversity is an empirical,
    current-batch source-view confidence-cell penalty; it is not an existing-
    map/global redundancy model or a rate-distortion objective.
    ``diversity_strength`` is an experimental initial value; it is neither
    frozen nor claimed to be theoretically optimal.
    """

    mode = ACTIVE_FIXED_K_MODE
    is_active = True

    def __init__(
        self,
        *,
        confidence_snapshot_getter: Callable[..., Any],
        logging_enabled: bool,
        device: Any,
        strategy: str,
        fixed_k: int = DEFAULT_FIXED_K,
        diversity_strength: float = DEFAULT_DIVERSITY_STRENGTH,
    ) -> None:
        super().__init__(
            confidence_snapshot_getter=confidence_snapshot_getter,
            logging_enabled=logging_enabled,
            device=device,
        )
        self.strategy = _normalize_strategy(strategy)
        self.fixed_k = _normalize_fixed_k(fixed_k)
        self.diversity_strength = _normalize_diversity_strength(
            diversity_strength
        )

    @staticmethod
    def _validate_active_candidates(
        *,
        xyz: torch.Tensor,
        features: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
    ) -> tuple[int, dict[str, torch.Tensor]]:
        return GaussianCandidateActiveTopKV1._validate_active_candidate_contract(
            xyz=xyz,
            features=features,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
        )

    @staticmethod
    def _active_intrinsics(
        camera: Any,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        values = (camera.fx, camera.fy, camera.cx, camera.cy)
        for name, value in zip(("fx", "fy", "cx", "cy"), values):
            if isinstance(value, torch.Tensor) and value.numel() != 1:
                raise ActiveConfidenceEvidenceError(
                    "invalid_camera_projection",
                    f"Camera {name} must be one scalar.",
                )
        has_device_tensor = any(
            isinstance(value, torch.Tensor) and value.device.type != "cpu"
            for value in values
        )
        if has_device_tensor:
            scalars = tuple(
                torch.as_tensor(value, device=device, dtype=dtype).reshape(())
                for value in values
            )
            return torch.stack(scalars)
        host_scalars = tuple(
            torch.as_tensor(
                value.detach() if isinstance(value, torch.Tensor) else value,
                device="cpu",
                dtype=dtype,
            ).reshape(())
            for value in values
        )
        return torch.stack(host_scalars).to(device=device)

    @classmethod
    def _active_project(
        cls,
        xyz: torch.Tensor,
        camera: Any,
    ) -> dict[str, torch.Tensor]:
        pose = getattr(camera, "pose", None)
        if not isinstance(pose, torch.Tensor) or tuple(pose.shape) != (4, 4):
            raise ActiveConfidenceEvidenceError(
                "invalid_camera_projection",
                "Camera pose must be a [4,4] W2C tensor.",
            )
        pose = pose.detach().to(device=xyz.device, dtype=xyz.dtype)
        intrinsics = cls._active_intrinsics(
            camera,
            device=xyz.device,
            dtype=xyz.dtype,
        )
        camera_xyz = xyz @ pose[:3, :3].transpose(0, 1) + pose[:3, 3]
        z = camera_xyz[:, 2]
        positive_z = z > 0
        safe_z = torch.where(positive_z, z, torch.ones_like(z))
        fx, fy, cx, cy = intrinsics.unbind()
        u_float = fx * camera_xyz[:, 0] / safe_z + cx
        v_float = fy * camera_xyz[:, 1] / safe_z + cy
        coordinates_finite = torch.isfinite(u_float) & torch.isfinite(v_float)
        safe_u = torch.where(coordinates_finite, u_float, torch.zeros_like(u_float))
        safe_v = torch.where(coordinates_finite, v_float, torch.zeros_like(v_float))
        intrinsics_valid = torch.isfinite(intrinsics).all() & (fx > 0) & (fy > 0)
        valid = (
            torch.isfinite(pose).all()
            & intrinsics_valid
            & torch.isfinite(camera_xyz).all(dim=1)
            & positive_z
            & coordinates_finite
        )
        return {
            "u": torch.round(safe_u).to(dtype=torch.long),
            "v": torch.round(safe_v).to(dtype=torch.long),
            "z": z,
            "valid": valid,
        }

    @staticmethod
    def _active_source_depth(
        camera: Any,
        depthmap: Any,
        depth_source: str,
        *,
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
                f"Unsupported depth_source={depth_source!r}.",
            )
        if source is None:
            raise ActiveConfidenceEvidenceError(
                "source_depth_unavailable",
                f"Source depth is missing for depth_source={depth_source!r}.",
            )
        source = torch.as_tensor(source, device=device, dtype=dtype).detach()
        if source.ndim != 2:
            raise ActiveConfidenceEvidenceError(
                "source_depth_shape_mismatch",
                f"Source depth must have shape [H,W], got {list(source.shape)}.",
            )
        return source

    def _active_confidence_components(
        self,
        *,
        xyz: torch.Tensor,
        camera: Any,
        depthmap: Any,
        depth_source: str,
        projection: Mapping[str, torch.Tensor],
    ) -> dict[str, Any]:
        source_frame_id = getattr(camera, "source_frame_id", None)
        source_timestamp = getattr(camera, "source_timestamp", None)
        buffer_index = getattr(camera, "buffer_index", None)
        if source_frame_id is None or source_timestamp is None or buffer_index is None:
            raise ActiveConfidenceEvidenceError(
                "missing_confidence",
                "Camera has no complete immutable confidence identity.",
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

        try:
            identity_checks = (
                int(snapshot.buffer_index) == int(buffer_index),
                int(snapshot.source_frame_id) == int(source_frame_id),
                float(snapshot.source_timestamp) == float(source_timestamp),
                int(snapshot.confidence_source_frame_id) == int(source_frame_id),
            )
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            raise ActiveConfidenceEvidenceError(
                "missing_confidence",
                "Confidence snapshot identity metadata is incomplete.",
            ) from error
        if not all(identity_checks):
            raise ActiveConfidenceEvidenceError(
                "confidence_identity_mismatch",
                "Confidence snapshot identity does not match the source Camera.",
            )
        if (
            not bool(getattr(snapshot, "is_current", False))
            or bool(getattr(snapshot, "is_stale", True))
            or int(getattr(snapshot, "confidence_version", 0)) <= 0
        ):
            raise ActiveConfidenceEvidenceError(
                "stale_confidence",
                "Confidence snapshot is stale.",
            )
        confidence = getattr(snapshot, "confidence", None)
        if not isinstance(confidence, torch.Tensor):
            raise ActiveConfidenceEvidenceError(
                "missing_confidence",
                "Confidence snapshot contains no tensor.",
            )
        if confidence.ndim != 2 or tuple(getattr(snapshot, "shape", ())) != tuple(confidence.shape):
            raise ActiveConfidenceEvidenceError(
                "confidence_shape_mismatch",
                "Confidence tensor or snapshot shape metadata is invalid.",
            )
        if confidence.device != xyz.device or str(getattr(snapshot, "device", None)) != str(confidence.device):
            raise ActiveConfidenceEvidenceError(
                "confidence_device_mismatch",
                "Confidence/candidate device or metadata mismatch.",
            )
        if (
            not confidence.dtype.is_floating_point
            or str(getattr(snapshot, "dtype", None)) != str(confidence.dtype)
        ):
            raise ActiveConfidenceEvidenceError(
                "confidence_dtype_mismatch",
                "Confidence dtype or metadata is invalid.",
            )
        if bool(getattr(snapshot, "requires_grad", not confidence.requires_grad)) != bool(confidence.requires_grad):
            raise ActiveConfidenceEvidenceError(
                "confidence_requires_grad_mismatch",
                "Confidence requires_grad metadata mismatch.",
            )

        source_depth = self._active_source_depth(
            camera,
            depthmap,
            depth_source,
            device=xyz.device,
            dtype=xyz.dtype,
        )
        source_height, source_width = map(int, source_depth.shape)
        camera_shape = (int(camera.image_height), int(camera.image_width))
        if (source_height, source_width) != camera_shape:
            raise ActiveConfidenceEvidenceError(
                "source_depth_camera_shape_mismatch",
                "Source depth and Camera image shapes differ.",
            )
        u = projection["u"]
        v = projection["v"]
        z = projection["z"]
        source_in_bounds = (
            projection["valid"]
            & (u >= 0)
            & (u < source_width)
            & (v >= 0)
            & (v < source_height)
        )
        safe_u = torch.clamp(u, 0, source_width - 1)
        safe_v = torch.clamp(v, 0, source_height - 1)
        sampled_depth = source_depth[safe_v, safe_u]
        depth_tolerance = 1.0e-3 + 1.0e-3 * torch.abs(z)
        lineage_valid = (
            source_in_bounds
            & torch.isfinite(sampled_depth)
            & (sampled_depth > 0)
            & (torch.abs(sampled_depth - z) <= depth_tolerance)
        )

        confidence_height, confidence_width = map(int, confidence.shape)
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
        cell_in_bounds = (
            source_in_bounds
            & (confidence_u >= 0)
            & (confidence_u < confidence_width)
            & (confidence_v >= 0)
            & (confidence_v < confidence_height)
        )
        safe_confidence_u = torch.clamp(confidence_u, 0, confidence_width - 1)
        safe_confidence_v = torch.clamp(confidence_v, 0, confidence_height - 1)
        sampled = confidence[safe_confidence_v, safe_confidence_u]
        valid = lineage_valid & cell_in_bounds & torch.isfinite(sampled)
        normalized = torch.clamp(sampled / CONFIDENCE_NORMALIZER, 0.0, 1.0)
        cell_ids = confidence_v * confidence_width + confidence_u
        return {
            "raw": sampled,
            "normalized": normalized,
            "valid": valid,
            "cell_ids": cell_ids,
            "cell_valid": cell_in_bounds,
            "metadata": {
                "confidence_version": int(snapshot.confidence_version),
                "confidence_source_frame_id": int(
                    snapshot.confidence_source_frame_id
                ),
                "confidence_sampling_method": CONFIDENCE_SAMPLING_METHOD,
                "source_resolution": [source_height, source_width],
                "confidence_resolution": [
                    confidence_height,
                    confidence_width,
                ],
            },
        }

    def _active_render_components(
        self,
        *,
        xyz: torch.Tensor,
        camera: Any,
        gaussian_before: int,
        mapper_update_id: int,
        evidence: Optional[PreinsertRenderEvidenceV1],
        projection: Mapping[str, torch.Tensor],
    ) -> tuple[Optional[dict[str, torch.Tensor]], str]:
        if evidence is None:
            return None, "missing_preinsert_render_evidence"
        try:
            evidence.validate_for_event(
                camera=camera,
                gaussian_count_current=gaussian_before,
                mapper_update_id=mapper_update_id,
            )
        except PreinsertRenderEvidenceError:
            return None, "invalid_preinsert_render_evidence"
        if not evidence.available:
            return None, str(evidence.unavailable_reason)

        height, width = int(evidence.height), int(evidence.width)
        u = projection["u"]
        v = projection["v"]
        z = projection["z"]
        in_bounds = (
            projection["valid"]
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )
        safe_u = torch.clamp(u, 0, width - 1)
        safe_v = torch.clamp(v, 0, height - 1)
        mask = self._mask(camera, device=xyz.device, height=height, width=width)
        mask_sampled = mask[safe_v, safe_u]
        alpha_raw = evidence.alpha_accum[0, safe_v, safe_u]
        alpha_valid = in_bounds & torch.isfinite(alpha_raw)
        alpha = torch.clamp(alpha_raw, 0.0, 1.0)
        coverage = 1.0 - alpha
        residual_domain = alpha_valid & mask_sampled

        render_rgb = evidence.render_rgb[:, safe_v, safe_u].transpose(0, 1)
        original = getattr(camera, "original_image", None)
        if (
            not isinstance(original, torch.Tensor)
            or original.device != xyz.device
            or tuple(original.shape) != (3, height, width)
        ):
            rgb_valid = torch.zeros_like(alpha_valid)
            rgb_l1 = torch.zeros_like(alpha)
        else:
            original_sampled = (
                original[:, safe_v, safe_u]
                .transpose(0, 1)
                .to(dtype=render_rgb.dtype)
                / 255.0
            )
            rgb_l1 = torch.mean(torch.abs(render_rgb - original_sampled), dim=1)
            rgb_valid = (
                residual_domain
                & torch.isfinite(render_rgb).all(dim=1)
                & torch.isfinite(original_sampled).all(dim=1)
                & torch.isfinite(rgb_l1)
            )
        rgb_l1 = torch.clamp(rgb_l1, 0.0, 1.0)

        depth_accum = evidence.depth_accum[0, safe_v, safe_u]
        old_depth = depth_accum / torch.clamp(alpha, min=NUMERICAL_EPSILON)
        depth_relative_error = torch.abs(old_depth - z) / torch.clamp(
            z,
            min=NUMERICAL_EPSILON,
        )
        depth_valid = (
            residual_domain
            & (alpha > 0)
            & torch.isfinite(depth_accum)
            & torch.isfinite(old_depth)
            & (old_depth > 0)
            & torch.isfinite(z)
            & (z > 0)
            & torch.isfinite(depth_relative_error)
        )
        depth_saturated = depth_relative_error / (1.0 + depth_relative_error)
        return {
            "alpha": alpha,
            "coverage": coverage,
            "coverage_valid": alpha_valid,
            "rgb_l1": rgb_l1,
            "rgb_valid": rgb_valid,
            "depth_saturated": depth_saturated,
            "depth_valid": depth_valid,
        }, "available"

    @staticmethod
    def _stable_topk(scores: torch.Tensor, fixed_k: int) -> torch.Tensor:
        ranked = torch.argsort(-scores, stable=True)
        return torch.sort(ranked[:fixed_k]).values

    @staticmethod
    def _index_candidates(
        tensors: Mapping[str, torch.Tensor],
        selected_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            name: torch.index_select(value, 0, selected_indices)
            for name, value in tensors.items()
        }

    @staticmethod
    def _count_active(mask: torch.Tensor) -> int:
        return int(torch.count_nonzero(mask).detach().item())

    @staticmethod
    def _weighted_evidence_quality(
        render_components: Mapping[str, torch.Tensor],
        confidence_components: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return continuous alpha-supported evidence quality on-device."""

        coverage_valid = render_components["coverage_valid"]
        rgb_valid = render_components["rgb_valid"] & confidence_components[
            "valid"
        ]
        depth_valid = render_components["depth_valid"] & confidence_components[
            "valid"
        ]
        alpha = render_components["alpha"]
        confidence = confidence_components["normalized"]
        zeros = torch.zeros_like(render_components["coverage"])
        numerator = torch.where(
            coverage_valid,
            render_components["coverage"],
            zeros,
        )
        denominator = coverage_valid.to(dtype=numerator.dtype)
        numerator = numerator + torch.where(
            rgb_valid,
            alpha * confidence * render_components["rgb_l1"],
            zeros,
        )
        denominator = denominator + alpha * rgb_valid.to(dtype=alpha.dtype)
        numerator = numerator + torch.where(
            depth_valid,
            alpha * confidence * render_components["depth_saturated"],
            zeros,
        )
        denominator = denominator + alpha * depth_valid.to(dtype=alpha.dtype)
        score_valid = denominator > 0
        quality = torch.where(
            score_valid,
            numerator / torch.clamp(denominator, min=NUMERICAL_EPSILON),
            zeros,
        )
        return quality, score_valid, rgb_valid, depth_valid

    def _projected_cell_scores(
        self,
        quality: torch.Tensor,
        *,
        cell_ids: torch.Tensor,
        cell_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Apply an empirical current-batch projected-cell rank penalty."""
        candidate_count = int(quality.shape[0])
        original_indices = torch.arange(
            candidate_count,
            dtype=torch.long,
            device=quality.device,
        )
        invalid_unique_cells = cell_ids.amax() + 1 + original_indices
        grouping_cells = torch.where(cell_valid, cell_ids, invalid_unique_cells)
        quality_order = torch.argsort(-quality, stable=True)
        grouped_order = quality_order[
            torch.argsort(grouping_cells[quality_order], stable=True)
        ]
        grouped_cells = grouping_cells[grouped_order]
        positions = torch.arange(
            candidate_count,
            dtype=torch.long,
            device=quality.device,
        )
        starts = torch.ones(
            candidate_count,
            dtype=torch.bool,
            device=quality.device,
        )
        starts[1:] = grouped_cells[1:] != grouped_cells[:-1]
        start_positions = torch.where(starts, positions, torch.zeros_like(positions))
        last_start = torch.cummax(start_positions, dim=0).values
        rank = (positions - last_start + 1).to(dtype=quality.dtype)
        rank_penalty = self.diversity_strength * (1.0 - torch.reciprocal(rank))
        adjusted = quality.clone()
        adjusted[grouped_order] = quality[grouped_order] - rank_penalty
        return adjusted

    def _active_base_fields(
        self,
        *,
        camera: Any,
        mapper_update_id: int,
        candidate_count: int,
        gaussian_before: int,
        init: bool,
    ) -> dict[str, Any]:
        sequence = self._next_event_sequence()
        identity = self._camera_identity_fields(camera)
        return {
            "schema": SCHEMA_VERSION,
            "event_type": "candidate_active_selection_v2",
            "event_id": (
                f"gcs-v2-active:{sequence}:{mapper_update_id}:"
                f"{identity['source_camera_id']}"
            ),
            "event_sequence": sequence,
            "mode": ACTIVE_FIXED_K_MODE,
            "strategy": self.strategy,
            "fixed_k": self.fixed_k,
            "diversity_strength": self.diversity_strength,
            "candidate_count": candidate_count,
            "selected_count": candidate_count,
            "selected_unique_cell_count": None,
            "fallback_reason": None,
            "alpha_valid_count": 0,
            "coverage_valid_count": 0,
            "confidence_valid_count": 0,
            "rgb_l1_valid_count": 0,
            "depth_relative_error_valid_count": 0,
            "cell_valid_count": 0,
            "multiplicity_valid_count": 0,
            "cell_redundancy_valid_count": 0,
            "selector_wall_time_ms": 0.0,
            "init_bypass": bool(init),
            "selected_indices_created": False,
            "gaussian_before": int(gaussian_before),
            "gaussian_after_extend": None,
            "actual_admitted_count": None,
            "actual_dropped_count": None,
            "m01_second_selection_applied": False,
            "conservation_check": None,
            "status": "ok",
            "reason": None,
            "error": None,
            **identity,
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
        preinsert_render_evidence: Optional[PreinsertRenderEvidenceV1],
    ) -> CandidateActiveSelectionResultV2:
        started_ns = time.perf_counter_ns()
        candidate_count, tensors = self._validate_active_candidates(
            xyz=xyz,
            features=features,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
        )
        input_state = self._tensor_state(tensors)
        fields = self._active_base_fields(
            camera=camera,
            mapper_update_id=mapper_update_id,
            candidate_count=candidate_count,
            gaussian_before=gaussian_before,
            init=init,
        )
        selected_indices: Optional[torch.Tensor] = None
        output = tensors

        if init:
            fields["reason"] = "init_bypass"
        elif candidate_count <= self.fixed_k:
            fields["reason"] = (
                "empty_candidate" if candidate_count == 0 else "keep_all_n_le_k"
            )
        else:
            projection = self._active_project(xyz, camera)
            render_components, evidence_reason = self._active_render_components(
                xyz=xyz,
                camera=camera,
                gaussian_before=gaussian_before,
                mapper_update_id=mapper_update_id,
                evidence=preinsert_render_evidence,
                projection=projection,
            )
            confidence_components = None
            confidence_reason = "not_required"
            if self.strategy != "coverage":
                try:
                    confidence_components = self._active_confidence_components(
                        xyz=xyz,
                        camera=camera,
                        depthmap=depthmap,
                        depth_source=depth_source,
                        projection=projection,
                    )
                except ActiveConfidenceEvidenceError as error:
                    confidence_reason = error.reason
                else:
                    confidence_reason = "current"

            uniform = deterministic_uniform_indices(
                candidate_count=candidate_count,
                fixed_budget=self.fixed_k,
                device=xyz.device,
            )
            negative_infinity = torch.full(
                (candidate_count,),
                -torch.inf,
                dtype=xyz.dtype,
                device=xyz.device,
            )
            coverage_valid_count = 0
            confidence_valid_count = 0
            rgb_valid_count = 0
            depth_valid_count = 0
            cell_valid_count = 0

            if render_components is not None:
                coverage_valid_count = self._count_active(
                    render_components["coverage_valid"]
                )
                rgb_valid_count = self._count_active(render_components["rgb_valid"])
                depth_valid_count = self._count_active(
                    render_components["depth_valid"]
                )
            if confidence_components is not None:
                confidence_valid_count = self._count_active(
                    confidence_components["valid"]
                )
                cell_valid_count = self._count_active(
                    confidence_components["cell_valid"]
                )
            render_usable = (
                render_components is not None and coverage_valid_count > 0
            )
            if render_components is not None and not render_usable:
                evidence_reason = "no_valid_coverage"

            if self.strategy == "coverage":
                if not render_usable:
                    selected_indices = uniform
                    fields["fallback_reason"] = evidence_reason
                    fields["reason"] = "deterministic_uniform_fallback"
                else:
                    scores = torch.where(
                        render_components["coverage_valid"],
                        render_components["coverage"],
                        negative_infinity,
                    )
                    selected_indices = self._stable_topk(scores, self.fixed_k)
                    fields["reason"] = "coverage_fixed_k"
            elif not render_usable:
                if confidence_components is None or confidence_valid_count == 0:
                    selected_indices = uniform
                    fields["fallback_reason"] = (
                        f"{evidence_reason}+{confidence_reason}"
                    )
                    fields["reason"] = "deterministic_uniform_fallback"
                else:
                    scores = torch.where(
                        confidence_components["valid"],
                        confidence_components["raw"],
                        negative_infinity,
                    )
                    selected_indices = self._stable_topk(scores, self.fixed_k)
                    fields["fallback_reason"] = evidence_reason
                    fields["reason"] = "confidence_topk_fallback"
            elif confidence_components is None or confidence_valid_count == 0:
                if coverage_valid_count == 0:
                    selected_indices = uniform
                    fields["fallback_reason"] = (
                        f"invalid_coverage+{confidence_reason}"
                    )
                    fields["reason"] = "deterministic_uniform_fallback"
                else:
                    scores = torch.where(
                        render_components["coverage_valid"],
                        render_components["coverage"],
                        negative_infinity,
                    )
                    selected_indices = self._stable_topk(scores, self.fixed_k)
                    fields["fallback_reason"] = confidence_reason
                    fields["reason"] = "coverage_only_fallback"
            else:
                quality, score_valid, _, _ = self._weighted_evidence_quality(
                    render_components,
                    confidence_components,
                )
                quality = torch.where(score_valid, quality, negative_infinity)
                if self.strategy == "evidence_projected_cell":
                    quality = self._projected_cell_scores(
                        quality,
                        cell_ids=confidence_components["cell_ids"],
                        cell_valid=confidence_components["cell_valid"],
                    )
                selected_indices = self._stable_topk(quality, self.fixed_k)
                fields["reason"] = f"{self.strategy}_fixed_k"

            output = self._index_candidates(tensors, selected_indices)
            fields.update(
                {
                    "selected_count": int(selected_indices.shape[0]),
                    "selected_indices_created": True,
                    "alpha_valid_count": coverage_valid_count,
                    "coverage_valid_count": coverage_valid_count,
                    "confidence_valid_count": confidence_valid_count,
                    "rgb_l1_valid_count": rgb_valid_count,
                    "depth_relative_error_valid_count": depth_valid_count,
                    "cell_valid_count": cell_valid_count,
                    "multiplicity_valid_count": cell_valid_count,
                    "cell_redundancy_valid_count": cell_valid_count,
                }
            )
            if confidence_components is not None:
                selected_cells = confidence_components["cell_ids"][selected_indices]
                selected_cell_valid = confidence_components["cell_valid"][
                    selected_indices
                ]
                fields["selected_unique_cell_count"] = int(
                    torch.unique(selected_cells[selected_cell_valid]).shape[0]
                )

        if self._tensor_state(tensors) != input_state:
            raise RuntimeError("GCS-v2 active selector mutated candidate inputs.")
        retained = int(output["xyz"].shape[0])
        if any(int(value.shape[0]) != retained for value in output.values()):
            raise RuntimeError("GCS-v2 selected candidate fields are misaligned.")
        fields["selected_count"] = retained
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        fields["selector_wall_time_ms"] = (
            elapsed_ms if math.isfinite(elapsed_ms) else None
        )
        return CandidateActiveSelectionResultV2(
            xyz=output["xyz"],
            features=output["features"],
            scales=output["scales"],
            rotations=output["rotations"],
            opacities=output["opacities"],
            selected_indices=selected_indices,
            token=CandidateActiveSelectionTokenV2(fields=fields),
        )

    def observe_active_empty(
        self,
        *,
        camera: Any,
        mapper_update_id: int,
        init: bool,
        gaussian_before: int,
    ) -> CandidateActiveSelectionTokenV2:
        fields = self._active_base_fields(
            camera=camera,
            mapper_update_id=mapper_update_id,
            candidate_count=0,
            gaussian_before=gaussian_before,
            init=init,
        )
        fields["reason"] = "init_bypass" if init else "empty_candidate"
        return CandidateActiveSelectionTokenV2(fields=fields)

    def record_active_after_extend(
        self,
        token: CandidateActiveSelectionTokenV2,
        *,
        admitted_candidate_count: int,
        dropped_candidate_count: int,
        gaussian_after_extend: int,
        m01_second_selection_applied: bool,
    ) -> CandidateActiveSelectionSummaryV2:
        fields = dict(token.fields)
        admitted = int(admitted_candidate_count)
        gaussian_after = int(gaussian_after_extend)
        second_selection = bool(m01_second_selection_applied)
        conservation = (
            admitted == int(fields["selected_count"])
            and int(dropped_candidate_count) == 0
            and gaussian_after == int(fields["gaussian_before"]) + admitted
            and not second_selection
        )
        fields.update(
            {
                "actual_admitted_count": admitted,
                "actual_dropped_count": int(fields["candidate_count"]) - admitted,
                "gaussian_after_extend": gaussian_after,
                "m01_second_selection_applied": second_selection,
                "conservation_check": conservation,
            }
        )
        if not conservation:
            fields.update(
                {
                    "status": "error",
                    "reason": "active_selection_conservation_failed",
                    "error": {
                        "type": "ActiveSelectionConservationError",
                        "message": (
                            "V2 fixed-K selection, M01 bypass, or Gaussian count "
                            "conservation failed."
                        ),
                    },
                }
            )
        summary = CandidateActiveSelectionSummaryV2(fields=fields)
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
                    "schema": SCHEMA_VERSION,
                    "event_type": "candidate_active_selection_v2",
                    "event_id": event.get("event_id", "gcs-v2-active:fallback"),
                    "event_sequence": event.get("event_sequence"),
                    "mode": ACTIVE_FIXED_K_MODE,
                    "strategy": event.get("strategy"),
                    "fixed_k": event.get("fixed_k"),
                    "candidate_count": event.get("candidate_count", 0),
                    "selected_count": event.get("selected_count", 0),
                    "status": "error",
                    "reason": "active_event_serialization_failed",
                    "error": _structured_error(error),
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        print(f"{LOG_PREFIX} {payload}", flush=True)


class GaussianCandidateDynamicKObserveV2(GaussianCandidateSelectorV2):
    """Observe frozen evidence-q histograms without selecting candidates."""

    mode = DYNAMIC_K_OBSERVE_MODE
    is_active = False
    # The borrowed confidence helper dispatches this static dependency via self.
    _active_source_depth = staticmethod(
        GaussianCandidateActiveFixedKV2._active_source_depth
    )

    def __init__(
        self,
        *,
        confidence_snapshot_getter: Callable[..., Any],
        logging_enabled: bool,
        device: Any,
        dynamic_budget_config: DynamicBudgetObserveConfigV1,
    ) -> None:
        super().__init__(
            confidence_snapshot_getter=confidence_snapshot_getter,
            logging_enabled=logging_enabled,
            device=device,
        )
        self.strategy = "evidence"
        self.dynamic_budget_observer = GaussianCandidateDynamicBudgetObserverV1(
            dynamic_budget_config
        )

    def _dynamic_base_fields(
        self,
        *,
        camera: Any,
        mapper_update_id: int,
        candidate_count: int,
        gaussian_before: int,
        init: bool,
    ) -> dict[str, Any]:
        sequence = self._next_event_sequence()
        fields = self._base_fields(
            sequence=sequence,
            camera=camera,
            mapper_update_id=mapper_update_id,
            candidate_count=candidate_count,
            gaussian_before=gaussian_before,
            init=init,
        )
        fields.update(
            {
                "event_type": "candidate_dynamic_budget_observe_v1",
                "event_id": (
                    f"gcs-v2-dynamic-observe:{sequence}:{mapper_update_id}:"
                    f"{fields['source_camera_id']}"
                ),
                "mode": DYNAMIC_K_OBSERVE_MODE,
                "strategy": "evidence",
                "selection_applied": False,
                "fallback_reason": None,
                "q_valid_count": 0,
                "q_invalid_count": int(candidate_count),
                "q_histogram_bins": (
                    self.dynamic_budget_observer.observe_histogram_bins
                ),
                "q_histogram_counts": [
                    0
                ] * self.dynamic_budget_observer.observe_histogram_bins,
                "q_histogram_range": [0.0, 1.0],
                "count_above_bin_edges_recoverable": True,
                "k_max_reference": (
                    self.dynamic_budget_observer.k_max_reference
                ),
                "diagnostic_gpu_to_cpu_sync_expected": False,
                "controller_wall_time_ms": 0.0,
            }
        )
        return fields

    def observe_before_extend(
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
        preinsert_render_evidence: Optional[PreinsertRenderEvidenceV1],
    ) -> CandidateMarginalUtilityTokenV2:
        started_ns = time.perf_counter_ns()
        candidate_count, tensors = (
            GaussianCandidateActiveFixedKV2._validate_active_candidates(
                xyz=xyz,
                features=features,
                scales=scales,
                rotations=rotations,
                opacities=opacities,
            )
        )
        input_state = self._tensor_state(tensors)
        fields = self._dynamic_base_fields(
            camera=camera,
            mapper_update_id=mapper_update_id,
            candidate_count=candidate_count,
            gaussian_before=gaussian_before,
            init=init,
        )

        if init:
            fields.update(
                reason="init_bypass",
                fallback_reason="init_bypass",
                evidence_reason="init_bypass",
                confidence_reason="init_bypass",
            )
        elif candidate_count == 0:
            fields.update(
                reason="empty_candidate",
                fallback_reason="empty_candidate",
                evidence_reason="empty_candidate",
                confidence_reason="empty_candidate",
            )
        else:
            try:
                projection = GaussianCandidateActiveFixedKV2._active_project(
                    xyz, camera
                )
                render_components, evidence_reason = (
                    GaussianCandidateActiveFixedKV2._active_render_components(
                        self,
                        xyz=xyz,
                        camera=camera,
                        gaussian_before=gaussian_before,
                        mapper_update_id=mapper_update_id,
                        evidence=preinsert_render_evidence,
                        projection=projection,
                    )
                )
            except Exception as error:
                fields.update(
                    reason="dynamic_q_unavailable",
                    fallback_reason="invalid_candidate_projection",
                    evidence_reason="invalid_candidate_projection",
                    evidence_error=_structured_error(error),
                )
                render_components = None
                evidence_reason = "invalid_candidate_projection"

            fields["evidence_reason"] = evidence_reason
            if render_components is None:
                fields.update(
                    reason="dynamic_q_unavailable",
                    fallback_reason=evidence_reason,
                    confidence_reason="not_read_without_evidence",
                )
            else:
                fields["evidence_available"] = True
                try:
                    confidence_components = (
                        GaussianCandidateActiveFixedKV2._active_confidence_components(
                            self,
                            xyz=xyz,
                            camera=camera,
                            depthmap=depthmap,
                            depth_source=depth_source,
                            projection=projection,
                        )
                    )
                except ActiveConfidenceEvidenceError as error:
                    fields.update(
                        reason="dynamic_q_unavailable",
                        fallback_reason=error.reason,
                        confidence_reason=error.reason,
                        confidence_error=_structured_error(error),
                    )
                else:
                    fields.update(
                        evidence_reason="available",
                        confidence_available=True,
                        confidence_reason="current",
                    )
                    quality, quality_valid, _, _ = (
                        GaussianCandidateActiveFixedKV2._weighted_evidence_quality(
                            render_components,
                            confidence_components,
                        )
                    )
                    quality_valid = quality_valid & torch.any(
                        confidence_components["valid"]
                    )
                    histogram = self.dynamic_budget_observer.observe(
                        quality,
                        quality_valid,
                    )
                    fields.update(histogram.to_fields())
                    if histogram.q_valid_count == 0:
                        fields.update(
                            reason="dynamic_q_unavailable",
                            fallback_reason="no_valid_evidence_quality",
                        )
                    else:
                        fields["reason"] = "dynamic_q_histogram_observed"

        if self._tensor_state(tensors) != input_state:
            raise RuntimeError("GCS-v2 dynamic observe mutated candidate inputs.")
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        fields["host_wall_elapsed_ms"] = (
            elapsed_ms if math.isfinite(elapsed_ms) else None
        )
        return CandidateMarginalUtilityTokenV2(fields=fields)

    def observe_empty(
        self,
        *,
        camera: Any,
        mapper_update_id: int,
        init: bool,
        gaussian_before: int,
        preinsert_render_evidence: Optional[PreinsertRenderEvidenceV1],
    ) -> CandidateMarginalUtilityTokenV2:
        del preinsert_render_evidence
        fields = self._dynamic_base_fields(
            camera=camera,
            mapper_update_id=mapper_update_id,
            candidate_count=0,
            gaussian_before=gaussian_before,
            init=init,
        )
        reason = "init_bypass" if init else "empty_candidate"
        fields.update(
            reason=reason,
            fallback_reason=reason,
            evidence_reason=reason,
            confidence_reason=reason,
        )
        return CandidateMarginalUtilityTokenV2(fields=fields)

    @staticmethod
    def _emit(event: dict[str, Any]) -> None:
        """Best-effort diagnostic logging must never interrupt admission."""

        try:
            payload = json.dumps(
                event,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            print(f"{LOG_PREFIX} {payload}", flush=True)
        except Exception:
            return
