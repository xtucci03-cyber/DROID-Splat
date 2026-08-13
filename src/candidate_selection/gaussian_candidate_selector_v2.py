"""Observe-only marginal-utility component diagnostics for GCS-v2 V2-A.

This module consumes the already-captured pre-insertion render evidence.  It
never renders, ranks, filters, or returns candidate indices.  Projection,
source-depth lineage, current-confidence provenance, low-resolution cell
mapping, and confidence sampling are delegated to the frozen GCS-v1 reader.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from typing import Any, Callable, Mapping, Optional

import torch

from .gaussian_candidate_active_topk_v1 import (
    ActiveConfidenceEvidenceError,
    GaussianCandidateActiveTopKV1,
)
from .preinsert_render_evidence_v1 import (
    PreinsertRenderEvidenceError,
    PreinsertRenderEvidenceV1,
)


LOG_PREFIX = "[GaussianCandidateSelectorV2]"
SCHEMA_VERSION = 1
MODE = "observe"
SUPPORTED_MODES = frozenset({"off", MODE})
NUMERICAL_EPSILON = 1.0e-6
CONFIDENCE_NORMALIZER = math.sqrt(2.0)

_TOP_LEVEL_FIELDS = frozenset({"mode", "logging"})
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


def build_gaussian_candidate_selector_v2(
    config: Optional[Mapping[str, Any]],
    *,
    candidate_selector_v1: Any,
    resource_admission_mode: str,
    preinsert_render_evidence_v1: Any,
    confidence_snapshot_getter: Callable[..., Any],
    device: Any,
) -> Optional["GaussianCandidateSelectorV2"]:
    """Build V2-A; exact off returns ``None`` without runtime state."""

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
    if mode == "off":
        return None
    if not logging_enabled:
        raise ValueError(
            "mapping.candidate_selector_v2.logging.enabled must be true "
            "in observe mode."
        )
    if str(resource_admission_mode).strip().lower() != "disabled":
        raise ValueError(
            "GCS-v2 observe mode requires mapping.resource_admission.mode=disabled."
        )
    if candidate_selector_v1 is not None:
        raise ValueError(
            "GCS-v2 observe mode requires mapping.candidate_selector_v1.mode=off."
        )
    if (
        preinsert_render_evidence_v1 is None
        or getattr(preinsert_render_evidence_v1, "mode", None) != "observe"
    ):
        raise ValueError(
            "GCS-v2 observe mode requires "
            "mapping.preinsert_render_evidence_v1.mode=observe."
        )
    if not callable(confidence_snapshot_getter):
        raise TypeError("confidence_snapshot_getter must be callable.")
    return GaussianCandidateSelectorV2(
        confidence_snapshot_getter=confidence_snapshot_getter,
        logging_enabled=logging_enabled,
        device=device,
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
