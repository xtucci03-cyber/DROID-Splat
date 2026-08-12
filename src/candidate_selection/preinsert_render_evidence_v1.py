"""Short-lived, observe-only render evidence for candidate admission events.

The tensor semantics are frozen to the repository rasterizer gitlink
43e21bff91cd24986ee3dd52fe0bb06952e50ec7:

* ``depth_accum`` is the unnormalized alpha-weighted camera-z sum
  ``sum(T_i * alpha_i * camera_z_i)``.
* ``alpha_accum`` is accumulated alpha, ``1 - final_transmittance``.

The installed server binary provenance remains unverified until the dedicated
CUDA gate.  This module neither normalizes depth nor assigns quality scores.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, Callable, Mapping, Optional

import torch


EVIDENCE_SEMANTICS_VERSION = 1
RASTERIZER_SOURCE_GITLINK = "43e21bff91cd24986ee3dd52fe0bb06952e50ec7"
RASTERIZER_BINARY_PROVENANCE = "unverified"
SUPPORTED_MODES = frozenset({"off", "observe"})
_CONFIG_FIELDS = frozenset({"mode"})


class PreinsertRenderEvidenceError(RuntimeError):
    """Fail-closed evidence configuration, rendering, or contract error."""


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise PreinsertRenderEvidenceError(f"{field} must be an integer.")
    normalized = int(value)
    if normalized < minimum:
        raise PreinsertRenderEvidenceError(
            f"{field} must be greater than or equal to {minimum}."
        )
    return normalized


def _camera_identity(camera: Any) -> tuple[int, int, int, float]:
    values = {
        "source_camera_id": getattr(camera, "uid", None),
        "source_buffer_index": getattr(camera, "buffer_index", None),
        "source_frame_id": getattr(camera, "source_frame_id", None),
    }
    normalized = {
        field: _require_int(value, field)
        for field, value in values.items()
    }
    timestamp = getattr(camera, "source_timestamp", None)
    try:
        timestamp = float(timestamp)
    except (TypeError, ValueError, OverflowError) as error:
        raise PreinsertRenderEvidenceError(
            "source_timestamp must be a finite number."
        ) from error
    if not math.isfinite(timestamp):
        raise PreinsertRenderEvidenceError(
            "source_timestamp must be a finite number."
        )
    return (
        normalized["source_camera_id"],
        normalized["source_buffer_index"],
        normalized["source_frame_id"],
        timestamp,
    )


@dataclass(frozen=True)
class PreinsertRenderEvidenceV1:
    """Immutable event-local references to one pre-insertion render."""

    render_rgb: Optional[torch.Tensor]
    depth_accum: Optional[torch.Tensor]
    alpha_accum: Optional[torch.Tensor]
    source_camera_id: int
    source_frame_id: int
    source_buffer_index: int
    source_timestamp: float
    height: int
    width: int
    gaussian_count_before_render: int
    available: bool
    unavailable_reason: Optional[str]
    mapper_update_id: int
    evidence_semantics_version: int = EVIDENCE_SEMANTICS_VERSION
    rasterizer_source_gitlink: str = RASTERIZER_SOURCE_GITLINK
    rasterizer_binary_provenance: str = RASTERIZER_BINARY_PROVENANCE

    def validate_for_event(
        self,
        *,
        camera: Any,
        gaussian_count_current: int,
        mapper_update_id: int,
    ) -> None:
        """Validate identity, freshness, shape, device, dtype, and autograd."""

        camera_id, buffer_index, frame_id, timestamp = _camera_identity(camera)
        expected_identity = (camera_id, buffer_index, frame_id, timestamp)
        evidence_identity = (
            _require_int(self.source_camera_id, "source_camera_id"),
            _require_int(self.source_buffer_index, "source_buffer_index"),
            _require_int(self.source_frame_id, "source_frame_id"),
            float(self.source_timestamp),
        )
        if evidence_identity != expected_identity:
            raise PreinsertRenderEvidenceError(
                "Preinsert render evidence Camera identity mismatch: "
                f"evidence={evidence_identity}, camera={expected_identity}."
            )
        if not math.isfinite(evidence_identity[3]):
            raise PreinsertRenderEvidenceError(
                "Preinsert render evidence timestamp is not finite."
            )
        current_count = _require_int(
            gaussian_count_current,
            "gaussian_count_current",
        )
        evidence_count = _require_int(
            self.gaussian_count_before_render,
            "gaussian_count_before_render",
        )
        if evidence_count != current_count:
            raise PreinsertRenderEvidenceError(
                "Preinsert render evidence is stale: Gaussian count changed "
                f"from {evidence_count} to {current_count}."
            )
        if _require_int(self.mapper_update_id, "mapper_update_id") != _require_int(
            mapper_update_id,
            "mapper_update_id",
        ):
            raise PreinsertRenderEvidenceError(
                "Preinsert render evidence mapper_update_id mismatch."
            )
        if self.evidence_semantics_version != EVIDENCE_SEMANTICS_VERSION:
            raise PreinsertRenderEvidenceError(
                "Unsupported preinsert render evidence semantics version."
            )
        if self.rasterizer_source_gitlink != RASTERIZER_SOURCE_GITLINK:
            raise PreinsertRenderEvidenceError(
                "Preinsert render evidence rasterizer gitlink mismatch."
            )
        if self.rasterizer_binary_provenance != RASTERIZER_BINARY_PROVENANCE:
            raise PreinsertRenderEvidenceError(
                "Preinsert render evidence binary provenance marker mismatch."
            )

        height = _require_int(self.height, "height", minimum=1)
        width = _require_int(self.width, "width", minimum=1)
        camera_shape = (
            _require_int(getattr(camera, "image_height", None), "camera.image_height", minimum=1),
            _require_int(getattr(camera, "image_width", None), "camera.image_width", minimum=1),
        )
        if (height, width) != camera_shape:
            raise PreinsertRenderEvidenceError(
                "Preinsert render evidence/Camera image shape mismatch: "
                f"evidence={(height, width)}, camera={camera_shape}."
            )

        if not isinstance(self.available, bool):
            raise PreinsertRenderEvidenceError("available must be bool.")
        tensors = (self.render_rgb, self.depth_accum, self.alpha_accum)
        if not self.available:
            if any(value is not None for value in tensors):
                raise PreinsertRenderEvidenceError(
                    "Unavailable evidence must not contain render tensors."
                )
            if not isinstance(self.unavailable_reason, str) or not self.unavailable_reason:
                raise PreinsertRenderEvidenceError(
                    "Unavailable evidence requires an explicit reason."
                )
            return

        if self.unavailable_reason is not None:
            raise PreinsertRenderEvidenceError(
                "Available evidence must not contain an unavailable reason."
            )
        if evidence_count == 0:
            raise PreinsertRenderEvidenceError(
                "Available evidence cannot represent an empty old map."
            )
        names_and_shapes = (
            ("render_rgb", self.render_rgb, (3, height, width)),
            ("depth_accum", self.depth_accum, (1, height, width)),
            ("alpha_accum", self.alpha_accum, (1, height, width)),
        )
        devices = set()
        dtypes = set()
        for name, tensor, shape in names_and_shapes:
            if not isinstance(tensor, torch.Tensor):
                raise PreinsertRenderEvidenceError(
                    f"{name} must be a Tensor when evidence is available."
                )
            if tuple(tensor.shape) != shape:
                raise PreinsertRenderEvidenceError(
                    f"{name} shape mismatch: expected {shape}, got "
                    f"{tuple(tensor.shape)}."
                )
            if not tensor.dtype.is_floating_point:
                raise PreinsertRenderEvidenceError(
                    f"{name} must have floating-point dtype."
                )
            if tensor.requires_grad or tensor.grad_fn is not None:
                raise PreinsertRenderEvidenceError(
                    f"{name} must not retain an autograd graph."
                )
            devices.add(tensor.device)
            dtypes.add(tensor.dtype)
        if len(devices) != 1 or len(dtypes) != 1:
            raise PreinsertRenderEvidenceError(
                "Preinsert render evidence tensors must share device and dtype."
            )
        camera_device = torch.device(getattr(camera, "device", next(iter(devices))))
        if next(iter(devices)) != camera_device:
            raise PreinsertRenderEvidenceError(
                "Preinsert render evidence/Camera device mismatch: "
                f"evidence={next(iter(devices))}, camera={camera_device}."
            )


class PreinsertRenderEvidenceObserverV1:
    """Create one non-persistent evidence object per admission event."""

    mode = "observe"

    def __init__(self, *, device: Any) -> None:
        self.device = torch.device(device)

    def unavailable(
        self,
        *,
        camera: Any,
        gaussian_count_before_render: int,
        mapper_update_id: int,
        reason: str,
    ) -> PreinsertRenderEvidenceV1:
        camera_id, buffer_index, frame_id, timestamp = _camera_identity(camera)
        evidence = PreinsertRenderEvidenceV1(
            render_rgb=None,
            depth_accum=None,
            alpha_accum=None,
            source_camera_id=camera_id,
            source_frame_id=frame_id,
            source_buffer_index=buffer_index,
            source_timestamp=timestamp,
            height=_require_int(camera.image_height, "camera.image_height", minimum=1),
            width=_require_int(camera.image_width, "camera.image_width", minimum=1),
            gaussian_count_before_render=_require_int(
                gaussian_count_before_render,
                "gaussian_count_before_render",
            ),
            available=False,
            unavailable_reason=reason,
            mapper_update_id=_require_int(mapper_update_id, "mapper_update_id"),
        )
        evidence.validate_for_event(
            camera=camera,
            gaussian_count_current=gaussian_count_before_render,
            mapper_update_id=mapper_update_id,
        )
        return evidence

    def capture(
        self,
        *,
        camera: Any,
        gaussians: Any,
        renderer: Callable[..., Any],
        pipeline_params: Any,
        background: torch.Tensor,
        mapper_update_id: int,
    ) -> PreinsertRenderEvidenceV1:
        """Render the current old map exactly once without retaining gradients."""

        if not callable(renderer):
            raise PreinsertRenderEvidenceError("renderer must be callable.")
        gaussian_count = _require_int(len(gaussians), "gaussian_count_before_render")
        if gaussian_count == 0:
            return self.unavailable(
                camera=camera,
                gaussian_count_before_render=0,
                mapper_update_id=mapper_update_id,
                reason="empty_old_map",
            )
        with torch.no_grad():
            render_package = renderer(
                camera,
                gaussians,
                pipeline_params,
                background,
                device=str(self.device),
            )
        if not isinstance(render_package, Mapping):
            raise PreinsertRenderEvidenceError(
                "Preinsert renderer returned no render package."
            )
        missing = [
            field
            for field in ("render", "depth", "opacity")
            if field not in render_package
        ]
        if missing:
            raise PreinsertRenderEvidenceError(
                f"Preinsert renderer output is missing fields: {missing}."
            )
        camera_id, buffer_index, frame_id, timestamp = _camera_identity(camera)
        evidence = PreinsertRenderEvidenceV1(
            render_rgb=render_package["render"].detach(),
            depth_accum=render_package["depth"].detach(),
            alpha_accum=render_package["opacity"].detach(),
            source_camera_id=camera_id,
            source_frame_id=frame_id,
            source_buffer_index=buffer_index,
            source_timestamp=timestamp,
            height=_require_int(camera.image_height, "camera.image_height", minimum=1),
            width=_require_int(camera.image_width, "camera.image_width", minimum=1),
            gaussian_count_before_render=gaussian_count,
            available=True,
            unavailable_reason=None,
            mapper_update_id=_require_int(mapper_update_id, "mapper_update_id"),
        )
        evidence.validate_for_event(
            camera=camera,
            gaussian_count_current=len(gaussians),
            mapper_update_id=mapper_update_id,
        )
        return evidence


def build_preinsert_render_evidence_v1(
    config: Optional[Mapping[str, Any]],
    *,
    device: Any,
) -> Optional[PreinsertRenderEvidenceObserverV1]:
    """Build the default-off observer without runtime work in off mode."""

    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise TypeError(
            "mapping.preinsert_render_evidence_v1 must be a mapping."
        )
    unknown = sorted(set(config.keys()) - _CONFIG_FIELDS)
    if unknown:
        raise ValueError(
            "mapping.preinsert_render_evidence_v1 contains unknown fields: "
            f"{unknown}."
        )
    mode_value = config.get("mode", "off")
    if not isinstance(mode_value, str):
        raise TypeError(
            "mapping.preinsert_render_evidence_v1.mode must be a string."
        )
    mode = mode_value.strip().lower()
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            "mapping.preinsert_render_evidence_v1.mode must be one of "
            f"{sorted(SUPPORTED_MODES)}, got {mode!r}."
        )
    if mode == "off":
        return None
    return PreinsertRenderEvidenceObserverV1(device=device)
