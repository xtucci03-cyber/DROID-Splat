from dataclasses import dataclass, field
import json
import time
from typing import Any, Dict, Optional

import torch


_MIB = 1024.0 * 1024.0


@dataclass
class AdmissionResult:
    xyz: torch.Tensor
    features: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor

    camera_uid: int
    kf_id: int
    init: bool
    gaussian_before: int

    candidate_count: int
    admitted_count: int
    dropped_count: int

    selected_indices: Optional[torch.Tensor]
    skip_extend: bool
    protected_init: bool
    admission_cpu_ms: float
    reason: str

    _tensor_metadata: Dict[str, Any] = field(default_factory=dict, repr=False)
    _before_cpu_ns: int = field(default=0, repr=False)


class ResourceAdmission:
    """Observe-only ResourceAdmission implementation.

    Disabled mode is represented by None and must not construct this class.
    Control mode is intentionally unavailable in this atomic change.
    """

    def __init__(self, mode: str):
        normalized_mode = str(mode).strip().lower()

        if normalized_mode == "control":
            raise NotImplementedError(
                "ResourceAdmission control mode is not implemented in the "
                "observe-only atomic change."
            )

        if normalized_mode == "disabled":
            raise ValueError("ResourceAdmission disabled mode must be represented by None.")

        if normalized_mode != "observe":
            raise ValueError(f"Unsupported ResourceAdmission mode: {normalized_mode!r}")

        self.mode = normalized_mode

    @staticmethod
    def _read_tensor_metadata(name: str, tensor: torch.Tensor) -> Dict[str, Any]:
        return {
            f"{name}_shape": list(tensor.shape),
            f"{name}_dtype": str(tensor.dtype),
            f"{name}_device": str(tensor.device),
            f"{name}_requires_grad": bool(tensor.requires_grad),
        }

    def observe_before_extend(
        self,
        *,
        xyz: torch.Tensor,
        features: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        camera_uid: int,
        kf_id: int,
        init: bool,
        gaussian_before: int,
    ) -> AdmissionResult:
        started_ns = time.perf_counter_ns()

        candidate_count = int(xyz.shape[0])

        tensor_metadata: Dict[str, Any] = {}
        tensor_metadata.update(self._read_tensor_metadata("xyz", xyz))
        tensor_metadata.update(self._read_tensor_metadata("features", features))
        tensor_metadata.update(self._read_tensor_metadata("scales", scales))
        tensor_metadata.update(self._read_tensor_metadata("rotations", rotations))
        tensor_metadata.update(self._read_tensor_metadata("opacities", opacities))

        finished_ns = time.perf_counter_ns()
        before_cpu_ns = finished_ns - started_ns

        return AdmissionResult(
            xyz=xyz,
            features=features,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            camera_uid=int(camera_uid),
            kf_id=int(kf_id),
            init=bool(init),
            gaussian_before=int(gaussian_before),
            candidate_count=candidate_count,
            admitted_count=candidate_count,
            dropped_count=0,
            selected_indices=None,
            skip_extend=False,
            protected_init=bool(init),
            admission_cpu_ms=before_cpu_ns / 1_000_000.0,
            reason="observe_passthrough",
            _tensor_metadata=tensor_metadata,
            _before_cpu_ns=before_cpu_ns,
        )

    def observe_after_extend(
        self,
        result: AdmissionResult,
        *,
        gaussian_after_extend: int,
    ) -> None:
        started_ns = time.perf_counter_ns()

        if result.xyz.is_cuda and torch.cuda.is_available():
            cuda_allocated_mib = torch.cuda.memory_allocated(device=result.xyz.device) / _MIB
            cuda_reserved_mib = torch.cuda.memory_reserved(device=result.xyz.device) / _MIB
        else:
            cuda_allocated_mib = None
            cuda_reserved_mib = None

        event: Dict[str, Any] = {
            "schema": 1,
            "event": "extend",
            "mode": self.mode,
            "camera_uid": result.camera_uid,
            "kf_id": result.kf_id,
            "init": result.init,
            "protected_init": result.protected_init,
            "candidate_count": result.candidate_count,
            "admitted_count": result.admitted_count,
            "dropped_count": result.dropped_count,
            "gaussian_before": result.gaussian_before,
            "gaussian_after_extend": int(gaussian_after_extend),
            "cuda_allocated_mib": (
                round(cuda_allocated_mib, 3) if cuda_allocated_mib is not None else None
            ),
            "cuda_reserved_mib": (
                round(cuda_reserved_mib, 3) if cuda_reserved_mib is not None else None
            ),
            "reason": result.reason,
        }
        event.update(result._tensor_metadata)

        finished_ns = time.perf_counter_ns()
        result.admission_cpu_ms = (result._before_cpu_ns + finished_ns - started_ns) / 1_000_000.0
        event["admission_cpu_ms"] = round(result.admission_cpu_ms, 6)

        print(
            "[M01:ResourceAdmission] "
            + json.dumps(
                event,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            flush=True,
        )
