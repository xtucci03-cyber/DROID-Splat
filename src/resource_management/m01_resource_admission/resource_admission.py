from dataclasses import dataclass, field
import json
from numbers import Integral
import time
from typing import Any, Dict, Optional

import torch


_MIB = 1024.0 * 1024.0


def deterministic_uniform_indices(
    *,
    candidate_count: int,
    fixed_budget: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the existing M01 deterministic-uniform candidate indices.

    The result is a one-dimensional ``torch.long`` tensor on ``device``.
    Empty inputs return an empty tensor, inputs within budget return every
    original index, and over-budget inputs retain the original inclusive,
    equally spaced floor rule.  The function never mutates an input tensor
    and does not consume random state.
    """

    if isinstance(candidate_count, bool) or not isinstance(
        candidate_count, Integral
    ):
        raise TypeError("candidate_count must be an integer and must not be bool.")
    if candidate_count < 0:
        raise ValueError("candidate_count must be greater than or equal to 0.")
    if isinstance(fixed_budget, bool) or not isinstance(fixed_budget, Integral):
        raise TypeError("fixed_budget must be an integer and must not be bool.")
    if fixed_budget < 1:
        raise ValueError("fixed_budget must be greater than or equal to 1.")

    normalized_device = torch.device(device)
    if candidate_count <= fixed_budget:
        return torch.arange(
            candidate_count,
            dtype=torch.long,
            device=normalized_device,
        )
    if fixed_budget == 1:
        return torch.full(
            (1,),
            (candidate_count - 1) // 2,
            dtype=torch.long,
            device=normalized_device,
        )

    positions = torch.arange(
        fixed_budget,
        dtype=torch.long,
        device=normalized_device,
    )
    return torch.div(
        positions * (candidate_count - 1),
        fixed_budget - 1,
        rounding_mode="floor",
    )


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
    fixed_budget: Optional[int] = None
    selection: Optional[str] = None

    _tensor_metadata: Dict[str, Any] = field(default_factory=dict, repr=False)
    _before_cpu_ns: int = field(default=0, repr=False)


class ResourceAdmission:
    """ResourceAdmission candidate observation and fixed-budget control.

    Disabled mode is represented by None and must not construct this class.
    Generic control mode is intentionally unavailable.
    """

    def __init__(
        self,
        mode: str,
        fixed_budget: Optional[int] = None,
        selection: Optional[str] = None,
    ):
        normalized_mode = str(mode).strip().lower()

        if normalized_mode == "control":
            raise NotImplementedError(
                "ResourceAdmission control mode is not implemented in the "
                "fixed-budget atomic change."
            )

        if normalized_mode == "disabled":
            raise ValueError("ResourceAdmission disabled mode must be represented by None.")

        if normalized_mode not in {"observe", "fixed_budget"}:
            raise ValueError(f"Unsupported ResourceAdmission mode: {normalized_mode!r}")

        self.mode = normalized_mode
        self.fixed_budget: Optional[int] = None
        self.selection: Optional[str] = None

        if self.mode == "fixed_budget":
            if isinstance(fixed_budget, bool) or not isinstance(fixed_budget, Integral):
                raise TypeError("fixed_budget must be an integer and must not be bool.")
            if fixed_budget < 1:
                raise ValueError("fixed_budget must be greater than or equal to 1.")

            normalized_selection = (
                str(selection).strip().lower()
                if selection is not None
                else None
            )

            if normalized_selection != "deterministic_uniform":
                raise ValueError(
                    "fixed_budget selection must be 'deterministic_uniform', "
                    f"got {selection!r}."
                )

            self.fixed_budget = int(fixed_budget)
            self.selection = normalized_selection

    @staticmethod
    def _read_tensor_metadata(name: str, tensor: torch.Tensor) -> Dict[str, Any]:
        return {
            f"{name}_shape": list(tensor.shape),
            f"{name}_dtype": str(tensor.dtype),
            f"{name}_device": str(tensor.device),
            f"{name}_requires_grad": bool(tensor.requires_grad),
        }

    @staticmethod
    def _validate_fixed_budget_contract(
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
        shapes = {name: list(tensor.shape) for name, tensor in tensors.items()}
        devices = {name: str(tensor.device) for name, tensor in tensors.items()}
        candidate_count = int(xyz.shape[0]) if xyz.ndim > 0 else None

        first_dimensions_match = candidate_count is not None and all(
            tensor.ndim > 0 and int(tensor.shape[0]) == candidate_count for tensor in tensors.values()
        )
        devices_match = all(tensor.device == xyz.device for tensor in tensors.values())

        if not first_dimensions_match or not devices_match:
            raise ValueError(
                "Fixed-budget candidate contract violation: "
                f"candidate_count={candidate_count}, shapes={shapes}, devices={devices}."
            )

        return int(candidate_count)

    @staticmethod
    def _deterministic_uniform_indices(
        *,
        candidate_count: int,
        fixed_budget: int,
        device: torch.device,
    ) -> torch.Tensor:
        return deterministic_uniform_indices(
            candidate_count=candidate_count,
            fixed_budget=fixed_budget,
            device=device,
        )

    def admit_before_extend(
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

        if self.mode == "fixed_budget":
            candidate_count = self._validate_fixed_budget_contract(
                xyz=xyz,
                features=features,
                scales=scales,
                rotations=rotations,
                opacities=opacities,
            )
        else:
            candidate_count = int(xyz.shape[0])

        admitted_xyz = xyz
        admitted_features = features
        admitted_scales = scales
        admitted_rotations = rotations
        admitted_opacities = opacities
        admitted_count = candidate_count
        dropped_count = 0
        selected_indices = None
        protected_init = bool(init)
        reason = "observe_passthrough"

        if self.mode == "fixed_budget":
            if init:
                reason = "init_protected"
            elif candidate_count <= self.fixed_budget:
                reason = "within_budget"
            else:
                selected_indices = self._deterministic_uniform_indices(
                    candidate_count=candidate_count,
                    fixed_budget=self.fixed_budget,
                    device=xyz.device,
                )
                admitted_xyz = torch.index_select(xyz, 0, selected_indices)
                admitted_features = torch.index_select(features, 0, selected_indices)
                admitted_scales = torch.index_select(scales, 0, selected_indices)
                admitted_rotations = torch.index_select(rotations, 0, selected_indices)
                admitted_opacities = torch.index_select(opacities, 0, selected_indices)
                admitted_count = self.fixed_budget
                dropped_count = candidate_count - self.fixed_budget
                reason = "fixed_budget_applied"

        tensor_metadata: Dict[str, Any] = {}
        tensor_metadata.update(self._read_tensor_metadata("xyz", admitted_xyz))
        tensor_metadata.update(self._read_tensor_metadata("features", admitted_features))
        tensor_metadata.update(self._read_tensor_metadata("scales", admitted_scales))
        tensor_metadata.update(self._read_tensor_metadata("rotations", admitted_rotations))
        tensor_metadata.update(self._read_tensor_metadata("opacities", admitted_opacities))

        finished_ns = time.perf_counter_ns()
        before_cpu_ns = finished_ns - started_ns

        return AdmissionResult(
            xyz=admitted_xyz,
            features=admitted_features,
            scales=admitted_scales,
            rotations=admitted_rotations,
            opacities=admitted_opacities,
            camera_uid=int(camera_uid),
            kf_id=int(kf_id),
            init=bool(init),
            gaussian_before=int(gaussian_before),
            candidate_count=candidate_count,
            admitted_count=admitted_count,
            dropped_count=dropped_count,
            selected_indices=selected_indices,
            skip_extend=False,
            protected_init=protected_init,
            admission_cpu_ms=before_cpu_ns / 1_000_000.0,
            reason=reason,
            fixed_budget=self.fixed_budget,
            selection=self.selection,
            _tensor_metadata=tensor_metadata,
            _before_cpu_ns=before_cpu_ns,
        )

    def record_after_extend(
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
        if self.mode == "fixed_budget":
            event.update(
                {
                    "fixed_budget": result.fixed_budget,
                    "selection": result.selection,
                    "selected_indices_count": (
                        int(result.selected_indices.shape[0])
                        if result.selected_indices is not None
                        else 0
                    ),
                }
            )
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
