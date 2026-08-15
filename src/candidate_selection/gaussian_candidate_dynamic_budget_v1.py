"""Fixed-size q histograms and bounded GCS-v2 dynamic-budget parameters."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
import time
from typing import Any, Mapping, Optional

import torch


DEFAULT_OBSERVE_HISTOGRAM_BINS = 32
DEFAULT_K_MAX_REFERENCE = 600
HISTOGRAM_RANGE = (0.0, 1.0)
_CONFIG_FIELDS = frozenset({
    "observe_histogram_bins", "k_max_reference", "threshold_bin", "k_min", "k_max",
})


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be a positive integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{field} must be greater than zero.")
    return value


def bounded_dynamic_target_v1(
    *, candidate_count: int, demand_count: int, k_min: int, k_max: int
) -> int:
    """Apply the frozen bounded-demand contract without touching tensors."""

    return min(candidate_count, max(k_min, min(k_max, demand_count)))


@dataclass(frozen=True)
class DynamicBudgetObserveConfigV1:
    observe_histogram_bins: int = DEFAULT_OBSERVE_HISTOGRAM_BINS
    k_max_reference: int = DEFAULT_K_MAX_REFERENCE
    threshold_bin: int = 1
    k_min: int = 256
    k_max: int = 600


def normalize_dynamic_budget_observe_config_v1(
    config: Optional[Mapping[str, Any]],
) -> DynamicBudgetObserveConfigV1:
    field = "mapping.candidate_selector_v2.dynamic_budget"
    config = {} if config is None else config
    if not isinstance(config, Mapping):
        raise TypeError(f"{field} must be a mapping, got {type(config).__name__}.")
    unknown = sorted(set(config) - _CONFIG_FIELDS)
    if unknown:
        raise ValueError(f"{field} contains unknown fields: {unknown}.")
    observe_histogram_bins = _positive_int(
        config.get("observe_histogram_bins", DEFAULT_OBSERVE_HISTOGRAM_BINS),
        f"{field}.observe_histogram_bins")
    threshold_bin = _positive_int(
        config.get("threshold_bin", 1), f"{field}.threshold_bin")
    if threshold_bin >= observe_histogram_bins:
        raise ValueError(
            f"{field}.threshold_bin must be less than observe_histogram_bins."
        )
    k_min = _positive_int(config.get("k_min", 256), f"{field}.k_min")
    k_max = _positive_int(config.get("k_max", 600), f"{field}.k_max")
    if k_min > k_max:
        raise ValueError(f"{field}.k_min must not exceed k_max.")
    return DynamicBudgetObserveConfigV1(
        observe_histogram_bins=observe_histogram_bins,
        k_max_reference=_positive_int(
            config.get("k_max_reference", DEFAULT_K_MAX_REFERENCE),
            f"{field}.k_max_reference",
        ),
        threshold_bin=threshold_bin,
        k_min=k_min,
        k_max=k_max,
    )


@dataclass(frozen=True)
class DynamicBudgetObserveResultV1:
    q_valid_count: int
    q_invalid_count: int
    q_histogram_bins: int
    q_histogram_counts: tuple[int, ...]
    q_histogram_range: tuple[float, float]
    k_max_reference: int
    diagnostic_gpu_to_cpu_sync_expected: bool
    controller_wall_time_ms: Optional[float]

    def to_fields(self) -> dict[str, Any]:
        return {
            "q_valid_count": self.q_valid_count,
            "q_invalid_count": self.q_invalid_count,
            "q_histogram_bins": self.q_histogram_bins,
            "q_histogram_counts": list(self.q_histogram_counts),
            "q_histogram_range": list(self.q_histogram_range),
            "count_above_bin_edges_recoverable": True,
            "k_max_reference": self.k_max_reference,
            "diagnostic_gpu_to_cpu_sync_expected": self.diagnostic_gpu_to_cpu_sync_expected,
            "controller_wall_time_ms": self.controller_wall_time_ms,
        }


class GaussianCandidateDynamicBudgetObserverV1:
    """Observe q on-device; retain configuration only, never input tensors."""

    def __init__(self, config: DynamicBudgetObserveConfigV1) -> None:
        if not isinstance(config, DynamicBudgetObserveConfigV1):
            raise TypeError("config must be DynamicBudgetObserveConfigV1.")
        self.observe_histogram_bins = config.observe_histogram_bins
        self.k_max_reference = config.k_max_reference
        self.threshold_bin = config.threshold_bin
        self.k_min = config.k_min
        self.k_max = config.k_max

    @property
    def utility_threshold(self) -> float:
        return self.threshold_bin / self.observe_histogram_bins

    def observe(
        self, quality: torch.Tensor, quality_valid: torch.Tensor
    ) -> DynamicBudgetObserveResultV1:
        started_ns = time.perf_counter_ns()
        if not isinstance(quality, torch.Tensor) or quality.ndim != 1:
            raise ValueError("quality must be a Tensor with shape [N].")
        if not quality.dtype.is_floating_point:
            raise TypeError("quality must have floating-point dtype.")
        if (
            not isinstance(quality_valid, torch.Tensor)
            or quality_valid.dtype != torch.bool
            or tuple(quality_valid.shape) != tuple(quality.shape)
            or quality_valid.device != quality.device
        ):
            raise ValueError("quality_valid must be bool [N] on the quality device.")

        q = quality.detach()
        valid = (
            quality_valid.detach()
            & torch.isfinite(q)
            & (q >= HISTOGRAM_RANGE[0])
            & (q <= HISTOGRAM_RANGE[1])
        )
        valid_count = torch.count_nonzero(valid).to(torch.int64)
        invalid_count = torch.full(
            (), q.shape[0], dtype=torch.int64, device=q.device
        ) - valid_count
        # Explicit uniform-bin assignment makes every threshold edge exact:
        # floor(q * B), with q == 1 clamped into the final bin.  In particular,
        # q == threshold_bin / B enters threshold_bin rather than the lower bin.
        bin_indices = torch.floor(
            q[valid] * self.observe_histogram_bins
        ).to(torch.long)
        bin_indices = torch.clamp(
            bin_indices, min=0, max=self.observe_histogram_bins - 1
        )
        histogram = torch.bincount(
            bin_indices, minlength=self.observe_histogram_bins
        ).to(torch.int64)
        # The sole DtoH is fixed-size: valid, invalid, and B histogram counts.
        status = torch.cat((valid_count[None], invalid_count[None], histogram)).cpu()
        counts = tuple(int(status[index + 2]) for index in range(self.observe_histogram_bins))
        valid_host, invalid_host = int(status[0]), int(status[1])
        if sum(counts) != valid_host or valid_host + invalid_host != q.shape[0]:
            raise RuntimeError("Dynamic-budget q histogram conservation failed.")
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        return DynamicBudgetObserveResultV1(
            q_valid_count=valid_host,
            q_invalid_count=invalid_host,
            q_histogram_bins=self.observe_histogram_bins,
            q_histogram_counts=counts,
            q_histogram_range=HISTOGRAM_RANGE,
            k_max_reference=self.k_max_reference,
            diagnostic_gpu_to_cpu_sync_expected=q.is_cuda and q.numel() > 0,
            controller_wall_time_ms=elapsed_ms if math.isfinite(elapsed_ms) else None,
        )
