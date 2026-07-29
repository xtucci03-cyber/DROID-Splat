"""Default-off deterministic scheduling for historical mapping cameras.

This module is deliberately pure Python. It does not import NumPy or Torch and
does not inspect camera tensors, mapping losses, or Gaussian state.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional


LOG_PREFIX = "[HistoricalCameraScheduler]"
SCHEMA_VERSION = 1
SUPPORTED_MODES = frozenset(
    {
        "baseline_random",
        "deterministic_stratified",
    }
)
_TOP_LEVEL_FIELDS = frozenset(
    {
        "enabled",
        "mode",
        "history_budget",
        "preserve_all_history_until",
        "logging",
    }
)
_LOGGING_FIELDS = frozenset({"enabled", "sample_every"})


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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


def _camera_uid(camera: Any) -> int:
    if not hasattr(camera, "uid"):
        raise TypeError("Every scheduled camera must expose a uid attribute.")
    uid = camera.uid
    if isinstance(uid, bool):
        raise TypeError("Camera uid must be an integer, not bool.")
    try:
        normalized = operator.index(uid)
    except TypeError as error:
        raise TypeError(f"Camera uid must be integer-compatible, got {uid!r}.") from error
    if normalized < 0:
        raise ValueError(f"Camera uid must be non-negative, got {normalized}.")
    return normalized


def _uids(cameras: Sequence[Any]) -> list[int]:
    return [_camera_uid(camera) for camera in cameras]


def _require_unique_uids(cameras: Sequence[Any], field: str) -> list[int]:
    uids = _uids(cameras)
    if len(uids) != len(set(uids)):
        raise ValueError(f"{field} contains duplicate camera uids: {uids}.")
    return uids


def _same_objects_in_order(left: Sequence[Any], right: Sequence[Any]) -> bool:
    return len(left) == len(right) and all(
        left_item is right_item
        for left_item, right_item in zip(left, right)
    )


@dataclass(frozen=True)
class HistoricalCameraSchedulerResult:
    frames: list[Any]
    selected_history_cameras: list[Any]
    protected_recent_cameras: list[Any]
    protected_new_cameras: list[Any]
    eligible_old_history_cameras: list[Any]
    mapper_update_id: int
    mapping_iteration: int
    scheduler_call_count: int
    active_selection_tick: Optional[int]
    selection_active: bool
    effective_history_budget: int


@dataclass(frozen=True)
class _ValidatedConfig:
    enabled: bool
    mode: str
    history_budget: int
    preserve_all_history_until: int
    logging_enabled: bool
    logging_sample_every: int


def _validate_config(
    config: Optional[Mapping[str, Any]],
    *,
    n_last_frames: int,
    n_rand_frames: int,
) -> Optional[_ValidatedConfig]:
    if config is None:
        return None
    config = _require_mapping(config, "mapping.camera_scheduler")
    _reject_unknown_fields(
        config,
        _TOP_LEVEL_FIELDS,
        "mapping.camera_scheduler",
    )

    enabled = config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError(
            "mapping.camera_scheduler.enabled must be bool, "
            f"got {enabled!r}."
        )

    mode = config.get("mode", "baseline_random")
    if not isinstance(mode, str):
        raise TypeError(
            "mapping.camera_scheduler.mode must be str, "
            f"got {type(mode).__name__}."
        )
    if mode == "quality_fair":
        raise NotImplementedError(
            "mapping.camera_scheduler.mode='quality_fair' is not implemented "
            "in HCS-v0."
        )
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            "mapping.camera_scheduler.mode must be one of "
            f"{sorted(SUPPORTED_MODES)}, got {mode!r}."
        )

    history_budget = config.get("history_budget", 20)
    if not _is_int(history_budget):
        raise TypeError(
            "mapping.camera_scheduler.history_budget must be an integer "
            f"and bool is not accepted, got {history_budget!r}."
        )
    if history_budget < 0:
        raise ValueError(
            "mapping.camera_scheduler.history_budget must be non-negative, "
            f"got {history_budget}."
        )

    preserve_all = config.get("preserve_all_history_until", None)
    if preserve_all is None:
        preserve_all = n_last_frames + n_rand_frames
    elif not _is_int(preserve_all):
        raise TypeError(
            "mapping.camera_scheduler.preserve_all_history_until must be "
            f"null or a non-negative integer, got {preserve_all!r}."
        )
    if preserve_all < 0:
        raise ValueError(
            "mapping.camera_scheduler.preserve_all_history_until must be "
            f"non-negative, got {preserve_all}."
        )
    if preserve_all < n_last_frames:
        raise ValueError(
            "mapping.camera_scheduler.preserve_all_history_until must be at "
            f"least n_last_frames ({n_last_frames}), got {preserve_all}."
        )

    logging_config = config.get("logging", {})
    logging_config = _require_mapping(
        logging_config,
        "mapping.camera_scheduler.logging",
    )
    _reject_unknown_fields(
        logging_config,
        _LOGGING_FIELDS,
        "mapping.camera_scheduler.logging",
    )
    logging_enabled = logging_config.get("enabled", False)
    if not isinstance(logging_enabled, bool):
        raise TypeError(
            "mapping.camera_scheduler.logging.enabled must be bool, "
            f"got {logging_enabled!r}."
        )
    sample_every = logging_config.get("sample_every", 1)
    if not _is_int(sample_every) or sample_every <= 0:
        raise ValueError(
            "mapping.camera_scheduler.logging.sample_every must be a "
            f"positive integer and bool is not accepted, got {sample_every!r}."
        )

    if not enabled:
        return None

    if history_budget > n_rand_frames:
        raise ValueError(
            "mapping.camera_scheduler.history_budget must be in "
            f"0..n_rand_frames ({n_rand_frames}) when enabled, "
            f"got {history_budget}."
        )

    return _ValidatedConfig(
        enabled=enabled,
        mode=mode,
        history_budget=history_budget,
        preserve_all_history_until=preserve_all,
        logging_enabled=logging_enabled,
        logging_sample_every=sample_every,
    )


def build_historical_camera_scheduler(
    config: Optional[Mapping[str, Any]],
    *,
    n_last_frames: int,
    n_rand_frames: int,
) -> Optional["HistoricalCameraScheduler"]:
    """Validate configuration and return None for the exact disabled path."""
    if not _is_int(n_last_frames) or n_last_frames < 0:
        raise ValueError(
            f"n_last_frames must be a non-negative integer, got {n_last_frames!r}."
        )
    if not _is_int(n_rand_frames) or n_rand_frames < 0:
        raise ValueError(
            f"n_rand_frames must be a non-negative integer, got {n_rand_frames!r}."
        )

    validated = _validate_config(
        config,
        n_last_frames=n_last_frames,
        n_rand_frames=n_rand_frames,
    )
    if validated is None:
        return None
    return HistoricalCameraScheduler(
        config=validated,
        n_last_frames=n_last_frames,
        n_rand_frames=n_rand_frames,
    )


class HistoricalCameraScheduler:
    """HCS-v0 baseline-random and deterministic-stratified scheduler."""

    def __init__(
        self,
        *,
        config: _ValidatedConfig,
        n_last_frames: int,
        n_rand_frames: int,
    ) -> None:
        self.mode = config.mode
        self.history_budget = config.history_budget
        self.preserve_all_history_until = config.preserve_all_history_until
        self.logging_enabled = config.logging_enabled
        self.logging_sample_every = config.logging_sample_every
        self.n_last_frames = n_last_frames
        self.n_rand_frames = n_rand_frames

        self.scheduler_call_count = 0
        self.active_selection_tick = 0
        self._last_call_key: Optional[tuple[int, int]] = None

    def _validate_call_key(
        self,
        mapper_update_id: int,
        mapping_iteration: int,
    ) -> tuple[int, int]:
        if not _is_int(mapper_update_id) or mapper_update_id < 0:
            raise ValueError(
                "mapper_update_id must be a non-negative integer, "
                f"got {mapper_update_id!r}."
            )
        if not _is_int(mapping_iteration) or mapping_iteration < 0:
            raise ValueError(
                "mapping_iteration must be a non-negative integer, "
                f"got {mapping_iteration!r}."
            )
        call_key = (mapper_update_id, mapping_iteration)
        if self._last_call_key is not None and call_key <= self._last_call_key:
            raise RuntimeError(
                "HistoricalCameraScheduler call_key must be strictly "
                f"increasing; previous={self._last_call_key}, current={call_key}."
            )
        return call_key

    def select(
        self,
        *,
        historical_cameras: Sequence[Any],
        new_cameras: Sequence[Any],
        mapper_update_id: int,
        mapping_iteration: int,
        baseline_selector: Optional[Callable[[], Sequence[Any]]] = None,
    ) -> HistoricalCameraSchedulerResult:
        call_key = self._validate_call_key(
            mapper_update_id,
            mapping_iteration,
        )
        history = list(historical_cameras)
        new = list(new_cameras)
        history_uids = _require_unique_uids(history, "historical_cameras")
        new_uids = _require_unique_uids(new, "new_cameras")
        overlap = sorted(set(history_uids) & set(new_uids))
        if overlap:
            raise ValueError(
                "historical_cameras and new_cameras must be disjoint; "
                f"overlapping uids={overlap}."
            )

        if self.mode == "baseline_random":
            result_data = self._select_baseline_random(
                history=history,
                new=new,
                baseline_selector=baseline_selector,
            )
        elif self.mode == "deterministic_stratified":
            result_data = self._select_deterministic_stratified(
                history=history,
                new=new,
            )
        else:
            raise RuntimeError(f"Unexpected validated scheduler mode: {self.mode!r}.")

        (
            frames,
            selected_history,
            recent,
            eligible_old,
            selection_active,
            tick_used,
            effective_budget,
            strategy_details,
        ) = result_data
        self._validate_result(
            history=history,
            new=new,
            frames=frames,
            selected_history=selected_history,
            recent=recent,
            eligible_old=eligible_old,
            selection_active=selection_active,
            effective_budget=effective_budget,
        )

        call_count = self.scheduler_call_count
        result = HistoricalCameraSchedulerResult(
            frames=frames,
            selected_history_cameras=selected_history,
            protected_recent_cameras=recent,
            protected_new_cameras=new,
            eligible_old_history_cameras=eligible_old,
            mapper_update_id=call_key[0],
            mapping_iteration=call_key[1],
            scheduler_call_count=call_count,
            active_selection_tick=tick_used,
            selection_active=selection_active,
            effective_history_budget=effective_budget,
        )

        self._last_call_key = call_key
        self.scheduler_call_count += 1
        if selection_active:
            self.active_selection_tick += 1
        self._emit_event(
            result=result,
            history=history,
            new=new,
            strategy_details=strategy_details,
        )
        return result

    def _split_history(
        self,
        history: list[Any],
    ) -> tuple[list[Any], list[Any]]:
        if self.n_last_frames == 0:
            return [], history
        return history[-self.n_last_frames :], history[: -self.n_last_frames]

    def _select_baseline_random(
        self,
        *,
        history: list[Any],
        new: list[Any],
        baseline_selector: Optional[Callable[[], Sequence[Any]]],
    ) -> tuple[
        list[Any],
        list[Any],
        list[Any],
        list[Any],
        bool,
        Optional[int],
        int,
        dict[str, Any],
    ]:
        if baseline_selector is None or not callable(baseline_selector):
            raise TypeError(
                "baseline_random mode requires the original select_keyframes "
                "callable."
            )
        selected_history = list(baseline_selector())
        selected_history_uids = _require_unique_uids(
            selected_history,
            "baseline selected history",
        )
        history_uid_set = set(_uids(history))
        if not set(selected_history_uids).issubset(history_uid_set):
            raise ValueError(
                "The original baseline selector returned a camera outside "
                "historical_cameras."
            )

        baseline_threshold = self.n_last_frames + self.n_rand_frames
        if len(history) <= baseline_threshold:
            recent = history
            eligible_old: list[Any] = []
            selection_active = False
            effective_budget = 0
        else:
            recent, old = self._split_history(history)
            eligible_old = sorted(old, key=_camera_uid)
            effective_budget = min(self.n_rand_frames, len(eligible_old))
            selection_active = 0 < effective_budget < len(eligible_old)

        tick_used = self.active_selection_tick if selection_active else None
        frames = selected_history + new
        return (
            frames,
            selected_history,
            recent,
            eligible_old,
            selection_active,
            tick_used,
            effective_budget,
            {
                "strategy": "original_select_keyframes",
                "base_positions": None,
                "rotated_positions": None,
            },
        )

    def _select_deterministic_stratified(
        self,
        *,
        history: list[Any],
        new: list[Any],
    ) -> tuple[
        list[Any],
        list[Any],
        list[Any],
        list[Any],
        bool,
        Optional[int],
        int,
        dict[str, Any],
    ]:
        if len(history) <= self.preserve_all_history_until:
            selected_history = history
            recent = history
            eligible_old: list[Any] = []
            selection_active = False
            tick_used = None
            effective_budget = 0
            base_positions: Optional[list[int]] = None
            rotated_positions: Optional[list[int]] = None
        else:
            recent, old = self._split_history(history)
            eligible_old = sorted(old, key=_camera_uid)
            pool_size = len(eligible_old)
            effective_budget = min(self.history_budget, pool_size)
            selection_active = 0 < effective_budget < pool_size

            if effective_budget == 0:
                selected_old: list[Any] = []
                tick_used = None
                base_positions = None
                rotated_positions = None
            elif effective_budget == pool_size:
                selected_old = eligible_old
                tick_used = None
                base_positions = list(range(pool_size))
                rotated_positions = list(base_positions)
            else:
                tick_used = self.active_selection_tick
                base_positions = [
                    (j * pool_size) // effective_budget
                    for j in range(effective_budget)
                ]
                rotated_positions = [
                    (position + tick_used) % pool_size
                    for position in base_positions
                ]
                if len(set(rotated_positions)) != effective_budget:
                    raise RuntimeError(
                        "deterministic_stratified produced duplicate positions."
                    )
                selected_old = [
                    eligible_old[position]
                    for position in rotated_positions
                ]
                selected_old.sort(key=_camera_uid)

            selected_history = recent + selected_old

        frames = selected_history + new
        return (
            frames,
            selected_history,
            recent,
            eligible_old,
            selection_active,
            tick_used,
            effective_budget,
            {
                "strategy": "deterministic_stratified_v1",
                "base_positions": base_positions,
                "rotated_positions": rotated_positions,
            },
        )

    def _validate_result(
        self,
        *,
        history: list[Any],
        new: list[Any],
        frames: list[Any],
        selected_history: list[Any],
        recent: list[Any],
        eligible_old: list[Any],
        selection_active: bool,
        effective_budget: int,
    ) -> None:
        history_uids = _uids(history)
        new_uids = _uids(new)
        frame_uids = _require_unique_uids(frames, "scheduled frames")
        selected_history_uids = _require_unique_uids(
            selected_history,
            "selected history",
        )
        recent_uids = _require_unique_uids(recent, "protected recent history")
        eligible_uids = _require_unique_uids(
            eligible_old,
            "eligible old history",
        )

        if not _same_objects_in_order(frames, selected_history + new):
            raise RuntimeError(
                "Scheduled frame order must be selected history followed by "
                "all new cameras."
            )
        if not set(new_uids).issubset(frame_uids):
            raise RuntimeError("Every new camera must be protected.")
        if not set(recent_uids).issubset(selected_history_uids):
            raise RuntimeError("Every protected recent camera must be selected.")
        if not set(selected_history_uids).issubset(set(history_uids)):
            raise RuntimeError("Selected history must be a subset of history.")
        if set(eligible_uids) & set(recent_uids):
            raise RuntimeError(
                "Eligible old history and protected recent history must be disjoint."
            )

        if self.mode == "deterministic_stratified":
            if len(history) <= self.preserve_all_history_until:
                if not _same_objects_in_order(selected_history, history):
                    raise RuntimeError(
                        "Preserve-all selection must retain the original "
                        "historical camera order and identity."
                    )
            else:
                selected_old_uids = selected_history_uids[len(recent_uids) :]
                if not set(selected_old_uids).issubset(set(eligible_uids)):
                    raise RuntimeError(
                        "Budget-selected cameras must come from eligible old history."
                    )
                if len(selected_old_uids) != effective_budget:
                    raise RuntimeError(
                        "Selected old-history count does not match effective budget."
                    )
                if selected_old_uids != sorted(selected_old_uids):
                    raise RuntimeError(
                        "Selected old-history cameras must enter frames in uid order."
                    )
                expected_active = (
                    0 < effective_budget < len(eligible_old)
                )
                if selection_active != expected_active:
                    raise RuntimeError("selection_active is inconsistent.")
        elif len(history) <= self.n_last_frames + self.n_rand_frames:
            if not _same_objects_in_order(selected_history, history):
                raise RuntimeError(
                    "The original baseline preserve-all selection changed "
                    "historical camera order or identity."
                )
        else:
            selected_old_uids = selected_history_uids[len(recent_uids) :]
            expected_budget = min(self.n_rand_frames, len(eligible_old))
            if not _same_objects_in_order(
                selected_history[: len(recent)],
                recent,
            ):
                raise RuntimeError(
                    "The original baseline selector did not keep the recent "
                    "window first and unchanged."
                )
            if len(selected_old_uids) != expected_budget:
                raise RuntimeError(
                    "The original baseline selector returned an unexpected "
                    "old-history count."
                )
            if not set(selected_old_uids).issubset(set(eligible_uids)):
                raise RuntimeError(
                    "The original baseline selector returned an ineligible "
                    "old-history camera."
                )

    def _emit_event(
        self,
        *,
        result: HistoricalCameraSchedulerResult,
        history: list[Any],
        new: list[Any],
        strategy_details: dict[str, Any],
    ) -> None:
        if not self.logging_enabled:
            return
        if result.scheduler_call_count % self.logging_sample_every != 0:
            return

        recent_uids = _uids(result.protected_recent_cameras)
        eligible_uids = _uids(result.eligible_old_history_cameras)
        selected_history_uids = _uids(result.selected_history_cameras)
        selected_old_uids = [
            uid for uid in selected_history_uids
            if uid not in set(recent_uids)
        ]
        new_uids = _uids(new)
        frame_uids = _uids(result.frames)
        event = {
            "schema": SCHEMA_VERSION,
            "event_type": "selection",
            "event_id": f"hcs:{result.scheduler_call_count}",
            "process_role": "mapper",
            "mode": self.mode,
            "status": "ok",
            "reason": (
                "partial_budget"
                if result.selection_active
                else "preserve_all_or_full_budget"
            ),
            "mapper_update_id": result.mapper_update_id,
            "mapping_iteration": result.mapping_iteration,
            "call_key": [
                result.mapper_update_id,
                result.mapping_iteration,
            ],
            "scheduler_call_count": result.scheduler_call_count,
            "scheduler_call_count_after": self.scheduler_call_count,
            "active_selection_tick": result.active_selection_tick,
            "active_selection_count_after": self.active_selection_tick,
            "selection_active": result.selection_active,
            "history_count": len(history),
            "new_camera_count": len(new),
            "n_last_frames": self.n_last_frames,
            "n_rand_frames": self.n_rand_frames,
            "history_budget": self.history_budget,
            "effective_history_budget": result.effective_history_budget,
            "preserve_all_history_until": self.preserve_all_history_until,
            "eligible_old_history_count": len(eligible_uids),
            "eligible_old_history_uids": eligible_uids,
            "protected_recent_count": len(recent_uids),
            "protected_recent_uids": recent_uids,
            "selected_old_history_count": len(selected_old_uids),
            "selected_old_history_uids": selected_old_uids,
            "selected_history_count": len(selected_history_uids),
            "selected_history_uids": selected_history_uids,
            "protected_new_count": len(new_uids),
            "protected_new_uids": new_uids,
            "final_selected_count": len(frame_uids),
            "final_selected_uids": frame_uids,
            "logging_sample_every": self.logging_sample_every,
            **strategy_details,
        }
        print(
            LOG_PREFIX
            + " "
            + json.dumps(
                event,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
