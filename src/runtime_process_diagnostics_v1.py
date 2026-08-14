"""Default-off process diagnostics for native-failure investigations.

This module deliberately has no Torch or project imports so the process
lifecycle and failure-monitoring contract can be tested on CPU-only hosts.
"""

from __future__ import annotations

import json
import multiprocessing.connection
import os
import queue as queue_module
import signal
import sys
import time
import traceback as traceback_module
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


SCHEMA_VERSION = 1
MAX_TRACEBACK_CHARS = 2_048
MAX_EXCEPTION_MESSAGE_CHARS = 512
MAX_JSONL_BYTES = 4_096
DEFAULT_POLL_INTERVAL_S = 0.05
DEFAULT_CLEANUP_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class ProcessRoleSpec:
    rank: int
    role: str
    process_name: str


PROCESS_ROLE_SPECS = (
    ProcessRoleSpec(0, "opencv_stream", "OpenCV Stream"),
    ProcessRoleSpec(1, "frontend_tracking", "Frontend Tracking"),
    ProcessRoleSpec(2, "backend", "Backend"),
    ProcessRoleSpec(3, "loop_detector", "Loop Detector"),
    ProcessRoleSpec(4, "visualizing", "Visualizing"),
    ProcessRoleSpec(5, "gaussian_mapping", "Gaussian Mapping"),
    ProcessRoleSpec(6, "mapping_gui", "Mapping GUI"),
)


@dataclass
class ProcessRecord:
    process: Any
    rank: int
    role: str
    process_name: str
    pid: Optional[int] = None
    sentinel: Any = None

    @classmethod
    def from_started_process(
        cls,
        process: Any,
        spec: ProcessRoleSpec,
    ) -> "ProcessRecord":
        return cls(
            process=process,
            rank=spec.rank,
            role=spec.role,
            process_name=spec.process_name,
            pid=process.pid,
            sentinel=process.sentinel,
        )


@dataclass(frozen=True)
class ChildFailureInfo:
    pid: Optional[int]
    rank: int
    role: str
    process_name: str
    exitcode: int
    signal_number: Optional[int]
    signal_name: Optional[str]


class ChildProcessFailure(RuntimeError):
    def __init__(self, failure: ChildFailureInfo):
        self.failure = failure
        super().__init__(
            "Child process failed: "
            f"role={failure.role}, rank={failure.rank}, pid={failure.pid}, "
            f"exitcode={failure.exitcode}, signal={failure.signal_name}"
        )


class RuntimeProcessDiagnosticsV1:
    """Small, best-effort JSONL emitter with a strict default-off path."""

    _ALLOWED_CONFIG_KEYS = {
        "enabled",
        "stage_logging",
        "fail_fast_on_child_error",
        "cleanup_timeout_s",
    }

    def __init__(
        self,
        *,
        enabled: bool = False,
        stage_logging: bool = True,
        fail_fast_on_child_error: bool = True,
        cleanup_timeout_s: float = DEFAULT_CLEANUP_TIMEOUT_S,
        event_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.stage_logging = bool(stage_logging)
        self.fail_fast_on_child_error = bool(fail_fast_on_child_error)
        self.cleanup_timeout_s = float(cleanup_timeout_s)
        if self.cleanup_timeout_s <= 0:
            raise ValueError("cleanup_timeout_s must be greater than zero")
        self._event_sink = event_sink
        self._active_stages: list[dict[str, Any]] = []

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "RuntimeProcessDiagnosticsV1":
        if config is None:
            return cls()

        unknown = set(config.keys()) - cls._ALLOWED_CONFIG_KEYS
        if unknown:
            raise ValueError(
                "Unknown runtime_process_diagnostics_v1 fields: "
                + ", ".join(sorted(str(key) for key in unknown))
            )

        for key in ("enabled", "stage_logging", "fail_fast_on_child_error"):
            value = config.get(key, getattr(cls(), key))
            if not isinstance(value, bool):
                raise TypeError(f"runtime_process_diagnostics_v1.{key} must be a boolean")

        cleanup_timeout_s = config.get("cleanup_timeout_s", DEFAULT_CLEANUP_TIMEOUT_S)
        if isinstance(cleanup_timeout_s, bool) or not isinstance(cleanup_timeout_s, (int, float)):
            raise TypeError("runtime_process_diagnostics_v1.cleanup_timeout_s must be numeric")

        return cls(
            enabled=config.get("enabled", False),
            stage_logging=config.get("stage_logging", True),
            fail_fast_on_child_error=config.get("fail_fast_on_child_error", True),
            cleanup_timeout_s=cleanup_timeout_s,
        )

    @staticmethod
    def _base_event(event: str) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "event": event,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "monotonic_ns": time.monotonic_ns(),
        }

    @staticmethod
    def _json_default(value: Any) -> str:
        try:
            rendered = str(value)[:128]
        except Exception:
            rendered = "unprintable"
        return f"<{type(value).__name__}:{rendered}>"

    @classmethod
    def _encode_json_line(cls, event: Mapping[str, Any]) -> bytes:
        normalized = dict(event)

        def encode(payload: Mapping[str, Any]) -> bytes:
            line = json.dumps(
                payload,
                default=cls._json_default,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ) + "\n"
            return line.encode("ascii")

        encoded = encode(normalized)
        if len(encoded) <= MAX_JSONL_BYTES:
            return encoded

        for key, limit in (
            ("traceback", 1_024),
            ("exception_message", 256),
        ):
            value = normalized.get(key)
            if isinstance(value, str):
                normalized[key] = value[:limit] + "<truncated>"
        normalized["jsonl_truncated"] = True
        encoded = encode(normalized)
        if len(encoded) <= MAX_JSONL_BYTES:
            return encoded

        core_keys = (
            "schema",
            "event",
            "pid",
            "ppid",
            "monotonic_ns",
            "rank",
            "role",
            "process_name",
            "exitcode",
            "signal",
            "signal_name",
            "stage",
            "phase",
            "status",
        )
        minimal = {
            key: normalized[key]
            for key in core_keys
            if key in normalized
        }
        minimal["jsonl_truncated"] = True
        encoded = encode(minimal)
        if len(encoded) <= MAX_JSONL_BYTES:
            return encoded

        # Core fields are short in the production contract. This final guard
        # preserves a valid single JSON line even for adversarial values.
        fallback = {
            "schema": SCHEMA_VERSION,
            "event": str(normalized.get("event", "diagnostic_event"))[:64],
            "pid": normalized.get("pid", os.getpid()),
            "ppid": normalized.get("ppid", os.getppid()),
            "monotonic_ns": normalized.get("monotonic_ns", time.monotonic_ns()),
            "jsonl_truncated": True,
        }
        encoded = encode(fallback)
        if len(encoded) <= MAX_JSONL_BYTES:
            return encoded
        return b'{"event":"diagnostic_event","jsonl_truncated":true,"schema":1}\n'

    @classmethod
    def _write_json_line(cls, event: Mapping[str, Any]) -> None:
        payload = cls._encode_json_line(event)

        try:
            os.write(sys.stdout.fileno(), payload)
        except Exception:
            # Preserve event atomicity: diagnostics are best-effort, so a
            # failed single write is dropped rather than retried in chunks.
            pass

    def emit(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return

        payload = self._base_event(event)
        payload.update(fields)
        try:
            if self._event_sink is not None:
                self._event_sink(payload)
            else:
                self._write_json_line(payload)
        except Exception:
            # Diagnostics must never become a new SLAM failure mode.
            pass

    def emit_stage(
        self,
        *,
        rank: int,
        role: str,
        stage: str,
        phase: str,
        **fields: Any,
    ) -> None:
        if not self.enabled or not self.stage_logging:
            return
        status = fields.pop(
            "status",
            "running" if phase == "begin" else "ok",
        )
        stage_identity = {
            "rank": int(rank),
            "role": str(role),
            "stage": str(stage),
        }
        if phase == "begin":
            self._active_stages.append(stage_identity)
        elif phase == "end":
            for index in range(len(self._active_stages) - 1, -1, -1):
                if self._active_stages[index] == stage_identity:
                    del self._active_stages[index]
                    break
        self.emit(
            "process_stage",
            **stage_identity,
            phase=str(phase),
            status=status,
            **fields,
        )

    def close_active_stages(self, *, status: str, **fields: Any) -> None:
        if not self.enabled or not self.stage_logging:
            return
        while self._active_stages:
            stage = self._active_stages.pop()
            self.emit(
                "process_stage",
                **stage,
                phase="end",
                status=status,
                **fields,
            )


def run_process_target_with_diagnostics(
    diagnostics: RuntimeProcessDiagnosticsV1,
    spec: ProcessRoleSpec,
    target: Callable[..., Any],
    args: Sequence[Any],
    interrupt_event: Any = None,
) -> None:
    diagnostics.emit(
        "process_target_enter",
        rank=spec.rank,
        role=spec.role,
        process_name=spec.process_name,
    )
    try:
        target(*args)
    except KeyboardInterrupt:
        if interrupt_event is not None:
            try:
                interrupt_event.set()
            except Exception:
                pass
        diagnostics.close_active_stages(status="interrupted")
        diagnostics.emit(
            "process_target_exit",
            rank=spec.rank,
            role=spec.role,
            process_name=spec.process_name,
            exit_kind="keyboard_interrupt",
        )
        raise
    except BaseException as error:
        try:
            exception_message = str(error)[:MAX_EXCEPTION_MESSAGE_CHARS]
        except Exception:
            exception_message = "<unprintable exception message>"
        try:
            formatted_traceback = "".join(
                traceback_module.format_exception(type(error), error, error.__traceback__)
            )[-MAX_TRACEBACK_CHARS:]
        except Exception:
            formatted_traceback = "<unprintable traceback>"
        diagnostics.close_active_stages(
            status="error",
            exception_type=type(error).__name__,
        )
        diagnostics.emit(
            "process_target_exit",
            rank=spec.rank,
            role=spec.role,
            process_name=spec.process_name,
            exit_kind="python_exception",
            exception_type=type(error).__name__,
            exception_message=exception_message,
            traceback=formatted_traceback,
        )
        raise
    else:
        diagnostics.emit(
            "process_target_exit",
            rank=spec.rank,
            role=spec.role,
            process_name=spec.process_name,
            exit_kind="normal",
            exitcode=0,
        )


def _signal_fields(exitcode: int) -> tuple[Optional[int], Optional[str]]:
    if exitcode >= 0:
        return None, None
    signal_number = -exitcode
    try:
        signal_name = signal.Signals(signal_number).name
    except (ValueError, OSError):
        signal_name = None
    return signal_number, signal_name


def find_child_process_failure(records: Iterable[ProcessRecord]) -> Optional[ChildFailureInfo]:
    for record in records:
        process = record.process
        exitcode = process.exitcode
        is_alive = process.is_alive()
        if exitcode is None or is_alive or exitcode == 0:
            continue
        signal_number, signal_name = _signal_fields(int(exitcode))
        return ChildFailureInfo(
            pid=record.pid,
            rank=record.rank,
            role=record.role,
            process_name=record.process_name,
            exitcode=int(exitcode),
            signal_number=signal_number,
            signal_name=signal_name,
        )
    return None


def raise_if_child_process_failed(
    records: Iterable[ProcessRecord],
    diagnostics: RuntimeProcessDiagnosticsV1,
) -> None:
    failure = find_child_process_failure(records)
    if failure is None:
        return
    diagnostics.emit(
        "child_process_failed",
        pid=failure.pid,
        ppid=os.getpid(),
        rank=failure.rank,
        role=failure.role,
        process_name=failure.process_name,
        exitcode=failure.exitcode,
        signal=failure.signal_number,
        signal_name=failure.signal_name,
    )
    raise ChildProcessFailure(failure)


def wait_for_condition_or_child_failure(
    predicate: Callable[[], bool],
    records: Sequence[ProcessRecord],
    diagnostics: RuntimeProcessDiagnosticsV1,
    *,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    interrupt_event: Any = None,
) -> None:
    while not predicate():
        if interrupt_event is not None and interrupt_event.is_set():
            raise KeyboardInterrupt
        raise_if_child_process_failed(records, diagnostics)
        sentinels = [
            record.sentinel
            for record in records
            if record.sentinel is not None and record.process.exitcode is None
        ]
        if sentinels:
            multiprocessing.connection.wait(sentinels, timeout=poll_interval_s)
        else:
            time.sleep(poll_interval_s)
    if interrupt_event is not None and interrupt_event.is_set():
        raise KeyboardInterrupt
    raise_if_child_process_failed(records, diagnostics)


def get_queue_item_or_child_failure(
    process_queue: Any,
    records: Sequence[ProcessRecord],
    diagnostics: RuntimeProcessDiagnosticsV1,
    *,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    interrupt_event: Any = None,
) -> Any:
    while True:
        if interrupt_event is not None and interrupt_event.is_set():
            raise KeyboardInterrupt
        raise_if_child_process_failed(records, diagnostics)
        try:
            return process_queue.get(timeout=poll_interval_s)
        except queue_module.Empty:
            continue


def cleanup_processes_bounded(
    records: Sequence[ProcessRecord],
    diagnostics: RuntimeProcessDiagnosticsV1,
    *,
    timeout_s: Optional[float] = None,
) -> list[ProcessRecord]:
    timeout_s = diagnostics.cleanup_timeout_s if timeout_s is None else float(timeout_s)
    timeout_s = max(timeout_s, 0.0)
    start = time.monotonic()
    deadline = start + timeout_s
    kill_grace_s = min(0.5, timeout_s / 2.0)
    terminate_deadline = deadline - kill_grace_s
    diagnostics.emit("process_cleanup", phase="begin", timeout_s=timeout_s)

    for record in records:
        try:
            if record.process.is_alive():
                record.process.terminate()
        except Exception:
            pass

    for record in records:
        remaining = max(0.0, terminate_deadline - time.monotonic())
        try:
            record.process.join(timeout=remaining)
        except Exception:
            pass

    still_alive = []
    for record in records:
        try:
            if record.process.is_alive():
                still_alive.append(record)
                kill = getattr(record.process, "kill", None)
                if callable(kill):
                    kill()
        except Exception:
            still_alive.append(record)

    for record in still_alive:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            record.process.join(timeout=remaining)
        except Exception:
            pass

    remaining_alive = []
    for record in records:
        try:
            if record.process.is_alive():
                remaining_alive.append(record)
        except Exception:
            remaining_alive.append(record)

    diagnostics.emit(
        "process_cleanup",
        phase="end",
        remaining_alive=[record.role for record in remaining_alive],
    )
    return remaining_alive
