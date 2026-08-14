import json
import multiprocessing
import os
import signal
import time
import unittest
from unittest import mock

from src.runtime_process_diagnostics_v1 import (
    ChildProcessFailure,
    MAX_JSONL_BYTES,
    PROCESS_ROLE_SPECS,
    ProcessRecord,
    ProcessRoleSpec,
    RuntimeProcessDiagnosticsV1,
    cleanup_processes_bounded,
    find_child_process_failure,
    get_queue_item_or_child_failure,
    run_process_target_with_diagnostics,
    wait_for_condition_or_child_failure,
)


def _normal_child():
    return None


def _exit_child(exitcode):
    os._exit(exitcode)


def _abort_child():
    os.abort()


def _raise_python_exception():
    raise ValueError("diagnostic-test-error")


class _UnprintableError(RuntimeError):
    def __str__(self):
        raise RuntimeError("stringification-failed")


def _raise_unprintable_exception():
    raise _UnprintableError()


def _raise_keyboard_interrupt():
    raise KeyboardInterrupt


def _sleep_child(seconds):
    time.sleep(seconds)


def _set_shared_value_after_delay(shared_value, delay_s):
    time.sleep(delay_s)
    shared_value.value = 1


def _raise_after_opening_stage(diagnostics):
    diagnostics.emit_stage(
        rank=2,
        role="backend",
        stage="backend_ba",
        phase="begin",
        update_id=7,
    )
    raise ValueError("stage-error")


class _StubbornFakeProcess:
    def __init__(self):
        self.exitcode = None
        self.terminate_called = False
        self.kill_called = False
        self.join_timeouts = []
        self._alive = True

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminate_called = True

    def kill(self):
        self.kill_called = True
        self._alive = False
        self.exitcode = -9

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)


class RuntimeProcessDiagnosticsV1Tests(unittest.TestCase):
    def _diagnostics(self, events, **overrides):
        values = {
            "enabled": True,
            "stage_logging": True,
            "fail_fast_on_child_error": True,
            "cleanup_timeout_s": 0.2,
            "event_sink": events.append,
        }
        values.update(overrides)
        return RuntimeProcessDiagnosticsV1(**values)

    def _start_record(self, target, args=(), *, rank=1, role="frontend_tracking"):
        context = multiprocessing.get_context("spawn")
        process = context.Process(target=target, args=args)
        process.start()
        spec = ProcessRoleSpec(rank, role, role)
        return ProcessRecord.from_started_process(process, spec)

    def test_diagnostics_disabled_emits_no_event(self):
        events = []
        diagnostics = RuntimeProcessDiagnosticsV1(enabled=False, event_sink=events.append)

        diagnostics.emit("process_started", rank=0, role="opencv_stream")
        diagnostics.emit_stage(
            rank=1,
            role="frontend_tracking",
            stage="frontend_update",
            phase="begin",
        )

        self.assertEqual(events, [])

        stage_disabled = self._diagnostics(events, stage_logging=False)
        stage_disabled.emit_stage(
            rank=1,
            role="frontend_tracking",
            stage="frontend_update",
            phase="begin",
        )
        self.assertEqual(events, [])

    def test_config_contract_is_default_off(self):
        diagnostics = RuntimeProcessDiagnosticsV1.from_config(None)

        self.assertFalse(diagnostics.enabled)
        self.assertTrue(diagnostics.stage_logging)
        self.assertTrue(diagnostics.fail_fast_on_child_error)
        with self.assertRaisesRegex(ValueError, "Unknown runtime_process_diagnostics_v1"):
            RuntimeProcessDiagnosticsV1.from_config({"unexpected": True})

    def test_normal_target_records_enter_and_normal_exit(self):
        events = []
        diagnostics = self._diagnostics(events)
        spec = ProcessRoleSpec(0, "opencv_stream", "OpenCV Stream")

        run_process_target_with_diagnostics(diagnostics, spec, _normal_child, ())

        self.assertEqual([event["event"] for event in events], [
            "process_target_enter",
            "process_target_exit",
        ])
        self.assertEqual(events[-1]["exit_kind"], "normal")
        self.assertEqual(events[-1]["exitcode"], 0)
        self.assertEqual(events[-1]["rank"], 0)
        self.assertEqual(events[-1]["role"], "opencv_stream")

    def test_python_exception_records_role_traceback_and_propagates(self):
        events = []
        diagnostics = self._diagnostics(events)
        spec = ProcessRoleSpec(2, "backend", "Backend")

        with self.assertRaisesRegex(ValueError, "diagnostic-test-error"):
            run_process_target_with_diagnostics(
                diagnostics,
                spec,
                _raise_python_exception,
                (),
            )

        exit_event = events[-1]
        self.assertEqual(exit_event["event"], "process_target_exit")
        self.assertEqual(exit_event["exit_kind"], "python_exception")
        self.assertEqual(exit_event["rank"], 2)
        self.assertEqual(exit_event["role"], "backend")
        self.assertEqual(exit_event["exception_type"], "ValueError")
        self.assertIn("diagnostic-test-error", exit_event["exception_message"])
        self.assertIn("_raise_python_exception", exit_event["traceback"])

    def test_python_exception_closes_open_stage_as_error_and_preserves_exception(self):
        events = []
        diagnostics = self._diagnostics(events)
        spec = ProcessRoleSpec(2, "backend", "Backend")

        with self.assertRaisesRegex(ValueError, "stage-error"):
            run_process_target_with_diagnostics(
                diagnostics,
                spec,
                _raise_after_opening_stage,
                (diagnostics,),
            )

        stage_events = [event for event in events if event["event"] == "process_stage"]
        self.assertEqual(
            [(event["phase"], event["status"]) for event in stage_events],
            [("begin", "running"), ("end", "error")],
        )
        self.assertEqual(events[-1]["exit_kind"], "python_exception")

    def test_nonzero_exitcode_maps_to_expected_role(self):
        record = self._start_record(_exit_child, (7,), rank=3, role="loop_detector")
        record.process.join(timeout=5.0)

        failure = find_child_process_failure([record])

        self.assertIsNotNone(failure)
        self.assertEqual(failure.exitcode, 7)
        self.assertEqual(failure.rank, 3)
        self.assertEqual(failure.role, "loop_detector")

    def test_abort_or_equivalent_native_exit_is_detected(self):
        record = self._start_record(
            _abort_child,
            rank=5,
            role="gaussian_mapping",
        )
        record.process.join(timeout=5.0)

        failure = find_child_process_failure([record])

        self.assertIsNotNone(failure)
        self.assertNotEqual(failure.exitcode, 0)
        self.assertEqual(failure.role, "gaussian_mapping")
        if os.name == "posix":
            self.assertEqual(failure.exitcode, -signal.SIGABRT)
            self.assertEqual(failure.signal_number, signal.SIGABRT)
            self.assertEqual(failure.signal_name, "SIGABRT")

    def test_disabled_process_normal_early_exit_is_not_failure(self):
        record = self._start_record(
            _normal_child,
            rank=4,
            role="visualizing",
        )
        record.process.join(timeout=5.0)

        self.assertEqual(record.process.exitcode, 0)
        self.assertIsNone(find_child_process_failure([record]))

    def test_wait_fails_fast_instead_of_waiting_forever(self):
        events = []
        diagnostics = self._diagnostics(events)
        record = self._start_record(_exit_child, (9,), rank=2, role="backend")
        start = time.monotonic()

        with self.assertRaises(ChildProcessFailure) as raised:
            wait_for_condition_or_child_failure(
                lambda: False,
                [record],
                diagnostics,
                poll_interval_s=0.01,
            )

        self.assertLess(time.monotonic() - start, 5.0)
        self.assertEqual(raised.exception.failure.role, "backend")
        self.assertTrue(any(event["event"] == "child_process_failed" for event in events))
        record.process.join(timeout=5.0)
        self.assertFalse(record.process.is_alive())

    def test_wait_accepts_normal_exit_after_shared_completion_update(self):
        events = []
        diagnostics = self._diagnostics(events)
        context = multiprocessing.get_context("spawn")
        completed = context.Value("i", 0)
        record = self._start_record(
            _set_shared_value_after_delay,
            (completed, 0.05),
            rank=4,
            role="visualizing",
        )

        wait_for_condition_or_child_failure(
            lambda: bool(completed.value),
            [record],
            diagnostics,
            poll_interval_s=0.01,
        )
        record.process.join(timeout=5.0)

        self.assertEqual(record.process.exitcode, 0)
        self.assertFalse(any(event["event"] == "child_process_failed" for event in events))

    def test_queue_block_fails_fast_when_child_exits(self):
        events = []
        diagnostics = self._diagnostics(events)
        context = multiprocessing.get_context("spawn")
        process_queue = context.Queue()
        record = self._start_record(_exit_child, (11,), rank=5, role="gaussian_mapping")
        start = time.monotonic()

        with self.assertRaises(ChildProcessFailure) as raised:
            get_queue_item_or_child_failure(
                process_queue,
                [record],
                diagnostics,
                poll_interval_s=0.01,
            )

        self.assertLess(time.monotonic() - start, 5.0)
        self.assertEqual(raised.exception.failure.role, "gaussian_mapping")
        record.process.join(timeout=5.0)
        self.assertFalse(record.process.is_alive())

    def test_cleanup_is_bounded_and_uses_terminate_then_kill(self):
        events = []
        diagnostics = self._diagnostics(events)
        process = _StubbornFakeProcess()
        record = ProcessRecord(
            process=process,
            rank=6,
            role="mapping_gui",
            process_name="Mapping GUI",
            pid=123,
        )
        start = time.monotonic()

        remaining = cleanup_processes_bounded(
            [record],
            diagnostics,
            timeout_s=0.01,
        )

        self.assertLess(time.monotonic() - start, 1.0)
        self.assertTrue(process.terminate_called)
        self.assertTrue(process.kill_called)
        self.assertEqual(remaining, [])
        self.assertTrue(all(timeout is not None for timeout in process.join_timeouts))

    def test_cleanup_leaves_no_live_spawned_child(self):
        events = []
        diagnostics = self._diagnostics(events, cleanup_timeout_s=1.0)
        record = self._start_record(
            _sleep_child,
            (30.0,),
            rank=3,
            role="loop_detector",
        )
        start = time.monotonic()

        remaining = cleanup_processes_bounded([record], diagnostics)

        self.assertLess(time.monotonic() - start, 1.5)
        self.assertEqual(remaining, [])
        self.assertFalse(record.process.is_alive())
        self.assertIsNotNone(record.process.exitcode)

    def test_keyboard_interrupt_is_not_classified_as_child_failure(self):
        events = []
        diagnostics = self._diagnostics(events)
        interrupt_event = multiprocessing.Event()
        spec = ProcessRoleSpec(1, "frontend_tracking", "Frontend Tracking")
        diagnostics.emit_stage(
            rank=1,
            role="frontend_tracking",
            stage="frontend_operator",
            phase="begin",
        )

        with self.assertRaises(KeyboardInterrupt):
            run_process_target_with_diagnostics(
                diagnostics,
                spec,
                _raise_keyboard_interrupt,
                (),
                interrupt_event,
            )
        self.assertTrue(interrupt_event.is_set())

        with self.assertRaises(KeyboardInterrupt):
            wait_for_condition_or_child_failure(
                lambda: False,
                [],
                diagnostics,
                poll_interval_s=0.01,
                interrupt_event=interrupt_event,
            )

        self.assertFalse(any(event["event"] == "child_process_failed" for event in events))
        self.assertEqual(events[-1]["exit_kind"], "keyboard_interrupt")
        stage_end = [
            event
            for event in events
            if event["event"] == "process_stage" and event["phase"] == "end"
        ]
        self.assertEqual(len(stage_end), 1)
        self.assertEqual(stage_end[0]["status"], "interrupted")

    def test_jsonl_is_one_atomic_bounded_valid_write(self):
        event = RuntimeProcessDiagnosticsV1._base_event("process_target_exit")
        event.update(
            exception_message="\U0001f4a5" * 10_000,
            traceback="\u8ffd\u8e2a" * 10_000,
            non_serializable=object(),
        )

        payload = RuntimeProcessDiagnosticsV1._encode_json_line(event)

        self.assertLessEqual(len(payload), MAX_JSONL_BYTES)
        self.assertEqual(payload.count(b"\n"), 1)
        self.assertTrue(payload.endswith(b"\n"))
        json.loads(payload.decode("ascii"))
        with mock.patch("src.runtime_process_diagnostics_v1.os.write") as write:
            RuntimeProcessDiagnosticsV1._write_json_line(event)
        write.assert_called_once()
        self.assertLessEqual(len(write.call_args.args[1]), MAX_JSONL_BYTES)

    def test_logging_failure_does_not_replace_target_exception(self):
        def failing_sink(_event):
            raise RuntimeError("logging-failed")

        diagnostics = RuntimeProcessDiagnosticsV1(
            enabled=True,
            event_sink=failing_sink,
        )
        spec = ProcessRoleSpec(2, "backend", "Backend")

        with self.assertRaisesRegex(ValueError, "diagnostic-test-error"):
            run_process_target_with_diagnostics(
                diagnostics,
                spec,
                _raise_python_exception,
                (),
            )

    def test_unprintable_exception_is_rethrown_unchanged(self):
        events = []
        diagnostics = self._diagnostics(events)
        spec = ProcessRoleSpec(2, "backend", "Backend")

        with self.assertRaises(_UnprintableError):
            run_process_target_with_diagnostics(
                diagnostics,
                spec,
                _raise_unprintable_exception,
                (),
            )

        self.assertEqual(events[-1]["exception_type"], "_UnprintableError")
        self.assertEqual(
            events[-1]["exception_message"],
            "<unprintable exception message>",
        )

    def test_spawn_context_can_pickle_module_level_wrapper_and_target(self):
        context = multiprocessing.get_context("spawn")
        diagnostics = RuntimeProcessDiagnosticsV1(enabled=False)
        spec = ProcessRoleSpec(4, "visualizing", "Visualizing")
        process = context.Process(
            target=run_process_target_with_diagnostics,
            args=(diagnostics, spec, _normal_child, ()),
        )

        process.start()
        process.join(timeout=5.0)

        self.assertFalse(process.is_alive())
        self.assertEqual(process.exitcode, 0)

    def test_structured_event_has_complete_identity_fields(self):
        events = []
        diagnostics = self._diagnostics(events)

        diagnostics.emit(
            "process_started",
            pid=321,
            ppid=123,
            rank=1,
            role="frontend_tracking",
            process_name="Frontend Tracking",
        )

        event = events[0]
        for key in (
            "schema",
            "event",
            "pid",
            "ppid",
            "rank",
            "role",
            "process_name",
            "monotonic_ns",
        ):
            self.assertIn(key, event)
        self.assertEqual(event["event"], "process_started")
        self.assertEqual(event["pid"], 321)

    def test_fixed_process_role_contract_covers_all_seven_roles(self):
        self.assertEqual(
            [(spec.rank, spec.role, spec.process_name) for spec in PROCESS_ROLE_SPECS],
            [
                (0, "opencv_stream", "OpenCV Stream"),
                (1, "frontend_tracking", "Frontend Tracking"),
                (2, "backend", "Backend"),
                (3, "loop_detector", "Loop Detector"),
                (4, "visualizing", "Visualizing"),
                (5, "gaussian_mapping", "Gaussian Mapping"),
                (6, "mapping_gui", "Mapping GUI"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
