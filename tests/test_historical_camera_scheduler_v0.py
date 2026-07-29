from __future__ import annotations

import ast
import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

from src.camera_scheduling.historical_camera_scheduler import (
    HistoricalCameraScheduler,
    build_historical_camera_scheduler,
)
from tools.analysis import build_historical_camera_scheduler_summary as parser


@dataclass(frozen=True)
class FakeCamera:
    uid: int


def cameras(count: int, *, start: int = 0) -> list[FakeCamera]:
    return [FakeCamera(uid) for uid in range(start, start + count)]


def config(
    *,
    enabled: bool = True,
    mode: str = "deterministic_stratified",
    history_budget: int = 10,
    preserve_all_history_until: int | None = None,
    logging_enabled: bool = False,
    sample_every: int = 1,
) -> dict:
    return {
        "enabled": enabled,
        "mode": mode,
        "history_budget": history_budget,
        "preserve_all_history_until": preserve_all_history_until,
        "logging": {
            "enabled": logging_enabled,
            "sample_every": sample_every,
        },
    }


def scheduler(**kwargs) -> HistoricalCameraScheduler:
    built = build_historical_camera_scheduler(
        config(**kwargs),
        n_last_frames=10,
        n_rand_frames=20,
    )
    assert built is not None
    return built


class ConfigGateTests(unittest.TestCase):
    def test_missing_config_is_disabled(self) -> None:
        self.assertIsNone(
            build_historical_camera_scheduler(
                None,
                n_last_frames=10,
                n_rand_frames=20,
            )
        )

    def test_disabled_default_does_not_construct_state(self) -> None:
        disabled = build_historical_camera_scheduler(
            config(
                enabled=False,
                mode="baseline_random",
                history_budget=20,
            ),
            n_last_frames=10,
            n_rand_frames=15,
        )
        self.assertIsNone(disabled)

    def test_enabled_budget_must_not_exceed_random_baseline(self) -> None:
        with self.assertRaisesRegex(ValueError, "0..n_rand_frames"):
            build_historical_camera_scheduler(
                config(history_budget=20),
                n_last_frames=10,
                n_rand_frames=15,
            )

    def test_bool_is_not_an_integer_budget(self) -> None:
        with self.assertRaisesRegex(TypeError, "bool is not accepted"):
            build_historical_camera_scheduler(
                config(history_budget=True),
                n_last_frames=10,
                n_rand_frames=20,
            )

    def test_unknown_top_level_field_fails(self) -> None:
        value = config()
        value["typo"] = 1
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            build_historical_camera_scheduler(
                value,
                n_last_frames=10,
                n_rand_frames=20,
            )

    def test_unknown_logging_field_fails(self) -> None:
        value = config()
        value["logging"]["typo"] = 1
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            build_historical_camera_scheduler(
                value,
                n_last_frames=10,
                n_rand_frames=20,
            )

    def test_quality_fair_fails_closed(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "not implemented"):
            build_historical_camera_scheduler(
                config(mode="quality_fair"),
                n_last_frames=10,
                n_rand_frames=20,
            )

    def test_unknown_mode_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be one of"):
            build_historical_camera_scheduler(
                config(mode="unknown"),
                n_last_frames=10,
                n_rand_frames=20,
            )

    def test_preserve_threshold_must_cover_recent(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least n_last_frames"):
            build_historical_camera_scheduler(
                config(preserve_all_history_until=9),
                n_last_frames=10,
                n_rand_frames=20,
            )

    def test_sample_every_must_be_positive_integer(self) -> None:
        for value in (0, -1, True):
            cfg = config()
            cfg["logging"]["sample_every"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_historical_camera_scheduler(
                    cfg,
                    n_last_frames=10,
                    n_rand_frames=20,
                )


class DeterministicStratifiedTests(unittest.TestCase):
    def test_preserve_all_retains_identity_order_and_new_cameras(self) -> None:
        history = cameras(30)
        new = cameras(2, start=30)
        result = scheduler().select(
            historical_cameras=history,
            new_cameras=new,
            mapper_update_id=0,
            mapping_iteration=0,
        )
        self.assertEqual(result.frames, history + new)
        for actual, expected in zip(result.frames, history + new):
            self.assertIs(actual, expected)
        self.assertFalse(result.selection_active)
        self.assertIsNone(result.active_selection_tick)
        self.assertEqual(result.scheduler_call_count, 0)

    def test_stratified_positions_are_spread_and_sorted_in_frames(self) -> None:
        history = cameras(40)
        new = cameras(2, start=40)
        result = scheduler().select(
            historical_cameras=history,
            new_cameras=new,
            mapper_update_id=0,
            mapping_iteration=0,
        )
        expected_old = list(range(0, 30, 3))
        expected_recent = list(range(30, 40))
        self.assertEqual(
            [camera.uid for camera in result.frames],
            expected_recent + expected_old + [40, 41],
        )
        self.assertEqual(result.active_selection_tick, 0)
        self.assertTrue(result.selection_active)

    def test_active_tick_advances_only_for_partial_budget(self) -> None:
        instance = scheduler()
        inactive = instance.select(
            historical_cameras=cameras(30),
            new_cameras=[],
            mapper_update_id=0,
            mapping_iteration=0,
        )
        first_active = instance.select(
            historical_cameras=cameras(40),
            new_cameras=[],
            mapper_update_id=0,
            mapping_iteration=1,
        )
        second_active = instance.select(
            historical_cameras=cameras(40),
            new_cameras=[],
            mapper_update_id=0,
            mapping_iteration=2,
        )
        self.assertIsNone(inactive.active_selection_tick)
        self.assertEqual(first_active.active_selection_tick, 0)
        self.assertEqual(second_active.active_selection_tick, 1)
        self.assertEqual(instance.scheduler_call_count, 3)
        self.assertEqual(instance.active_selection_tick, 2)

    def test_zero_budget_does_not_advance_active_tick(self) -> None:
        instance = scheduler(history_budget=0)
        result = instance.select(
            historical_cameras=cameras(40),
            new_cameras=[],
            mapper_update_id=0,
            mapping_iteration=0,
        )
        self.assertEqual([camera.uid for camera in result.frames], list(range(30, 40)))
        self.assertFalse(result.selection_active)
        self.assertEqual(instance.active_selection_tick, 0)

    def test_full_old_pool_budget_does_not_advance_active_tick(self) -> None:
        instance = scheduler(history_budget=20, preserve_all_history_until=10)
        history = cameras(25)
        result = instance.select(
            historical_cameras=history,
            new_cameras=[],
            mapper_update_id=0,
            mapping_iteration=0,
        )
        self.assertEqual(
            [camera.uid for camera in result.frames],
            list(range(15, 25)) + list(range(15)),
        )
        self.assertFalse(result.selection_active)
        self.assertEqual(instance.active_selection_tick, 0)

    def test_fixed_pool_l_ticks_select_each_uid_exactly_b_times(self) -> None:
        for pool_size in range(2, 18):
            for budget in range(1, pool_size):
                instance = build_historical_camera_scheduler(
                    config(
                        history_budget=budget,
                        preserve_all_history_until=0,
                    ),
                    n_last_frames=0,
                    n_rand_frames=20,
                )
                assert instance is not None
                counts = {uid: 0 for uid in range(pool_size)}
                for tick in range(pool_size):
                    result = instance.select(
                        historical_cameras=cameras(pool_size),
                        new_cameras=[],
                        mapper_update_id=0,
                        mapping_iteration=tick,
                    )
                    for camera in result.selected_history_cameras:
                        counts[camera.uid] += 1
                with self.subTest(pool_size=pool_size, budget=budget):
                    self.assertEqual(set(counts.values()), {budget})

    def test_two_instances_are_deterministic(self) -> None:
        left = scheduler()
        right = scheduler()
        for iteration in range(8):
            left_result = left.select(
                historical_cameras=cameras(47),
                new_cameras=cameras(2, start=47),
                mapper_update_id=0,
                mapping_iteration=iteration,
            )
            right_result = right.select(
                historical_cameras=cameras(47),
                new_cameras=cameras(2, start=47),
                mapper_update_id=0,
                mapping_iteration=iteration,
            )
            self.assertEqual(
                [camera.uid for camera in left_result.frames],
                [camera.uid for camera in right_result.frames],
            )

    def test_duplicate_and_reverse_call_keys_fail_closed(self) -> None:
        instance = scheduler()
        instance.select(
            historical_cameras=cameras(40),
            new_cameras=[],
            mapper_update_id=1,
            mapping_iteration=2,
        )
        with self.assertRaisesRegex(RuntimeError, "strictly increasing"):
            instance.select(
                historical_cameras=cameras(40),
                new_cameras=[],
                mapper_update_id=1,
                mapping_iteration=2,
            )
        with self.assertRaisesRegex(RuntimeError, "strictly increasing"):
            instance.select(
                historical_cameras=cameras(40),
                new_cameras=[],
                mapper_update_id=1,
                mapping_iteration=1,
            )

    def test_duplicate_or_overlapping_uids_fail_closed(self) -> None:
        instance = scheduler()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            instance.select(
                historical_cameras=[FakeCamera(0), FakeCamera(0)],
                new_cameras=[],
                mapper_update_id=0,
                mapping_iteration=0,
            )
        with self.assertRaisesRegex(ValueError, "overlapping"):
            instance.select(
                historical_cameras=[FakeCamera(0)],
                new_cameras=[FakeCamera(0)],
                mapper_update_id=0,
                mapping_iteration=0,
            )

    def test_non_integer_uid_fails_closed(self) -> None:
        instance = scheduler()
        with self.assertRaisesRegex(TypeError, "integer-compatible"):
            instance.select(
                historical_cameras=[FakeCamera(1.5)],
                new_cameras=[],
                mapper_update_id=0,
                mapping_iteration=0,
            )

    def test_python_and_numpy_rng_states_are_unchanged(self) -> None:
        python_before = random.getstate()
        try:
            import numpy as np
        except ImportError:
            np = None
        numpy_before = np.random.get_state() if np is not None else None

        instance = scheduler()
        for iteration in range(5):
            instance.select(
                historical_cameras=cameras(45),
                new_cameras=[],
                mapper_update_id=0,
                mapping_iteration=iteration,
            )

        self.assertEqual(random.getstate(), python_before)
        if np is not None:
            numpy_after = np.random.get_state()
            self.assertEqual(numpy_before[0], numpy_after[0])
            self.assertTrue((numpy_before[1] == numpy_after[1]).all())
            self.assertEqual(numpy_before[2:], numpy_after[2:])

    def test_scheduler_module_has_no_rng_or_tensor_imports(self) -> None:
        module_path = (
            Path(__file__).parents[1]
            / "src"
            / "camera_scheduling"
            / "historical_camera_scheduler.py"
        )
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imports = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertFalse({"random", "numpy", "torch"} & imports)


class BaselineRandomTests(unittest.TestCase):
    def test_original_selector_is_called_once_and_order_is_unchanged(self) -> None:
        history = cameras(40)
        new = cameras(2, start=40)
        selected = history[-10:] + [
            history[index]
            for index in (
                9, 1, 20, 4, 18, 2, 29, 11, 6, 25,
                0, 14, 7, 21, 3, 27, 12, 5, 23, 16,
            )
        ]
        calls = 0

        def baseline_selector() -> list[FakeCamera]:
            nonlocal calls
            calls += 1
            return selected

        result = scheduler(
            mode="baseline_random",
            history_budget=20,
        ).select(
            historical_cameras=history,
            new_cameras=new,
            mapper_update_id=0,
            mapping_iteration=0,
            baseline_selector=baseline_selector,
        )
        self.assertEqual(calls, 1)
        self.assertEqual(result.frames, selected + new)
        for actual, expected in zip(result.frames, selected + new):
            self.assertIs(actual, expected)

    def test_original_selector_contract_is_checked_without_reordering(self) -> None:
        history = cameras(40)
        invalid = history[-9:] + history[:20]
        with self.assertRaisesRegex(RuntimeError, "protected recent"):
            scheduler(
                mode="baseline_random",
                history_budget=20,
            ).select(
                historical_cameras=history,
                new_cameras=[],
                mapper_update_id=0,
                mapping_iteration=0,
                baseline_selector=lambda: invalid,
            )

    def test_invalid_call_key_fails_before_original_rng_selector(self) -> None:
        history = cameras(40)
        instance = scheduler(mode="baseline_random", history_budget=20)
        calls = 0

        def baseline_selector() -> list[FakeCamera]:
            nonlocal calls
            calls += 1
            return history[-10:] + history[:20]

        instance.select(
            historical_cameras=history,
            new_cameras=[],
            mapper_update_id=0,
            mapping_iteration=0,
            baseline_selector=baseline_selector,
        )
        with self.assertRaises(RuntimeError):
            instance.select(
                historical_cameras=history,
                new_cameras=[],
                mapper_update_id=0,
                mapping_iteration=0,
                baseline_selector=baseline_selector,
            )
        self.assertEqual(calls, 1)

    def test_baseline_random_log_is_valid_without_reordering(self) -> None:
        history = cameras(40)
        selected = history[-10:] + history[:20]
        instance = scheduler(
            mode="baseline_random",
            history_budget=20,
            logging_enabled=True,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            result = instance.select(
                historical_cameras=history,
                new_cameras=[],
                mapper_update_id=0,
                mapping_iteration=0,
                baseline_selector=lambda: selected,
            )
        self.assertEqual(result.frames, selected)
        events = parser.extract_events_from_lines(output.getvalue().splitlines())
        parser.validate_events(events)
        self.assertEqual(events[0]["strategy"], "original_select_keyframes")


class LoggingAndParserTests(unittest.TestCase):
    def _captured_events(self, count: int = 3) -> tuple[str, list[dict]]:
        instance = scheduler(logging_enabled=True)
        output = io.StringIO()
        with redirect_stdout(output):
            for iteration in range(count):
                instance.select(
                    historical_cameras=cameras(40),
                    new_cameras=cameras(2, start=40),
                    mapper_update_id=0,
                    mapping_iteration=iteration,
                )
        text = output.getvalue()
        events = parser.extract_events_from_lines(text.splitlines())
        return text, events

    def test_logged_events_parse_and_replay(self) -> None:
        _, events = self._captured_events()
        parser.validate_events(events)
        summary = parser.build_summary(events)
        self.assertTrue(summary["validation_pass"])
        self.assertTrue(summary["deterministic_replay_pass"])
        self.assertEqual(summary["event_count"], 3)

    def test_tqdm_prefix_trailing_text_and_multiple_markers(self) -> None:
        text, events = self._captured_events(count=2)
        raw_lines = text.splitlines()
        combined = (
            "progress 10% "
            + raw_lines[0]
            + " trailing-terminal-output "
            + raw_lines[1]
            + " done"
        )
        recovered = parser.extract_events_from_lines([combined])
        self.assertEqual(recovered, events)
        parser.validate_events(recovered)

    def test_duplicate_event_id_fails(self) -> None:
        _, events = self._captured_events(count=2)
        events[1]["event_id"] = events[0]["event_id"]
        with self.assertRaisesRegex(parser.ValidationError, "event_id"):
            parser.validate_events(events)

    def test_corrupt_selection_fails_replay(self) -> None:
        _, events = self._captured_events(count=1)
        events[0]["selected_old_history_uids"][0] = 2
        events[0]["selected_history_uids"][10] = 2
        events[0]["final_selected_uids"][10] = 2
        with self.assertRaises(parser.ValidationError):
            parser.validate_events(events)

    def test_bool_schema_and_call_key_fail_closed(self) -> None:
        _, events = self._captured_events(count=1)
        events[0]["schema"] = True
        with self.assertRaisesRegex(parser.ValidationError, "schema"):
            parser.validate_events(events)

        _, events = self._captured_events(count=1)
        events[0]["call_key"][0] = False
        with self.assertRaisesRegex(parser.ValidationError, "call_key"):
            parser.validate_events(events)

    def test_invalid_position_type_fails_closed(self) -> None:
        _, events = self._captured_events(count=1)
        events[0]["base_positions"][0] = True
        with self.assertRaisesRegex(parser.ValidationError, "base_positions"):
            parser.validate_events(events)

    def test_sample_every_controls_only_logging(self) -> None:
        instance = scheduler(logging_enabled=True, sample_every=2)
        output = io.StringIO()
        with redirect_stdout(output):
            for iteration in range(5):
                instance.select(
                    historical_cameras=cameras(40),
                    new_cameras=[],
                    mapper_update_id=0,
                    mapping_iteration=iteration,
                )
        events = parser.extract_events_from_lines(output.getvalue().splitlines())
        self.assertEqual(
            [event["scheduler_call_count"] for event in events],
            [0, 2, 4],
        )
        parser.validate_events(events)

    def test_cli_writes_valid_jsonl_and_refuses_overwrite(self) -> None:
        text, events = self._captured_events(count=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_log = root / "run.log"
            jsonl = root / "events.jsonl"
            summary = root / "summary.json"
            run_log.write_text(text, encoding="utf-8")

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                first_rc = parser.main(
                    [
                        str(run_log),
                        "--jsonl",
                        str(jsonl),
                        "--summary",
                        str(summary),
                    ]
                )
            self.assertEqual(first_rc, 0)
            recovered = [
                json.loads(line)
                for line in jsonl.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(recovered, events)
            built_summary = json.loads(summary.read_text(encoding="utf-8"))
            self.assertTrue(built_summary["validation_pass"])
            run_log_before = run_log.read_bytes()
            jsonl_before = jsonl.read_bytes()
            summary_before = summary.read_bytes()

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                second_rc = parser.main(
                    [
                        str(run_log),
                        "--jsonl",
                        str(jsonl),
                        "--summary",
                        str(summary),
                    ]
                )
            self.assertNotEqual(second_rc, 0)
            self.assertEqual(run_log.read_bytes(), run_log_before)
            self.assertEqual(jsonl.read_bytes(), jsonl_before)
            self.assertEqual(summary.read_bytes(), summary_before)

    def test_invalid_log_creates_no_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_log = root / "run.log"
            jsonl = root / "events.jsonl"
            summary = root / "summary.json"
            run_log.write_text(
                "prefix [HistoricalCameraScheduler] {broken\n",
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                rc = parser.main(
                    [
                        str(run_log),
                        "--jsonl",
                        str(jsonl),
                        "--summary",
                        str(summary),
                    ]
                )
            self.assertNotEqual(rc, 0)
            self.assertFalse(jsonl.exists())
            self.assertFalse(summary.exists())


class OnlineIntegrationSourceContractTests(unittest.TestCase):
    def test_disabled_mapper_branch_keeps_original_selection_expression(self) -> None:
        source = (
            Path(__file__).parents[1] / "src" / "gaussian_mapping.py"
        ).read_text(encoding="utf-8")
        expected = (
            "if self.camera_scheduler is None:\n"
            "                frames = self.select_keyframes()[0] + self.new_cameras"
        )
        self.assertIn(expected, source)

    def test_scheduler_does_not_read_quality_or_gaussian_state(self) -> None:
        source = (
            Path(__file__).parents[1]
            / "src"
            / "camera_scheduling"
            / "historical_camera_scheduler.py"
        ).read_text(encoding="utf-8")
        for forbidden in (
            "last_frame_loss",
            "n_optimized",
            "densify",
            "prune_points",
            "mapping_step",
            "torch.cuda",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
