from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from tools.analysis import build_gaussian_candidate_observer_summary as parser


def valid_event(
    *,
    mapper_update_id: int = 2,
    source_camera_id: int = 5,
) -> dict:
    return {
        "schema": 1,
        "event_type": "candidate_observation",
        "event_id": (
            f"gco-v0:{mapper_update_id}:{source_camera_id}"
        ),
        "status": "ok",
        "reason": "observed",
        "mapper_update_id": mapper_update_id,
        "source_camera_id": source_camera_id,
        "init": False,
        "observer_mode": "observe",
        "gpu_timing_enabled": False,
        "memory_observation_enabled": False,
        "depth_source": "estimated_clean_depth",
        "depth_pixel_count": 4,
        "valid_depth_count": 3,
        "valid_depth_ratio": 0.75,
        "pre_downsample_point_count": 4,
        "post_downsample_point_count": 2,
        "candidate_3d_count": 2,
        "candidate_finite_count": 2,
        "candidate_nonfinite_count": 0,
        "existing_gaussian_count": 1,
        "coverage_method": "voxel_occupancy",
        "coverage_voxel_size": 1.0,
        "candidate_unique_voxel_count": 2,
        "candidate_intra_voxel_duplicate_count": 0,
        "occupied_candidate_count": 1,
        "novel_candidate_count": 1,
        "occupied_ratio": 0.5,
        "novel_ratio": 0.5,
        "spatial_extent_x": 1.0,
        "spatial_extent_y": 0.0,
        "spatial_extent_z": 0.0,
        "observer_cpu_ms": 0.125,
        "observer_gpu_ms": None,
        "allocated_before_bytes": None,
        "allocated_after_bytes": None,
        "reserved_before_bytes": None,
        "reserved_after_bytes": None,
        "gaussian_before": 1,
        "gaussian_after_extend": 3,
        "admitted_candidate_count": 2,
        "dropped_candidate_count": 0,
        "empty_candidate": False,
        "all_candidates_admitted": True,
        "conservation_pass": True,
        "error": None,
    }


def line_for(event: dict) -> str:
    return (
        parser.PREFIX
        + " "
        + json.dumps(event, allow_nan=False, separators=(",", ":"))
    )


class CandidateObserverParserExtractionTests(unittest.TestCase):
    def test_valid_event_passes(self) -> None:
        event = valid_event()
        parser.validate_event(event)
        validated = parser.validate_events([event])
        self.assertEqual(validated, [event])

    def test_tqdm_prefix_and_trailing_text(self) -> None:
        event = valid_event()
        events = parser.extract_events_from_lines(
            ["progress 20% " + line_for(event) + " trailing"]
        )
        self.assertEqual(events, [event])

    def test_multiple_events_on_one_physical_line(self) -> None:
        first = valid_event()
        second = valid_event(mapper_update_id=3, source_camera_id=6)
        events = parser.extract_events_from_lines(
            [line_for(first) + " noise " + line_for(second)]
        )
        self.assertEqual(events, [first, second])

    def test_duplicate_event_fails(self) -> None:
        event = valid_event()
        with self.assertRaisesRegex(parser.ValidationError, "Duplicate"):
            parser.validate_events([event, copy.deepcopy(event)])

    def test_missing_field_fails(self) -> None:
        event = valid_event()
        del event["candidate_finite_count"]
        with self.assertRaisesRegex(parser.ValidationError, "field mismatch"):
            parser.validate_event(event)

    def test_conservation_failure_fails(self) -> None:
        event = valid_event()
        event["gaussian_after_extend"] += 1
        with self.assertRaisesRegex(parser.ValidationError, "conservation"):
            parser.validate_event(event)

    def test_status_error_fails_formal_validation(self) -> None:
        event = valid_event()
        event["status"] = "error"
        event["reason"] = "observer_evidence_error"
        event["error"] = {"type": "RuntimeError", "message": "synthetic"}
        with self.assertRaisesRegex(parser.ValidationError, "error events"):
            parser.validate_events([event])

    def test_nonfinite_summary_error_with_null_extent_is_rejected_formally(
        self,
    ) -> None:
        event = valid_event()
        event["status"] = "error"
        event["reason"] = "nonfinite_observer_summary"
        event["spatial_extent_x"] = None
        event["spatial_extent_y"] = None
        event["spatial_extent_z"] = None
        event["error"] = {
            "type": "NonFiniteObserverSummaryError",
            "message": "Synthetic non-finite extent.",
        }
        parser.validate_event(event)
        with self.assertRaisesRegex(parser.ValidationError, "error events"):
            parser.validate_events([event])

    def test_nonfinite_json_number_fails(self) -> None:
        event = valid_event()
        event["observer_cpu_ms"] = float("nan")
        text = parser.PREFIX + " " + json.dumps(event)
        with self.assertRaises(parser.ValidationError):
            parser.extract_events_from_lines([text])

    def test_invalid_ratio_fails(self) -> None:
        event = valid_event()
        event["occupied_ratio"] = 0.25
        with self.assertRaisesRegex(parser.ValidationError, "does not match"):
            parser.validate_event(event)

    def test_negative_count_fails(self) -> None:
        event = valid_event()
        event["candidate_nonfinite_count"] = -1
        with self.assertRaisesRegex(parser.ValidationError, "non-negative"):
            parser.validate_event(event)


class CandidateObserverParserOutputTests(unittest.TestCase):
    def test_summary_contains_required_aggregates(self) -> None:
        first = valid_event()
        second = valid_event(mapper_update_id=3, source_camera_id=6)
        summary = parser.build_summary([first, second])
        self.assertTrue(summary["validation_pass"])
        self.assertEqual(summary["event_count"], 2)
        self.assertEqual(summary["candidate_total"], 4)
        self.assertEqual(summary["occupied_candidate_total"], 2)
        self.assertEqual(summary["novel_candidate_total"], 2)
        self.assertTrue(summary["config_fingerprint"].startswith("sha256:"))

    def test_config_fingerprint_distinguishes_monitoring_scope(self) -> None:
        baseline = valid_event()
        gpu_timed = valid_event()
        gpu_timed["gpu_timing_enabled"] = True
        gpu_timed["observer_gpu_ms"] = 0.25
        memory_observed = valid_event()
        memory_observed["memory_observation_enabled"] = True
        memory_observed["allocated_before_bytes"] = 100
        memory_observed["allocated_after_bytes"] = 120
        memory_observed["reserved_before_bytes"] = 200
        memory_observed["reserved_after_bytes"] = 220

        fingerprints = {
            parser.build_summary([event])["config_fingerprint"]
            for event in (baseline, gpu_timed, memory_observed)
        }
        self.assertEqual(len(fingerprints), 3)

    def test_cli_writes_all_outputs_and_jsonl_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_log = root / "run.log"
            run_log.write_text(
                line_for(valid_event()) + "\n",
                encoding="utf-8",
            )
            output_dir = root / "summary"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = parser.main(
                    [
                        str(run_log),
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            self.assertEqual(result, 0)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                set(parser.OUTPUT_FILENAMES),
            )
            for line in (
                output_dir / "candidate_observer_events.jsonl"
            ).read_text(encoding="utf-8").splitlines():
                json.loads(line)
            json.loads(
                (
                    output_dir / "candidate_observer_summary.json"
                ).read_text(encoding="utf-8")
            )

    def test_existing_output_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_log = root / "run.log"
            run_log.write_text(
                line_for(valid_event()) + "\n",
                encoding="utf-8",
            )
            output_dir = root / "existing"
            output_dir.mkdir()
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = parser.main(
                    [
                        str(run_log),
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            self.assertEqual(result, 1)
            self.assertIn("refusing overwrite", stderr.getvalue())
            self.assertEqual(list(output_dir.iterdir()), [])

    def test_validation_failure_leaves_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_log = root / "run.log"
            event = valid_event()
            event["conservation_pass"] = False
            original_log = line_for(event) + "\n"
            run_log.write_text(original_log, encoding="utf-8")
            output_dir = root / "summary"
            with redirect_stderr(io.StringIO()):
                result = parser.main(
                    [
                        str(run_log),
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            self.assertEqual(result, 1)
            self.assertFalse(output_dir.exists())
            self.assertEqual(
                run_log.read_text(encoding="utf-8"),
                original_log,
            )


if __name__ == "__main__":
    unittest.main()
