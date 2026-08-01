from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from tools.analysis import (
    build_gaussian_candidate_selector_dry_run_summary as parser,
)


def valid_event(
    *,
    event_sequence: int = 0,
    mapper_update_id: int = 0,
    source_camera_id: int = 0,
    init: bool = False,
) -> dict:
    return {
        "schema": 1,
        "event_type": "candidate_selector_dry_run",
        "event_id": (
            f"gcs-v0:{event_sequence}:{mapper_update_id}:{source_camera_id}"
        ),
        "event_sequence": event_sequence,
        "status": "ok",
        "reason": "dry_run_observed",
        "mode": "dry_run",
        "algorithm": "occupied_voxel_cap_k_v0",
        "mapper_update_id": mapper_update_id,
        "source_camera_id": source_camera_id,
        "init": init,
        "protected_init": init,
        "coverage_method": "voxel_occupancy",
        "coverage_voxel_size": 0.05,
        "k_values": [1, 2, 4, 8],
        "candidate_count": 7,
        "candidate_finite_count": 7,
        "candidate_nonfinite_count": 0,
        "existing_gaussian_count": 10,
        "candidate_unique_voxel_count": 4,
        "occupied_candidate_count": 5,
        "novel_candidate_count": 2,
        "occupied_unique_voxel_count": 2,
        "novel_unique_voxel_count": 2,
        "occupied_multiplicity_histogram": [
            {"multiplicity": 2, "voxel_count": 1},
            {"multiplicity": 3, "voxel_count": 1},
        ],
        "occupied_multiplicity_mean": 2.5,
        "occupied_multiplicity_max": 3,
        "occupied_multiplicity_p50": 2,
        "occupied_multiplicity_p90": 3,
        "occupied_multiplicity_p95": 3,
        "occupied_multiplicity_p99": 3,
        "k_scan": [
            {
                "k": 1,
                "estimated_admitted_count": 4,
                "estimated_dropped_count": 3,
                "admitted_ratio": 4 / 7,
                "dropped_ratio": 3 / 7,
            },
            {
                "k": 2,
                "estimated_admitted_count": 6,
                "estimated_dropped_count": 1,
                "admitted_ratio": 6 / 7,
                "dropped_ratio": 1 / 7,
            },
            {
                "k": 4,
                "estimated_admitted_count": 7,
                "estimated_dropped_count": 0,
                "admitted_ratio": 1.0,
                "dropped_ratio": 0.0,
            },
            {
                "k": 8,
                "estimated_admitted_count": 7,
                "estimated_dropped_count": 0,
                "admitted_ratio": 1.0,
                "dropped_ratio": 0.0,
            },
        ],
        "selection_applied": False,
        "selected_indices": None,
        "all_candidates_forwarded": True,
        "gaussian_before": 10,
        "actual_admitted_count": 7,
        "actual_dropped_count": 0,
        "gaussian_after_extend": 17,
        "actual_conservation_pass": True,
        "empty_candidate": False,
        "dry_run_wall_ms": 0.25,
        "error": None,
    }


def line_for(event: dict) -> str:
    return parser.PREFIX + " " + json.dumps(
        event,
        allow_nan=False,
        separators=(",", ":"),
    )


class SelectorParserValidationTests(unittest.TestCase):
    def test_valid_event_passes(self) -> None:
        event = valid_event()
        parser.validate_event(event)
        self.assertEqual(parser.validate_events([event]), [event])

    def test_histogram_recomputes_every_k(self) -> None:
        event = valid_event()
        event["k_scan"][1]["estimated_admitted_count"] += 1
        with self.assertRaisesRegex(parser.ValidationError, "recomputed"):
            parser.validate_event(event)

    def test_histogram_count_and_quantile_corruption_fail(self) -> None:
        event = valid_event()
        event["occupied_multiplicity_histogram"][0]["voxel_count"] = 2
        with self.assertRaises(parser.ValidationError):
            parser.validate_event(event)
        event = valid_event()
        event["occupied_multiplicity_p90"] = 2
        with self.assertRaisesRegex(parser.ValidationError, "p90"):
            parser.validate_event(event)

    def test_protected_init_keeps_every_candidate_for_all_k(self) -> None:
        event = valid_event(init=True)
        for item in event["k_scan"]:
            item.update(
                {
                    "estimated_admitted_count": 7,
                    "estimated_dropped_count": 0,
                    "admitted_ratio": 1.0,
                    "dropped_ratio": 0.0,
                }
            )
        parser.validate_event(event)
        event["protected_init"] = False
        with self.assertRaisesRegex(parser.ValidationError, "protected_init"):
            parser.validate_event(event)

    def test_error_event_is_structurally_valid_but_formally_rejected(self) -> None:
        event = valid_event()
        event["status"] = "error"
        event["reason"] = "nonfinite_candidate_coordinates"
        event["error"] = {"type": "Error", "message": "synthetic"}
        parser.validate_event(event)
        with self.assertRaisesRegex(parser.ValidationError, "error events"):
            parser.validate_events([event])

    def test_duplicate_event_id_and_sequence_gap_fail(self) -> None:
        event = valid_event()
        with self.assertRaisesRegex(parser.ValidationError, "Duplicate"):
            parser.validate_events([event, copy.deepcopy(event)])
        event = valid_event(event_sequence=1)
        with self.assertRaisesRegex(parser.ValidationError, "event_sequence"):
            parser.validate_events([event])

    def test_event_sequence_allows_repeated_camera_key(self) -> None:
        first = valid_event()
        second = valid_event(event_sequence=1)
        validated = parser.validate_events([first, second])
        self.assertEqual([item["event_sequence"] for item in validated], [0, 1])

    def test_selection_output_is_forbidden(self) -> None:
        event = valid_event()
        event["selection_applied"] = True
        event["selected_indices"] = [0]
        with self.assertRaisesRegex(parser.ValidationError, "cannot apply"):
            parser.validate_event(event)

    def test_missing_extra_nonfinite_and_wrong_types_fail(self) -> None:
        event = valid_event()
        del event["dry_run_wall_ms"]
        with self.assertRaisesRegex(parser.ValidationError, "field mismatch"):
            parser.validate_event(event)
        event = valid_event()
        event["extra"] = 1
        with self.assertRaisesRegex(parser.ValidationError, "field mismatch"):
            parser.validate_event(event)
        event = valid_event()
        event["dry_run_wall_ms"] = float("nan")
        with self.assertRaises(parser.ValidationError):
            parser.validate_event(event)
        event = valid_event()
        event["init"] = 1
        event["protected_init"] = 1
        with self.assertRaises(parser.ValidationError):
            parser.validate_event(event)


class SelectorParserExtractionTests(unittest.TestCase):
    def test_tqdm_trailing_and_multiple_events(self) -> None:
        first = valid_event()
        second = valid_event(
            event_sequence=1,
            mapper_update_id=1,
            source_camera_id=1,
        )
        events = parser.extract_events_from_lines(
            [
                "progress "
                + line_for(first)
                + " trailing "
                + line_for(second)
                + " done"
            ]
        )
        self.assertEqual(events, [first, second])

    def test_invalid_json_and_no_events_fail(self) -> None:
        with self.assertRaises(parser.ValidationError):
            parser.extract_events_from_lines([parser.PREFIX + " {"])
        with self.assertRaises(parser.ValidationError):
            parser.extract_events_from_lines(["nothing"])


class SelectorParserOutputTests(unittest.TestCase):
    def test_summary_and_outputs_are_valid_and_refuse_overwrite(self) -> None:
        event = valid_event()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_log = root / "run.log"
            run_log.write_text(line_for(event) + "\n", encoding="utf-8")
            output_dir = root / "summary"
            with redirect_stdout(io.StringIO()):
                result = parser.main(
                    [str(run_log), "--output-dir", str(output_dir)]
                )
            self.assertEqual(result, 0)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                set(parser.OUTPUT_FILENAMES),
            )
            for line in (
                output_dir / "candidate_selector_dry_run_events.jsonl"
            ).read_text(encoding="utf-8").splitlines():
                json.loads(line)
            summary = json.loads(
                (
                    output_dir / "candidate_selector_dry_run_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(summary["validation_pass"])
            self.assertEqual(summary["k_scan_totals"][0]["estimated_admitted_total"], 4)
            self.assertEqual(
                summary["occupied_multiplicity_histogram"],
                [
                    {"multiplicity": 2, "voxel_count": 1},
                    {"multiplicity": 3, "voxel_count": 1},
                ],
            )
            self.assertEqual(summary["occupied_multiplicity_mean"], 2.5)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                second = parser.main(
                    [str(run_log), "--output-dir", str(output_dir)]
                )
            self.assertEqual(second, 1)
            self.assertIn("Refusing to overwrite", stderr.getvalue())

    def test_invalid_input_leaves_no_outputs(self) -> None:
        event = valid_event()
        event["k_scan"][0]["estimated_admitted_count"] = 99
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_log = root / "run.log"
            run_log.write_text(line_for(event) + "\n", encoding="utf-8")
            output_dir = root / "summary"
            with redirect_stderr(io.StringIO()):
                result = parser.main(
                    [str(run_log), "--output-dir", str(output_dir)]
                )
            self.assertEqual(result, 1)
            self.assertFalse(output_dir.exists())

    def test_summary_is_counterfactual_and_aggregated(self) -> None:
        event = valid_event()
        summary = parser.build_summary([event])
        self.assertTrue(summary["validation_pass"])
        self.assertIn("unmodified map trajectory", summary["interpretation_limit"])
        self.assertTrue(summary["config_fingerprint"].startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()
