from __future__ import annotations

import io
import json
import math
import random
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import torch

import src.gaussian_candidate_observer as observer_module
from src.gaussian_candidate_observer import (
    CandidateGenerationMetadata,
    GaussianCandidateObserver,
    build_gaussian_candidate_observer,
    collect_candidate_generation_metadata,
)


def observer_config(
    *,
    mode: str = "observe",
    voxel_size=1.0,
    method: str = "voxel_occupancy",
    gpu_timing: bool = False,
    memory_enabled: bool = False,
    logging_enabled: bool = True,
) -> dict:
    return {
        "mode": mode,
        "coverage": {
            "method": method,
            "voxel_size": voxel_size,
        },
        "timing": {"gpu": gpu_timing},
        "memory": {"enabled": memory_enabled},
        "logging": {"enabled": logging_enabled},
    }


def build_observer(**kwargs) -> GaussianCandidateObserver:
    built = build_gaussian_candidate_observer(
        observer_config(**kwargs),
        resource_admission_mode="observe",
        device="cpu",
    )
    assert built is not None
    return built


def candidates(xyz: torch.Tensor) -> tuple[torch.Tensor, ...]:
    count = int(xyz.shape[0])
    return (
        xyz,
        torch.arange(count * 3, dtype=torch.float32).reshape(count, 3, 1),
        torch.ones((count, 3), dtype=torch.float32),
        torch.nn.functional.pad(
            torch.ones((count, 1), dtype=torch.float32),
            (0, 3),
        ),
        torch.zeros((count, 1), dtype=torch.float32),
    )


def metadata(
    count: int,
    *,
    pre_count: int | None = None,
    valid_depth_count: int = 4,
    depth_pixel_count: int = 4,
) -> CandidateGenerationMetadata:
    return CandidateGenerationMetadata(
        depth_source="estimated_clean_depth",
        depth_pixel_count=depth_pixel_count,
        valid_depth_count=valid_depth_count,
        pre_downsample_point_count=(
            max(count, 6) if pre_count is None else pre_count
        ),
        post_downsample_point_count=count,
    )


def observe(
    xyz: torch.Tensor,
    current: torch.Tensor,
    *,
    built: GaussianCandidateObserver | None = None,
    event_camera_id: int = 7,
) -> tuple[dict, object]:
    built = built or build_observer()
    xyz, features, scales, rotations, opacities = candidates(xyz)
    token = built.observe_before_extend(
        xyz=xyz,
        features=features,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        current_gaussian_xyz=current,
        metadata=metadata(int(xyz.shape[0])),
        mapper_update_id=3,
        source_camera_id=event_camera_id,
        init=False,
    )
    output = io.StringIO()
    with redirect_stdout(output):
        summary = built.record_after_extend(
            token,
            admitted_candidate_count=int(xyz.shape[0]),
            dropped_candidate_count=0,
            gaussian_after_extend=(
                int(current.shape[0]) + int(xyz.shape[0])
            ),
        )
    line = output.getvalue().strip()
    event = json.loads(line[line.index("{") :])
    return event, summary


class CandidateObserverConfigTests(unittest.TestCase):
    def test_missing_config_is_off(self) -> None:
        self.assertIsNone(
            build_gaussian_candidate_observer(
                None,
                resource_admission_mode="disabled",
                device="cpu",
            )
        )

    def test_mode_off_does_not_construct_observer(self) -> None:
        self.assertIsNone(
            build_gaussian_candidate_observer(
                observer_config(mode="off", voxel_size=None),
                resource_admission_mode="fixed_budget",
                device="cpu",
            )
        )

    def test_unknown_mode_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be one of"):
            build_gaussian_candidate_observer(
                observer_config(mode="control"),
                resource_admission_mode="observe",
                device="cpu",
            )

    def test_unknown_coverage_method_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "voxel_occupancy"):
            build_gaussian_candidate_observer(
                observer_config(method="knn"),
                resource_admission_mode="observe",
                device="cpu",
            )

    def test_invalid_voxel_sizes_fail_closed(self) -> None:
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(
                (TypeError, ValueError)
            ):
                build_gaussian_candidate_observer(
                    observer_config(voxel_size=value),
                    resource_admission_mode="observe",
                    device="cpu",
                )

    def test_observe_requires_explicit_voxel_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicitly provided"):
            build_gaussian_candidate_observer(
                observer_config(voxel_size=None),
                resource_admission_mode="observe",
                device="cpu",
            )

    def test_fixed_budget_conflict_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires M01"):
            build_gaussian_candidate_observer(
                observer_config(),
                resource_admission_mode="fixed_budget",
                device="cpu",
            )

    def test_observe_requires_logging(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be true"):
            build_gaussian_candidate_observer(
                observer_config(logging_enabled=False),
                resource_admission_mode="observe",
                device="cpu",
            )

    def test_unknown_field_fails_closed(self) -> None:
        value = observer_config()
        value["typo"] = 1
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            build_gaussian_candidate_observer(
                value,
                resource_admission_mode="observe",
                device="cpu",
            )


class CandidateGenerationMetadataTests(unittest.TestCase):
    def test_depth_metadata_uses_cpu_array_and_depth_truncation(self) -> None:
        depth = np.array(
            [[0.0, 1.0, 100.0], [101.0, np.nan, np.inf]],
            dtype=np.float32,
        )
        result = collect_candidate_generation_metadata(
            depth=depth,
            depth_source="depth_prior",
            pre_downsample_point_count=3,
            post_downsample_point_count=2,
            depth_trunc=100.0,
        )
        self.assertEqual(result.depth_source, "depth_prior")
        self.assertEqual(result.depth_pixel_count, 6)
        self.assertEqual(result.valid_depth_count, 2)
        self.assertIsNone(result.error)

    def test_unknown_depth_source_is_explicit(self) -> None:
        result = collect_candidate_generation_metadata(
            depth=np.ones((2, 2), dtype=np.float32),
            depth_source="typo",
            pre_downsample_point_count=4,
            post_downsample_point_count=4,
            depth_trunc=100.0,
        )
        self.assertEqual(result.depth_source, "unknown")


class CandidateVoxelEvidenceTests(unittest.TestCase):
    def test_initial_one_dimensional_empty_map_is_normalized(self) -> None:
        built = build_observer()
        xyz = torch.tensor(
            [[0.1, 0.1, 0.1], [1.2, 0.0, 0.0]],
            dtype=torch.float32,
        )
        bundle = candidates(xyz)
        identities = [id(tensor) for tensor in bundle]
        clones = [tensor.clone() for tensor in bundle]
        current = torch.empty(0, dtype=torch.float32)

        token = built.observe_before_extend(
            xyz=bundle[0],
            features=bundle[1],
            scales=bundle[2],
            rotations=bundle[3],
            opacities=bundle[4],
            current_gaussian_xyz=current,
            metadata=metadata(2),
            mapper_update_id=0,
            source_camera_id=0,
            init=True,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            summary = built.record_after_extend(
                token,
                admitted_candidate_count=2,
                dropped_candidate_count=0,
                gaussian_after_extend=2,
            )

        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        event = json.loads(lines[0][lines[0].index("{") :])
        self.assertEqual(event["status"], "ok")
        self.assertIsNone(event["error"])
        self.assertTrue(event["init"])
        self.assertEqual(event["candidate_finite_count"], 2)
        self.assertEqual(event["candidate_nonfinite_count"], 0)
        self.assertEqual(event["existing_gaussian_count"], 0)
        self.assertEqual(event["occupied_candidate_count"], 0)
        self.assertEqual(event["novel_candidate_count"], 2)
        self.assertEqual(event["candidate_unique_voxel_count"], 2)
        self.assertEqual(event["admitted_candidate_count"], 2)
        self.assertEqual(event["dropped_candidate_count"], 0)
        self.assertTrue(event["all_candidates_admitted"])
        self.assertTrue(event["conservation_pass"])
        self.assertEqual(event["gaussian_before"], 0)
        self.assertEqual(event["gaussian_after_extend"], 2)
        self.assertEqual([id(tensor) for tensor in bundle], identities)
        for actual, expected in zip(bundle, clones):
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(summary.to_event(), event)

    def test_empty_map_marks_all_finite_candidates_novel(self) -> None:
        event, _ = observe(
            torch.tensor([[0.1, 0.1, 0.1], [1.2, 0.0, 0.0]]),
            torch.empty((0, 3)),
        )
        self.assertEqual(event["status"], "ok")
        self.assertIsNone(event["error"])
        self.assertEqual(event["existing_gaussian_count"], 0)
        self.assertEqual(event["occupied_candidate_count"], 0)
        self.assertEqual(event["novel_candidate_count"], 2)
        self.assertEqual(event["candidate_unique_voxel_count"], 2)

    def test_standard_and_nonempty_maps_preserve_original_objects(self) -> None:
        standard_empty = torch.empty((0, 3), dtype=torch.float32)
        nonempty = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            dtype=torch.float32,
        )

        self.assertIs(
            GaussianCandidateObserver._normalize_current_gaussian_xyz(
                standard_empty
            ),
            standard_empty,
        )
        self.assertIs(
            GaussianCandidateObserver._normalize_current_gaussian_xyz(
                nonempty
            ),
            nonempty,
        )

        event, _ = observe(
            torch.tensor([[0.1, 0.1, 0.1], [2.0, 0.0, 0.0]]),
            nonempty,
        )
        self.assertEqual(event["status"], "ok")
        self.assertEqual(event["existing_gaussian_count"], 2)

    def test_nonempty_one_dimensional_map_remains_invalid(self) -> None:
        with self.assertRaisesRegex(ValueError, r"shape \[G,3\]"):
            GaussianCandidateObserver._normalize_current_gaussian_xyz(
                torch.tensor([1.0, 2.0, 3.0])
            )

        event, _ = observe(
            torch.tensor([[0.1, 0.1, 0.1]]),
            torch.tensor([1.0, 2.0, 3.0]),
        )
        self.assertEqual(event["status"], "error")
        self.assertEqual(event["error"]["type"], "ValueError")

    def test_invalid_two_dimensional_map_remains_invalid(self) -> None:
        with self.assertRaisesRegex(ValueError, r"shape \[G,3\]"):
            GaussianCandidateObserver._normalize_current_gaussian_xyz(
                torch.empty((2, 2))
            )

        event, _ = observe(
            torch.tensor([[0.1, 0.1, 0.1]]),
            torch.empty((2, 2)),
        )
        self.assertEqual(event["status"], "error")
        self.assertEqual(event["error"]["type"], "ValueError")

    def test_empty_candidate_has_null_ratios_and_extents(self) -> None:
        event, _ = observe(
            torch.empty((0, 3)),
            torch.tensor([[0.0, 0.0, 0.0]]),
        )
        self.assertTrue(event["empty_candidate"])
        self.assertIsNone(event["occupied_ratio"])
        self.assertIsNone(event["spatial_extent_x"])

    def test_single_candidate(self) -> None:
        event, _ = observe(
            torch.tensor([[0.2, 0.3, 0.4]]),
            torch.empty((0, 3)),
        )
        self.assertEqual(event["candidate_finite_count"], 1)
        self.assertEqual(event["candidate_unique_voxel_count"], 1)
        self.assertEqual(event["spatial_extent_x"], 0.0)

    def test_all_candidates_in_one_voxel(self) -> None:
        event, _ = observe(
            torch.tensor(
                [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.9, 0.9, 0.9]]
            ),
            torch.empty((0, 3)),
        )
        self.assertEqual(event["candidate_unique_voxel_count"], 1)
        self.assertEqual(event["candidate_intra_voxel_duplicate_count"], 2)

    def test_occupied_and_novel_are_candidate_counts(self) -> None:
        event, _ = observe(
            torch.tensor(
                [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [2.0, 0.0, 0.0]]
            ),
            torch.tensor([[0.5, 0.5, 0.5]]),
        )
        self.assertEqual(event["occupied_candidate_count"], 2)
        self.assertEqual(event["novel_candidate_count"], 1)

    def test_multiple_existing_gaussians_in_one_voxel(self) -> None:
        event, _ = observe(
            torch.tensor([[0.3, 0.3, 0.3]]),
            torch.tensor([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]]),
        )
        self.assertEqual(event["occupied_candidate_count"], 1)

    def test_negative_coordinates_use_floor(self) -> None:
        event, _ = observe(
            torch.tensor([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]),
            torch.tensor([[-0.5, 0.0, 0.0]]),
        )
        self.assertEqual(event["occupied_candidate_count"], 1)
        self.assertEqual(event["novel_candidate_count"], 1)

    def test_voxel_boundary(self) -> None:
        event, _ = observe(
            torch.tensor([[0.999, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            torch.tensor([[1.5, 0.0, 0.0]]),
        )
        self.assertEqual(event["occupied_candidate_count"], 1)
        self.assertEqual(event["novel_candidate_count"], 1)

    def test_nan_candidate_is_diagnostic_only(self) -> None:
        event, _ = observe(
            torch.tensor([[float("nan"), 0.0, 0.0], [1.0, 0.0, 0.0]]),
            torch.empty((0, 3)),
        )
        self.assertEqual(event["candidate_finite_count"], 1)
        self.assertEqual(event["candidate_nonfinite_count"], 1)
        self.assertEqual(event["novel_candidate_count"], 1)
        self.assertNotIn("NaN", json.dumps(event, allow_nan=False))

    def test_inf_candidate_is_diagnostic_only(self) -> None:
        event, _ = observe(
            torch.tensor([[float("inf"), 0.0, 0.0], [1.0, 0.0, 0.0]]),
            torch.empty((0, 3)),
        )
        self.assertEqual(event["candidate_finite_count"], 1)
        self.assertEqual(event["candidate_nonfinite_count"], 1)

    def test_empty_leq5_path_preserves_raw_point_count(self) -> None:
        built = build_observer()
        raw_metadata = metadata(3, pre_count=10)
        token = built.observe_empty(
            current_gaussian_xyz=torch.tensor([[0.0, 0.0, 0.0]]),
            metadata=raw_metadata,
            mapper_update_id=1,
            source_camera_id=2,
            init=True,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            summary = built.record_after_extend(
                token,
                admitted_candidate_count=0,
                dropped_candidate_count=0,
                gaussian_after_extend=1,
            )
        self.assertEqual(summary.candidate_3d_count, 0)
        self.assertEqual(summary.post_downsample_point_count, 3)
        self.assertTrue(summary.conservation_pass)


class CandidateObserverTransparencyTests(unittest.TestCase):
    def test_first_dimension_mismatch_becomes_error_event(self) -> None:
        built = build_observer()
        xyz = torch.zeros((2, 3))
        token = built.observe_before_extend(
            xyz=xyz,
            features=torch.zeros((1, 3, 1)),
            scales=torch.zeros((2, 3)),
            rotations=torch.zeros((2, 4)),
            opacities=torch.zeros((2, 1)),
            current_gaussian_xyz=torch.empty((0, 3)),
            metadata=metadata(2),
            mapper_update_id=0,
            source_camera_id=0,
            init=False,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            summary = built.record_after_extend(
                token,
                admitted_candidate_count=2,
                dropped_candidate_count=0,
                gaussian_after_extend=2,
            )
        self.assertEqual(summary.status, "error")
        self.assertEqual(summary.error["type"], "ValueError")

    def test_inputs_identity_content_grad_and_rng_are_unchanged(self) -> None:
        built = build_observer()
        bundle = candidates(
            torch.tensor([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]])
        )
        current = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
        identities = [id(tensor) for tensor in bundle]
        clones = [tensor.clone() for tensor in bundle]
        requires_grad = [tensor.requires_grad for tensor in bundle]

        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()

        token = built.observe_before_extend(
            xyz=bundle[0],
            features=bundle[1],
            scales=bundle[2],
            rotations=bundle[3],
            opacities=bundle[4],
            current_gaussian_xyz=current,
            metadata=metadata(2),
            mapper_update_id=1,
            source_camera_id=2,
            init=False,
        )

        self.assertEqual([id(tensor) for tensor in bundle], identities)
        for actual, expected in zip(bundle, clones):
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(
            [tensor.requires_grad for tensor in bundle],
            requires_grad,
        )
        self.assertEqual(random.getstate(), python_state)
        current_numpy_state = np.random.get_state()
        self.assertEqual(current_numpy_state[0], numpy_state[0])
        self.assertTrue(
            np.array_equal(current_numpy_state[1], numpy_state[1])
        )
        self.assertEqual(current_numpy_state[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertFalse(
            any(
                isinstance(value, torch.Tensor)
                for value in token.fields.values()
            )
        )

    def test_observer_returns_summary_not_candidates(self) -> None:
        event, summary = observe(
            torch.tensor([[0.1, 0.1, 0.1]]),
            torch.empty((0, 3)),
        )
        self.assertEqual(summary.to_event(), event)
        self.assertFalse(
            any(
                isinstance(value, torch.Tensor)
                for value in summary.to_event().values()
            )
        )

    def test_evidence_error_does_not_prevent_mock_extend(self) -> None:
        built = build_observer()
        xyz = torch.zeros((2, 3))
        token = built.observe_before_extend(
            xyz=xyz,
            features=torch.zeros((1, 3, 1)),
            scales=torch.zeros((2, 3)),
            rotations=torch.zeros((2, 4)),
            opacities=torch.zeros((2, 1)),
            current_gaussian_xyz=torch.empty((0, 3)),
            metadata=metadata(2),
            mapper_update_id=0,
            source_camera_id=0,
            init=False,
        )
        extended = []

        def original_extend() -> int:
            extended.append(True)
            return 2

        gaussian_after = original_extend()
        with redirect_stdout(io.StringIO()):
            summary = built.record_after_extend(
                token,
                admitted_candidate_count=2,
                dropped_candidate_count=0,
                gaussian_after_extend=gaussian_after,
            )
        self.assertEqual(extended, [True])
        self.assertEqual(summary.status, "error")

    def test_json_serialization_rejects_no_values(self) -> None:
        event, _ = observe(
            torch.tensor([[float("nan"), float("inf"), 0.0]]),
            torch.empty((0, 3)),
        )
        encoded = json.dumps(event, allow_nan=False)
        self.assertNotIn("NaN", encoded)
        self.assertNotIn("Infinity", encoded)

    def test_extreme_finite_float32_extent_emits_one_safe_error_event(
        self,
    ) -> None:
        limit = torch.finfo(torch.float32).max
        built = build_observer()
        xyz = torch.tensor(
            [[-limit, 0.0, 0.0], [limit, 0.0, 0.0]],
            dtype=torch.float32,
        )
        bundle = candidates(xyz)
        token = built.observe_before_extend(
            xyz=bundle[0],
            features=bundle[1],
            scales=bundle[2],
            rotations=bundle[3],
            opacities=bundle[4],
            current_gaussian_xyz=torch.empty((0, 3)),
            metadata=metadata(2),
            mapper_update_id=4,
            source_camera_id=8,
            init=False,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            summary = built.record_after_extend(
                token,
                admitted_candidate_count=2,
                dropped_candidate_count=0,
                gaussian_after_extend=2,
            )

        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        event = json.loads(lines[0][lines[0].index("{") :])
        self.assertEqual(event["status"], "error")
        self.assertEqual(event["reason"], "nonfinite_observer_summary")
        self.assertIsNone(event["spatial_extent_x"])
        self.assertEqual(event["admitted_candidate_count"], 2)
        self.assertTrue(event["all_candidates_admitted"])
        self.assertTrue(event["conservation_pass"])
        self.assertEqual(event["gaussian_after_extend"], 2)
        encoded = json.dumps(event, allow_nan=False)
        self.assertNotIn("NaN", encoded)
        self.assertNotIn("Infinity", encoded)
        self.assertNotIn("-Infinity", encoded)
        self.assertEqual(summary.to_event(), event)

    def test_finite_coordinate_extent_remains_ok(self) -> None:
        event, _ = observe(
            torch.tensor([[-2.0, 0.0, 0.0], [3.0, 1.0, 0.0]]),
            torch.empty((0, 3)),
        )
        self.assertEqual(event["status"], "ok")
        self.assertEqual(event["spatial_extent_x"], 5.0)

    def test_first_serialization_failure_emits_one_fallback_error_event(
        self,
    ) -> None:
        built = build_observer()
        xyz = torch.tensor([[0.1, 0.2, 0.3]])
        bundle = candidates(xyz)
        token = built.observe_before_extend(
            xyz=bundle[0],
            features=bundle[1],
            scales=bundle[2],
            rotations=bundle[3],
            opacities=bundle[4],
            current_gaussian_xyz=torch.empty((0, 3)),
            metadata=metadata(1),
            mapper_update_id=5,
            source_camera_id=9,
            init=False,
        )
        real_dumps = json.dumps
        call_count = 0

        def fail_once(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("synthetic serialization failure")
            return real_dumps(*args, **kwargs)

        output = io.StringIO()
        with mock.patch.object(
            observer_module.json,
            "dumps",
            side_effect=fail_once,
        ), redirect_stdout(output):
            summary = built.record_after_extend(
                token,
                admitted_candidate_count=1,
                dropped_candidate_count=0,
                gaussian_after_extend=1,
            )

        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        event = json.loads(lines[0][lines[0].index("{") :])
        self.assertEqual(event["status"], "error")
        self.assertEqual(
            event["reason"],
            "observer_json_serialization_failed",
        )
        self.assertEqual(event["event_id"], "gco-v0:5:9")
        self.assertEqual(event["candidate_3d_count"], 1)
        self.assertEqual(event["admitted_candidate_count"], 1)
        self.assertEqual(event["gaussian_before"], 0)
        self.assertEqual(event["gaussian_after_extend"], 1)
        self.assertTrue(event["conservation_pass"])
        self.assertEqual(summary.to_event(), event)


if __name__ == "__main__":
    unittest.main()
