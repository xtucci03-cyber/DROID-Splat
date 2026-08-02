from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from src.candidate_selection.gaussian_candidate_selector_v1 import (
    GaussianCandidateSelectorV1,
    LOG_PREFIX,
    build_gaussian_candidate_selector_v1,
)


class FakeCamera:
    def __init__(self, depth: torch.Tensor, pose: torch.Tensor | None = None):
        self.uid = 7
        self.buffer_index = 3
        self.source_frame_id = 101
        self.source_timestamp = 12.5
        self.fx = 100.0
        self.fy = 100.0
        self.cx = 0.0
        self.cy = 0.0
        self.image_height = int(depth.shape[0])
        self.image_width = int(depth.shape[1])
        self.depth = depth
        self.depth_prior = depth.clone()
        self._pose = torch.eye(4) if pose is None else pose

    @property
    def pose(self):
        return self._pose


class SnapshotGetter:
    def __init__(self, confidence: torch.Tensor):
        self.confidence = confidence
        self.calls = 0
        self.overrides = {}

    def __call__(self, camera, require_current=True):
        self.calls += 1
        values = {
            "buffer_index": camera.buffer_index,
            "source_frame_id": camera.source_frame_id,
            "source_timestamp": camera.source_timestamp,
            "confidence_source_frame_id": camera.source_frame_id,
            "confidence_version": 4,
            "confidence_up_version": 4,
            "is_current": True,
            "is_stale": False,
            "confidence": self.confidence.clone(),
            "shape": tuple(self.confidence.shape),
            "dtype": str(self.confidence.dtype),
            "device": str(self.confidence.device),
            "requires_grad": self.confidence.requires_grad,
        }
        values.update(self.overrides)
        return SimpleNamespace(**values)


def make_candidates(device="cpu"):
    # With fx=100 and z=1 these are source pixels (0,0), (1,0),
    # (2,0), (6,1).  The first three share a 5 cm world voxel.
    xyz = torch.tensor(
        [[0.00, 0.00, 1.0], [0.01, 0.00, 1.0], [0.02, 0.00, 1.0], [0.06, 0.01, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    count = xyz.shape[0]
    return (
        xyz,
        torch.arange(count * 3, dtype=torch.float32, device=device).reshape(count, 3, 1),
        torch.ones(count, 1, dtype=torch.float32, device=device),
        torch.nn.functional.one_hot(
            torch.zeros(count, dtype=torch.long, device=device), num_classes=4
        ).to(dtype=torch.float32),
        torch.zeros(count, 1, dtype=torch.float32, device=device),
    )


def make_fixture():
    depth = torch.zeros(3, 8, dtype=torch.float32)
    depth[0, 0:3] = 1.0
    depth[1, 6] = 1.0
    confidence = torch.zeros_like(depth)
    confidence[0, 0] = 0.1
    confidence[0, 1] = 0.9
    confidence[0, 2] = 0.5
    confidence[1, 6] = 0.7
    camera = FakeCamera(depth)
    getter = SnapshotGetter(confidence)
    observer = GaussianCandidateSelectorV1(
        voxel_size=0.05,
        confidence_snapshot_getter=getter,
        logging_enabled=True,
        device="cpu",
    )
    candidates = make_candidates()
    current_map = candidates[0][0:1].clone()
    return observer, getter, camera, candidates, current_map


def observe(observer, camera, candidates, current_map, *, init=False):
    return observer.observe_before_extend(
        xyz=candidates[0],
        features=candidates[1],
        scales=candidates[2],
        rotations=candidates[3],
        opacities=candidates[4],
        current_gaussian_xyz=current_map,
        camera=camera,
        depthmap=None,
        depth_source="estimated_clean_depth",
        mapper_update_id=11,
        init=init,
    )


class GaussianCandidateSelectorV1Tests(unittest.TestCase):
    def test_factory_missing_and_off_are_exact_none(self):
        getter = SnapshotGetter(torch.zeros(2, 2))
        candidate_observer = SimpleNamespace(voxel_size=0.05)
        self.assertIsNone(
            build_gaussian_candidate_selector_v1(
                None,
                candidate_observer=candidate_observer,
                resource_admission_mode="disabled",
                confidence_snapshot_getter=getter,
                device="cpu",
            )
        )
        self.assertIsNone(
            build_gaussian_candidate_selector_v1(
                {"mode": "off"},
                candidate_observer=candidate_observer,
                resource_admission_mode="disabled",
                confidence_snapshot_getter=getter,
                device="cpu",
            )
        )
        self.assertEqual(getter.calls, 0)

    def test_factory_fail_closed_configuration(self):
        getter = SnapshotGetter(torch.zeros(2, 2))
        with self.assertRaises(ValueError):
            build_gaussian_candidate_selector_v1(
                {"mode": "observe"},
                candidate_observer=None,
                resource_admission_mode="disabled",
                confidence_snapshot_getter=getter,
                device="cpu",
            )
        with self.assertRaises(ValueError):
            build_gaussian_candidate_selector_v1(
                {"mode": "observe"},
                candidate_observer=SimpleNamespace(voxel_size=0.05),
                resource_admission_mode="fixed_budget",
                confidence_snapshot_getter=getter,
                device="cpu",
            )
        with self.assertRaises(ValueError):
            build_gaussian_candidate_selector_v1(
                {"mode": "active_topk"},
                candidate_observer=SimpleNamespace(voxel_size=0.05),
                resource_admission_mode="disabled",
                confidence_snapshot_getter=getter,
                device="cpu",
            )

    def test_projection_recovers_source_pixels_and_original_indices(self):
        observer, _, camera, candidates, _ = make_fixture()
        u, v, z, valid = observer.project_world_to_source_pixels(
            candidates[0], camera
        )
        self.assertEqual(u.tolist(), [0, 1, 2, 6])
        self.assertEqual(v.tolist(), [0, 0, 0, 1])
        self.assertEqual(z.tolist(), [1.0, 1.0, 1.0, 1.0])
        self.assertTrue(bool(valid.all()))

    def test_observe_is_no_mutation_and_all_candidates_forwarded(self):
        observer, getter, camera, candidates, current_map = make_fixture()
        clones = [value.clone() for value in candidates]
        pointers = [value.data_ptr() for value in candidates]
        versions = [value._version for value in candidates]
        token = observe(observer, camera, candidates, current_map)
        self.assertEqual(token.fields["status"], "ok")
        self.assertEqual(token.fields["raw_candidate_count"], 4)
        self.assertEqual(token.fields["valid_confidence_count"], 4)
        self.assertEqual(token.fields["occupied_candidate_count"], 3)
        self.assertEqual(token.fields["novel_candidate_count"], 1)
        self.assertTrue(token.fields["observer_no_mutation"])
        self.assertNotIn("selected_indices", token.fields)
        self.assertEqual(getter.calls, 1)
        for index, value in enumerate(candidates):
            self.assertEqual(value.data_ptr(), pointers[index])
            self.assertEqual(value._version, versions[index])
            self.assertTrue(torch.equal(value, clones[index]))

        stream = io.StringIO()
        with redirect_stdout(stream):
            summary = observer.record_after_extend(
                token,
                admitted_candidate_count=4,
                dropped_candidate_count=0,
                gaussian_after_extend=5,
            )
        self.assertTrue(summary.fields["all_candidates_forwarded"])
        self.assertTrue(summary.fields["actual_conservation_pass"])
        self.assertTrue(stream.getvalue().startswith(LOG_PREFIX + " "))
        event = json.loads(stream.getvalue().split(" ", 1)[1])
        self.assertEqual(event["status"], "ok")
        self.assertFalse(event["active_topk_applied"])
        self.assertFalse(event["selected_indices_created"])

    def test_counterfactual_topk_is_stable_and_quality_ordered(self):
        observer, _, camera, candidates, current_map = make_fixture()
        first = observe(observer, camera, candidates, current_map).fields
        second = observe(observer, camera, candidates, current_map).fields
        self.assertEqual(
            first["candidate_ordering_fingerprint"],
            second["candidate_ordering_fingerprint"],
        )
        first_scan = first["counterfactual_topk"]
        second_scan = second["counterfactual_topk"]
        self.assertEqual(first_scan, second_scan)
        self.assertEqual([row["k"] for row in first_scan], [1, 2, 4, 8])
        self.assertEqual(first_scan[0]["counterfactual_selected_occupied_count"], 1)
        self.assertAlmostEqual(first_scan[0]["selected_confidence"]["mean"], 0.9)
        self.assertEqual(first_scan[1]["counterfactual_selected_occupied_count"], 2)
        self.assertEqual(first_scan[2]["counterfactual_selected_occupied_count"], 3)

    def test_empty_candidates_do_not_read_confidence(self):
        observer, getter, camera, candidates, current_map = make_fixture()
        empty = tuple(value[:0] for value in candidates)
        token = observe(observer, camera, empty, current_map)
        self.assertEqual(token.fields["status"], "ok")
        self.assertEqual(token.fields["raw_candidate_count"], 0)
        self.assertEqual(token.fields["valid_confidence_count"], 0)
        self.assertTrue(token.fields["observer_no_mutation"])
        self.assertEqual(getter.calls, 0)

    def test_init_with_empty_legacy_map_is_observed_only(self):
        observer, _, camera, candidates, _ = make_fixture()
        token = observe(
            observer,
            camera,
            candidates,
            torch.empty(0),
            init=True,
        )
        self.assertEqual(token.fields["status"], "ok")
        self.assertTrue(token.fields["protected_init"])
        self.assertEqual(token.fields["occupied_candidate_count"], 0)
        self.assertEqual(token.fields["novel_candidate_count"], 4)

    def test_wrong_pose_direction_or_depth_lineage_fails_closed(self):
        observer, _, camera, candidates, current_map = make_fixture()
        wrong_pose = torch.eye(4)
        wrong_pose[0, 3] = 10.0
        wrong_camera = FakeCamera(camera.depth, wrong_pose)
        token = observe(observer, wrong_camera, candidates, current_map)
        self.assertEqual(token.fields["status"], "error")
        self.assertEqual(
            token.fields["reason"], "invalid_projection_or_depth_lineage"
        )
        self.assertTrue(token.fields["observer_no_mutation"])

        mismatch_camera = FakeCamera(torch.full_like(camera.depth, 2.0))
        token = observe(observer, mismatch_camera, candidates, current_map)
        self.assertEqual(token.fields["status"], "error")
        self.assertEqual(token.fields["depth_consistent_count"], 0)

    def test_nonfinite_xyz_and_out_of_bounds_fail_closed(self):
        observer, _, camera, candidates, current_map = make_fixture()
        invalid = list(candidates)
        invalid[0] = invalid[0].clone()
        invalid[0][1, 0] = torch.nan
        token = observe(observer, camera, tuple(invalid), current_map)
        self.assertEqual(token.fields["status"], "error")
        self.assertEqual(token.fields["reason"], "nonfinite_xyz")

        out_of_bounds = list(candidates)
        out_of_bounds[0] = out_of_bounds[0].clone()
        out_of_bounds[0][-1, 0] = 100.0
        token = observe(observer, camera, tuple(out_of_bounds), current_map)
        self.assertEqual(token.fields["status"], "error")
        self.assertEqual(token.fields["in_bounds_count"], 3)

    def test_snapshot_identity_timestamp_version_and_shape_are_checked(self):
        cases = (
            ({"source_frame_id": 999}, "confidence_identity_mismatch"),
            ({"source_timestamp": 99.0}, "confidence_timestamp_mismatch"),
            (
                {
                    "confidence_up_version": 3,
                    "is_current": False,
                    "is_stale": True,
                },
                "stale_confidence",
            ),
            ({"shape": (1, 1)}, "confidence_shape_mismatch"),
        )
        for overrides, expected_reason in cases:
            with self.subTest(overrides=overrides):
                observer, getter, camera, candidates, current_map = make_fixture()
                getter.overrides.update(overrides)
                token = observe(observer, camera, candidates, current_map)
                self.assertEqual(token.fields["status"], "error")
                self.assertEqual(token.fields["reason"], expected_reason)
                self.assertTrue(token.fields["observer_no_mutation"])

    def test_missing_nonfinite_and_mismatched_confidence_fail_closed(self):
        observer, getter, camera, candidates, current_map = make_fixture()
        getter.confidence[0, 0] = torch.inf
        token = observe(observer, camera, candidates, current_map)
        self.assertEqual(token.fields["status"], "error")
        self.assertEqual(token.fields["reason"], "nonfinite_confidence")

        observer, getter, camera, candidates, current_map = make_fixture()
        getter.confidence = torch.zeros(2, 2)
        token = observe(observer, camera, candidates, current_map)
        self.assertEqual(token.fields["status"], "error")
        self.assertEqual(token.fields["reason"], "confidence_shape_mismatch")

        observer, getter, camera, candidates, current_map = make_fixture()

        def missing(*args, **kwargs):
            raise RuntimeError("confidence is not valid")

        observer._confidence_snapshot_getter = missing
        token = observe(observer, camera, candidates, current_map)
        self.assertEqual(token.fields["status"], "error")
        self.assertEqual(token.fields["reason"], "missing_confidence")

    def test_confidence_candidate_device_mismatch_is_classified_and_forwarded(self):
        observer, getter, camera, candidates, current_map = make_fixture()
        clones = [value.clone() for value in candidates]
        pointers = [value.data_ptr() for value in candidates]
        versions = [value._version for value in candidates]
        getter.confidence = torch.empty(
            camera.image_height,
            camera.image_width,
            device="meta",
        )

        token = observe(observer, camera, candidates, current_map)

        self.assertEqual(token.fields["status"], "error")
        self.assertEqual(token.fields["reason"], "confidence_device_mismatch")
        self.assertEqual(getter.calls, 1)
        self.assertTrue(token.fields["observer_no_mutation"])
        for index, value in enumerate(candidates):
            self.assertEqual(value.data_ptr(), pointers[index])
            self.assertEqual(value._version, versions[index])
            self.assertTrue(torch.equal(value, clones[index]))

        with redirect_stdout(io.StringIO()):
            summary = observer.record_after_extend(
                token,
                admitted_candidate_count=4,
                dropped_candidate_count=0,
                gaussian_after_extend=5,
            )
        self.assertTrue(summary.fields["all_candidates_forwarded"])
        self.assertTrue(summary.fields["actual_conservation_pass"])

    def test_source_depth_camera_shape_mismatch_precedes_confidence_read(self):
        observer, getter, camera, candidates, current_map = make_fixture()
        clones = [value.clone() for value in candidates]
        pointers = [value.data_ptr() for value in candidates]
        versions = [value._version for value in candidates]
        camera.image_height += 1

        token = observe(observer, camera, candidates, current_map)

        self.assertEqual(token.fields["status"], "error")
        self.assertEqual(
            token.fields["reason"],
            "source_depth_camera_shape_mismatch",
        )
        self.assertEqual(getter.calls, 0)
        self.assertTrue(token.fields["observer_no_mutation"])
        for index, value in enumerate(candidates):
            self.assertEqual(value.data_ptr(), pointers[index])
            self.assertEqual(value._version, versions[index])
            self.assertTrue(torch.equal(value, clones[index]))

        with redirect_stdout(io.StringIO()):
            summary = observer.record_after_extend(
                token,
                admitted_candidate_count=4,
                dropped_candidate_count=0,
                gaussian_after_extend=5,
            )
        self.assertTrue(summary.fields["all_candidates_forwarded"])
        self.assertTrue(summary.fields["actual_conservation_pass"])

    def test_all_zero_confidence_is_valid_evidence(self):
        observer, getter, camera, candidates, current_map = make_fixture()
        getter.confidence.zero_()
        token = observe(observer, camera, candidates, current_map)
        self.assertEqual(token.fields["status"], "ok")
        self.assertEqual(token.fields["valid_confidence_count"], 4)
        self.assertEqual(token.fields["confidence_zero_count"], 4)
        self.assertEqual(token.fields["confidence_zero_ratio"], 1.0)

    def test_nonpositive_z_error_event_is_still_valid_json(self):
        observer, _, camera, candidates, current_map = make_fixture()
        invalid = list(candidates)
        invalid[0] = invalid[0].clone()
        invalid[0][0, 2] = -1.0
        token = observe(observer, camera, tuple(invalid), current_map)
        self.assertEqual(token.fields["status"], "error")
        self.assertLess(token.fields["positive_camera_z_count"], 4)
        stream = io.StringIO()
        with redirect_stdout(stream):
            summary = observer.record_after_extend(
                token,
                admitted_candidate_count=4,
                dropped_candidate_count=0,
                gaussian_after_extend=5,
            )
        event = json.loads(stream.getvalue().split(" ", 1)[1])
        self.assertEqual(event["status"], "error")
        self.assertTrue(event["all_candidates_forwarded"])
        self.assertTrue(event["observer_no_mutation"])
        self.assertFalse(summary.fields["active_topk_applied"])

    def test_candidate_contract_rejects_length_mismatch_without_mutation(self):
        observer, _, camera, candidates, current_map = make_fixture()
        invalid = list(candidates)
        invalid[1] = invalid[1][:-1]
        token = observe(observer, camera, tuple(invalid), current_map)
        self.assertEqual(token.fields["status"], "error")
        self.assertEqual(
            token.fields["reason"], "candidate_contract_error"
        )

    def test_hook_order_and_off_path_are_static_and_no_extra_render(self):
        root = Path(__file__).resolve().parents[1]
        model_source = (
            root / "src/gaussian_splatting/scene/gaussian_model.py"
        ).read_text(encoding="utf-8")
        mapper_source = (root / "src/gaussian_mapping.py").read_text(
            encoding="utf-8"
        )
        module_source = (
            root
            / "src/candidate_selection/gaussian_candidate_selector_v1.py"
        ).read_text(encoding="utf-8")
        model_function = model_source[model_source.index("def extend_from_pcd_seq") :]
        self.assertLess(
            model_function.index("candidate_observer.observe_before_extend"),
            model_function.index("candidate_selector_v1.observe_before_extend"),
        )
        self.assertLess(
            model_function.index("candidate_selector_v1.observe_before_extend"),
            model_function.index("# OURS-M01"),
        )
        self.assertIn('cfg.mapping.get("candidate_selector_v1", None)', mapper_source)
        self.assertNotIn("render(", module_source)
        self.assertNotIn("selected_indices =", module_source)


if __name__ == "__main__":
    unittest.main()
