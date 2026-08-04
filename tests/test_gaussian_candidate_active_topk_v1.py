from contextlib import redirect_stdout
import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

import src.candidate_selection.gaussian_candidate_active_topk_v1 as active_module

from src.candidate_selection.gaussian_candidate_active_topk_v1 import (
    ACTIVE_EVENT_FIELDS,
    ACTIVE_BUDGET,
    ACTIVE_MODE,
    ActiveConfidenceEvidenceError,
    CandidateActiveSelectionResult,
    GaussianCandidateActiveTopKV1,
    LOG_PREFIX,
    stable_quality_topk_indices,
)
from src.candidate_selection.gaussian_candidate_selector_v1 import (
    GaussianCandidateSelectorV1,
    build_gaussian_candidate_selector_v1,
)
from src.resource_management.m01_resource_admission.resource_admission import (
    ResourceAdmission,
    deterministic_uniform_indices,
)


class ActiveCamera:
    def __init__(self, depth: torch.Tensor):
        self.uid = 17
        self.buffer_index = 5
        self.source_frame_id = 7001
        self.source_timestamp = 44.25
        self.fx = 1.0
        self.fy = 1.0
        self.cx = 0.0
        self.cy = 0.0
        self.image_height = int(depth.shape[0])
        self.image_width = int(depth.shape[1])
        self.depth = depth
        self.depth_prior = depth.clone()
        self._pose = torch.eye(4, dtype=torch.float32)

    @property
    def pose(self):
        return self._pose


class ActiveSnapshotGetter:
    def __init__(self, confidence: torch.Tensor):
        self.confidence = confidence
        self.calls = 0
        self.overrides = {}

    def __call__(self, camera, require_current=True, *, upsampled=None):
        self.calls += 1
        values = {
            "buffer_index": camera.buffer_index,
            "source_frame_id": camera.source_frame_id,
            "source_timestamp": camera.source_timestamp,
            "confidence_source_frame_id": camera.source_frame_id,
            "confidence_version": 9,
            "confidence_up_version": 3,
            "is_current": True,
            "is_stale": False,
            "confidence": self.confidence.clone(),
            "shape": tuple(self.confidence.shape),
            "dtype": str(self.confidence.dtype),
            "device": str(self.confidence.device),
            "requires_grad": bool(self.confidence.requires_grad),
        }
        values.update(self.overrides)
        return SimpleNamespace(**values)


def make_candidates(count: int, *, dtype=torch.float32):
    indices = torch.arange(count, dtype=dtype)
    xyz = torch.stack(
        (indices, torch.zeros_like(indices), torch.ones_like(indices)),
        dim=1,
    )
    features = torch.arange(
        count * 3,
        dtype=dtype,
    ).reshape(count, 3, 1)
    scales = torch.arange(count, dtype=dtype).reshape(count, 1) + 1.0
    rotations = torch.zeros(count, 4, dtype=dtype)
    if count > 0:
        rotations[:, 0] = 1.0
    opacities = torch.arange(count, dtype=dtype).reshape(count, 1)
    return xyz, features, scales, rotations, opacities


def make_active_fixture(count: int, scores: torch.Tensor | None = None):
    depth = torch.ones(1, max(count, 1), dtype=torch.float32)
    camera = ActiveCamera(depth)
    if scores is None:
        scores = torch.arange(max(count, 1), dtype=torch.float32).reshape(1, -1)
    elif scores.ndim == 1:
        scores = scores.reshape(1, -1)
    getter = ActiveSnapshotGetter(scores)
    selector = GaussianCandidateActiveTopKV1(
        confidence_snapshot_getter=getter,
        logging_enabled=True,
        device="cpu",
        active_budget=ACTIVE_BUDGET,
    )
    return selector, getter, camera, make_candidates(count)


def select_active(selector, camera, candidates, *, init=False):
    return selector.select_before_extend(
        xyz=candidates[0],
        features=candidates[1],
        scales=candidates[2],
        rotations=candidates[3],
        opacities=candidates[4],
        camera=camera,
        depthmap=None,
        depth_source="estimated_clean_depth",
        mapper_update_id=23,
        init=init,
        gaussian_before=11,
    )


def load_gaussian_model_extend_from_pcd_seq():
    """Compile and execute the real GaussianModel method without CUDA imports."""

    root = Path(__file__).resolve().parents[1]
    source = (
        root / "src/gaussian_splatting/scene/gaussian_model.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GaussianModel"
    )
    function_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "extend_from_pcd_seq"
    )
    function_node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function_node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"GaussianCandidateActiveTopKV1": GaussianCandidateActiveTopKV1}
    exec(compile(module, str(root / "src/gaussian_splatting/scene/gaussian_model.py"), "exec"), namespace)
    return namespace["extend_from_pcd_seq"]


class GaussianModelHookHarness:
    def __init__(self, candidates):
        self.candidates = candidates
        self.resource_admission = None
        self.extend_calls = 0
        self.m01_calls = 0
        self.last_extended_count = None
        self._gaussian_count = 11

    def __len__(self):
        return self._gaussian_count

    @property
    def get_xyz(self):
        return torch.empty(self._gaussian_count, 3)

    def create_pcd_from_image(self, *args, collect_candidate_metadata=False, **kwargs):
        if collect_candidate_metadata:
            return SimpleNamespace(
                candidates=self.candidates,
                metadata=SimpleNamespace(depth_source="estimated_clean_depth"),
            )
        return self.candidates

    def extend_from_pcd(
        self,
        xyz,
        features,
        scales,
        rotations,
        opacities,
        kf_id,
    ):
        self.extend_calls += 1
        self.last_extended_count = int(xyz.shape[0])
        self._gaussian_count += int(xyz.shape[0])


class DeterministicUniformIndexTests(unittest.TestCase):
    def test_empty_returns_empty_long_cpu(self):
        result = deterministic_uniform_indices(
            candidate_count=0, fixed_budget=600, device=torch.device("cpu")
        )
        self.assertEqual(result.tolist(), [])
        self.assertEqual(result.dtype, torch.long)
        self.assertEqual(result.device.type, "cpu")

    def test_n_less_than_budget_retains_every_index(self):
        result = deterministic_uniform_indices(
            candidate_count=4, fixed_budget=6, device=torch.device("cpu")
        )
        self.assertEqual(result.tolist(), [0, 1, 2, 3])

    def test_n_equal_budget_retains_every_index(self):
        result = deterministic_uniform_indices(
            candidate_count=4, fixed_budget=4, device=torch.device("cpu")
        )
        self.assertEqual(result.tolist(), [0, 1, 2, 3])

    def test_over_budget_matches_frozen_formula(self):
        result = deterministic_uniform_indices(
            candidate_count=10, fixed_budget=4, device=torch.device("cpu")
        )
        expected = torch.div(
            torch.arange(4) * 9,
            3,
            rounding_mode="floor",
        )
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.tolist(), [0, 3, 6, 9])

    def test_budget_one_matches_existing_middle_rule(self):
        result = deterministic_uniform_indices(
            candidate_count=10, fixed_budget=1, device=torch.device("cpu")
        )
        self.assertEqual(result.tolist(), [4])

    def test_repeat_is_exact_and_rng_free(self):
        before = torch.get_rng_state().clone()
        first = deterministic_uniform_indices(
            candidate_count=997, fixed_budget=600, device=torch.device("cpu")
        )
        second = deterministic_uniform_indices(
            candidate_count=997, fixed_budget=600, device=torch.device("cpu")
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_invalid_count_and_budget_fail_closed(self):
        for count in (True, -1, 1.5):
            with self.assertRaises((TypeError, ValueError)):
                deterministic_uniform_indices(
                    candidate_count=count,
                    fixed_budget=1,
                    device=torch.device("cpu"),
                )
        for budget in (True, 0, 1.5):
            with self.assertRaises((TypeError, ValueError)):
                deterministic_uniform_indices(
                    candidate_count=3,
                    fixed_budget=budget,
                    device=torch.device("cpu"),
                )

    def test_resource_admission_uses_shared_exact_indices(self):
        candidates = make_candidates(10)
        clones = tuple(value.clone() for value in candidates)
        pointers = tuple(value.data_ptr() for value in candidates)
        versions = tuple(value._version for value in candidates)
        admission = ResourceAdmission(
            mode="fixed_budget",
            fixed_budget=4,
            selection="deterministic_uniform",
        )
        result = admission.admit_before_extend(
            xyz=candidates[0],
            features=candidates[1],
            scales=candidates[2],
            rotations=candidates[3],
            opacities=candidates[4],
            camera_uid=1,
            kf_id=1,
            init=False,
            gaussian_before=0,
        )
        expected = deterministic_uniform_indices(
            candidate_count=10, fixed_budget=4, device=torch.device("cpu")
        )
        self.assertTrue(torch.equal(result.selected_indices, expected))
        outputs = (
            result.xyz,
            result.features,
            result.scales,
            result.rotations,
            result.opacities,
        )
        for index, (original, clone, selected) in enumerate(
            zip(candidates, clones, outputs)
        ):
            self.assertTrue(torch.equal(selected, original.index_select(0, expected)))
            self.assertEqual(selected.dtype, original.dtype)
            self.assertEqual(selected.device, original.device)
            self.assertNotEqual(selected.data_ptr(), original.data_ptr())
            self.assertTrue(torch.equal(original, clone))
            self.assertEqual(original.data_ptr(), pointers[index])
            self.assertEqual(original._version, versions[index])


class StableQualityTopKTests(unittest.TestCase):
    def test_empty_n_less_and_n_equal_k(self):
        for values, k in (([], 3), ([0.2, 0.1], 3), ([0.2, 0.1], 2)):
            confidence = torch.tensor(values, dtype=torch.float32)
            result = stable_quality_topk_indices(
                confidence,
                candidate_count=len(values),
                requested_k=k,
                device=torch.device("cpu"),
            )
            self.assertEqual(result.tolist(), list(range(len(values))))

    def test_higher_score_wins_and_output_restores_source_order(self):
        confidence = torch.tensor([0.1, 4.0, 3.0, 5.0, 2.0])
        result = stable_quality_topk_indices(
            confidence,
            candidate_count=5,
            requested_k=3,
            device=torch.device("cpu"),
        )
        self.assertEqual(result.tolist(), [1, 2, 3])

    def test_all_equal_prefers_lower_original_indices(self):
        confidence = torch.ones(8)
        result = stable_quality_topk_indices(
            confidence,
            candidate_count=8,
            requested_k=3,
            device=torch.device("cpu"),
        )
        self.assertEqual(result.tolist(), [0, 1, 2])

    def test_partial_ties_are_stable(self):
        confidence = torch.tensor([3.0, 2.0, 3.0, 2.0, 3.0])
        result = stable_quality_topk_indices(
            confidence,
            candidate_count=5,
            requested_k=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(result.tolist(), [0, 2])

    def test_negative_positive_and_above_one_are_ordered(self):
        confidence = torch.tensor([-2.0, 1.2, 0.0, 1.4, -0.1])
        result = stable_quality_topk_indices(
            confidence,
            candidate_count=5,
            requested_k=3,
            device=torch.device("cpu"),
        )
        self.assertEqual(result.tolist(), [1, 2, 3])

    def test_repeated_selection_and_input_are_unchanged(self):
        confidence = torch.tensor([0.1, 0.9, 0.7, 0.8], requires_grad=True)
        clone = confidence.detach().clone()
        pointer = confidence.data_ptr()
        version = confidence._version
        first = stable_quality_topk_indices(
            confidence,
            candidate_count=4,
            requested_k=2,
            device=torch.device("cpu"),
        )
        second = stable_quality_topk_indices(
            confidence,
            candidate_count=4,
            requested_k=2,
            device=torch.device("cpu"),
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(confidence.detach(), clone))
        self.assertEqual(confidence.data_ptr(), pointer)
        self.assertEqual(confidence._version, version)
        self.assertFalse(first.requires_grad)

    def test_nonfinite_shape_length_dtype_and_device_fail(self):
        invalid = (
            torch.tensor([0.0, torch.nan]),
            torch.tensor([0.0, torch.inf]),
            torch.tensor([0.0, -torch.inf]),
        )
        for confidence in invalid:
            with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                stable_quality_topk_indices(
                    confidence,
                    candidate_count=2,
                    requested_k=1,
                    device=torch.device("cpu"),
                )
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            stable_quality_topk_indices(
                torch.zeros(1, 2),
                candidate_count=2,
                requested_k=1,
                device=torch.device("cpu"),
            )
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            stable_quality_topk_indices(
                torch.zeros(3),
                candidate_count=2,
                requested_k=1,
                device=torch.device("cpu"),
            )
        with self.assertRaisesRegex(ValueError, "dtype mismatch"):
            stable_quality_topk_indices(
                torch.zeros(2, dtype=torch.long),
                candidate_count=2,
                requested_k=1,
                device=torch.device("cpu"),
            )
        with self.assertRaisesRegex(ValueError, "device mismatch"):
            stable_quality_topk_indices(
                torch.empty(2, device="meta"),
                candidate_count=2,
                requested_k=1,
                device=torch.device("cpu"),
            )


class ActiveSelectorIntegrationTests(unittest.TestCase):
    def test_factory_active_is_isolated_and_budget_is_frozen(self):
        getter = ActiveSnapshotGetter(torch.ones(1, 1))
        selector = build_gaussian_candidate_selector_v1(
            {"mode": "active", "budget": 600, "logging": {"enabled": True}},
            candidate_observer=None,
            resource_admission_mode="disabled",
            confidence_snapshot_getter=getter,
            device="cpu",
        )
        self.assertTrue(selector.is_active)
        self.assertEqual(selector.active_budget, 600)
        for budget in (True, 599, 601):
            with self.assertRaises((TypeError, ValueError)):
                build_gaussian_candidate_selector_v1(
                    {"mode": "active", "budget": budget},
                    candidate_observer=None,
                    resource_admission_mode="disabled",
                    confidence_snapshot_getter=getter,
                    device="cpu",
                )

    def test_factory_active_rejects_gco_and_every_m01_instance(self):
        getter = ActiveSnapshotGetter(torch.ones(1, 1))
        with self.assertRaisesRegex(ValueError, "candidate_observer.mode=off"):
            build_gaussian_candidate_selector_v1(
                {"mode": "active"},
                candidate_observer=SimpleNamespace(voxel_size=0.05),
                resource_admission_mode="disabled",
                confidence_snapshot_getter=getter,
                device="cpu",
            )
        for mode in ("observe", "fixed_budget"):
            with self.assertRaisesRegex(ValueError, "requires M01"):
                build_gaussian_candidate_selector_v1(
                    {"mode": "active"},
                    candidate_observer=None,
                    resource_admission_mode=mode,
                    confidence_snapshot_getter=getter,
                    device="cpu",
                )

    def test_active_and_observe_public_apis_are_isolated(self):
        active, _, camera, candidates = make_active_fixture(4)
        self.assertFalse(hasattr(active, "observe_before_extend"))
        observe_selector = GaussianCandidateSelectorV1(
            voxel_size=0.05,
            confidence_snapshot_getter=ActiveSnapshotGetter(torch.ones(1, 4)),
            logging_enabled=True,
            device="cpu",
        )
        self.assertFalse(hasattr(observe_selector, "select_before_extend"))

    def test_empty_candidates_bypass_confidence(self):
        selector, getter, camera, candidates = make_active_fixture(0)
        result = select_active(selector, camera, candidates)
        self.assertIsInstance(result, CandidateActiveSelectionResult)
        self.assertEqual(result.retained_candidate_count, 0)
        self.assertIsNone(result.selected_indices)
        self.assertEqual(getter.calls, 0)
        self.assertEqual(result.token.fields["selector"], "keep_all_n_le_k")

    def test_init_over_budget_keeps_all_and_never_reads_confidence(self):
        selector, getter, camera, candidates = make_active_fixture(603)
        result = select_active(selector, camera, candidates, init=True)
        self.assertEqual(result.retained_candidate_count, 603)
        self.assertIsNone(result.selected_indices)
        self.assertEqual(getter.calls, 0)
        self.assertTrue(result.token.fields["init_bypass"])
        self.assertEqual(result.token.fields["selector"], "init_bypass")

    def test_n_less_and_equal_budget_keep_original_objects(self):
        for count in (4, 600):
            selector, getter, camera, candidates = make_active_fixture(count)
            result = select_active(selector, camera, candidates)
            outputs = (
                result.xyz,
                result.features,
                result.scales,
                result.rotations,
                result.opacities,
            )
            self.assertEqual(getter.calls, 0)
            self.assertIsNone(result.selected_indices)
            for original, output in zip(candidates, outputs):
                self.assertIs(original, output)

    def test_quality_topk_selects_exactly_600_and_preserves_source_order(self):
        count = 603
        scores = torch.arange(count, dtype=torch.float32)
        selector, getter, camera, candidates = make_active_fixture(count, scores)
        result = select_active(selector, camera, candidates)
        expected = torch.arange(3, count)
        self.assertTrue(torch.equal(result.selected_indices, expected))
        self.assertEqual(result.retained_candidate_count, 600)
        self.assertEqual(getter.calls, 1)
        self.assertEqual(result.token.fields["selector"], "quality_topk")
        self.assertFalse(result.token.fields["fallback"])
        self.assertEqual(result.token.fields["evidence_frame_id"], 7001)
        self.assertEqual(result.token.fields["confidence_version"], 9)

    def test_all_equal_selects_first_600_deterministically(self):
        selector, _, camera, candidates = make_active_fixture(603, torch.ones(603))
        first = select_active(selector, camera, candidates)
        second = select_active(selector, camera, candidates)
        expected = torch.arange(600)
        self.assertTrue(torch.equal(first.selected_indices, expected))
        self.assertTrue(torch.equal(second.selected_indices, expected))
        self.assertEqual(
            first.token.fields["selected_indices_fingerprint"],
            second.token.fields["selected_indices_fingerprint"],
        )

    def test_five_fields_use_one_index_and_preserve_dtype_content(self):
        selector, _, camera, candidates = make_active_fixture(603)
        clones = tuple(value.clone() for value in candidates)
        pointers = tuple(value.data_ptr() for value in candidates)
        versions = tuple(value._version for value in candidates)
        result = select_active(selector, camera, candidates)
        outputs = (
            result.xyz,
            result.features,
            result.scales,
            result.rotations,
            result.opacities,
        )
        for index, (original, clone, output) in enumerate(
            zip(candidates, clones, outputs)
        ):
            self.assertTrue(
                torch.equal(output, original.index_select(0, result.selected_indices))
            )
            self.assertEqual(output.dtype, original.dtype)
            self.assertEqual(output.device, original.device)
            self.assertTrue(torch.equal(original, clone))
            self.assertEqual(original.data_ptr(), pointers[index])
            self.assertEqual(original._version, versions[index])

    def test_requires_grad_semantics_follow_index_select_without_input_mutation(self):
        selector, _, camera, candidates = make_active_fixture(603)
        candidates = tuple(value.requires_grad_() for value in candidates)
        versions = tuple(value._version for value in candidates)
        result = select_active(selector, camera, candidates)
        for index, output in enumerate((
            result.xyz,
            result.features,
            result.scales,
            result.rotations,
            result.opacities,
        )):
            self.assertTrue(output.requires_grad)
            self.assertEqual(candidates[index]._version, versions[index])

    def _assert_fixed_fallback(self, selector, camera, candidates, code):
        result = select_active(selector, camera, candidates)
        expected = deterministic_uniform_indices(
            candidate_count=603,
            fixed_budget=600,
            device=torch.device("cpu"),
        )
        self.assertTrue(torch.equal(result.selected_indices, expected))
        self.assertTrue(result.token.fields["fallback"])
        self.assertEqual(result.token.fields["selector"], "fixed600_fallback")
        self.assertEqual(result.token.fields["fallback_reason"]["code"], code)
        self.assertIsNone(result.token.fields["confidence_min"])
        self.assertIsNone(result.token.fields["confidence_max"])

    def test_missing_confidence_falls_back_exactly(self):
        selector, _, camera, candidates = make_active_fixture(603)

        def missing(*args, **kwargs):
            raise RuntimeError("confidence is not valid")

        selector._confidence_snapshot_getter = missing
        self._assert_fixed_fallback(
            selector, camera, candidates, "missing_confidence"
        )

    def test_unknown_internal_error_is_not_absorbed_by_fallback(self):
        selector, _, camera, candidates = make_active_fixture(603)
        clones = tuple(value.clone() for value in candidates)

        def programmer_bug(*args, **kwargs):
            raise RuntimeError("unexpected active implementation defect")

        selector._confidence_snapshot_getter = programmer_bug
        with mock.patch(
            "src.candidate_selection.gaussian_candidate_active_topk_v1."
            "deterministic_uniform_indices",
            wraps=deterministic_uniform_indices,
        ) as fallback:
            with self.assertRaisesRegex(RuntimeError, "implementation defect"):
                select_active(selector, camera, candidates)
        self.assertEqual(fallback.call_count, 0)
        for original, clone in zip(candidates, clones):
            self.assertTrue(torch.equal(original, clone))

    def test_fallback_record_reports_no_m01_second_selection(self):
        selector, _, camera, candidates = make_active_fixture(603)

        def missing(*args, **kwargs):
            raise RuntimeError("confidence is not valid")

        selector._confidence_snapshot_getter = missing
        result = select_active(selector, camera, candidates)
        with redirect_stdout(io.StringIO()):
            summary = selector.record_active_after_extend(
                result.token,
                admitted_candidate_count=600,
                dropped_candidate_count=0,
                gaussian_after_extend=611,
                m01_second_selection_applied=False,
            )
        self.assertTrue(summary.fields["fallback"])
        self.assertFalse(summary.fields["m01_second_selection_applied"])
        self.assertTrue(summary.fields["conservation_check"])

    def test_stale_confidence_falls_back_exactly(self):
        selector, getter, camera, candidates = make_active_fixture(603)
        getter.overrides.update({"is_current": False, "is_stale": True})
        self._assert_fixed_fallback(
            selector, camera, candidates, "stale_confidence"
        )

    def test_source_identity_and_timestamp_mismatch_fall_back(self):
        cases = (
            ({"confidence_source_frame_id": 999}, "confidence_identity_mismatch"),
            ({"source_timestamp": 45.0}, "confidence_timestamp_mismatch"),
        )
        for overrides, reason in cases:
            selector, getter, camera, candidates = make_active_fixture(603)
            getter.overrides.update(overrides)
            self._assert_fixed_fallback(selector, camera, candidates, reason)

    def test_short_and_long_confidence_vectors_fall_back(self):
        for length in (602, 604):
            selector, _, camera, candidates = make_active_fixture(603)
            selector._read_active_confidence = lambda **kwargs: (
                torch.zeros(length),
                {},
            )
            self._assert_fixed_fallback(
                selector,
                camera,
                candidates,
                "confidence_length_mismatch",
            )

    def test_nan_positive_inf_and_negative_inf_fall_back(self):
        for value in (torch.nan, torch.inf, -torch.inf):
            scores = torch.arange(603, dtype=torch.float32)
            scores[7] = value
            selector, _, camera, candidates = make_active_fixture(603, scores)
            self._assert_fixed_fallback(
                selector, camera, candidates, "nonfinite_confidence"
            )

    def test_confidence_shape_dtype_and_device_failures_fall_back(self):
        selector, getter, camera, candidates = make_active_fixture(603)
        getter.confidence = torch.zeros(1, 1, 603)
        self._assert_fixed_fallback(
            selector, camera, candidates, "confidence_shape_mismatch"
        )

        selector, getter, camera, candidates = make_active_fixture(603)
        getter.confidence = torch.ones(1, 603, dtype=torch.long)
        self._assert_fixed_fallback(
            selector, camera, candidates, "confidence_dtype_mismatch"
        )

        selector, getter, camera, candidates = make_active_fixture(603)
        getter.confidence = torch.empty(1, 603, device="meta")
        self._assert_fixed_fallback(
            selector, camera, candidates, "confidence_device_mismatch"
        )

    def test_candidate_length_and_shape_contract_fail_without_fallback(self):
        selector, _, camera, candidates = make_active_fixture(603)
        invalid = list(candidates)
        invalid[1] = invalid[1][:-1]
        with self.assertRaisesRegex(ValueError, "first-dimension mismatch"):
            select_active(selector, camera, tuple(invalid))

        invalid = list(candidates)
        invalid[3] = torch.zeros(603, 3)
        with self.assertRaisesRegex(ValueError, "field shape contract"):
            select_active(selector, camera, tuple(invalid))

    def test_candidate_device_contract_fails_without_fallback(self):
        selector, _, camera, candidates = make_active_fixture(603)
        invalid = list(candidates)
        invalid[4] = torch.empty(603, 1, device="meta")
        with self.assertRaisesRegex(ValueError, "device mismatch"):
            select_active(selector, camera, tuple(invalid))

    def _assert_complete_event_contract(self, event, scenario):
        self.assertEqual(frozenset(event), ACTIVE_EVENT_FIELDS)
        for field in (
            "schema",
            "event_sequence",
            "mapper_update_id",
            "source_camera_id",
            "source_frame_id",
            "buffer_index",
            "input_candidate_count",
            "retained_candidate_count",
            "requested_k",
            "effective_k",
            "selected_indices_count",
            "gaussian_before",
            "gaussian_after_extend",
            "actual_admitted_count",
            "actual_dropped_count",
        ):
            self.assertIs(type(event[field]), int, field)
        for field in (
            "init_bypass",
            "fallback",
            "selected_indices_created",
            "conservation_check",
            "m01_second_selection_applied",
            "diagnostic_gpu_to_cpu_sync",
        ):
            self.assertIs(type(event[field]), bool, field)
        for field in (
            "event_type",
            "event_id",
            "status",
            "reason",
            "mode",
            "confidence_status",
            "selector",
        ):
            self.assertIs(type(event[field]), str, field)
        self.assertIs(type(event["source_timestamp"]), float)
        self.assertIs(type(event["selection_wall_ms"]), float)
        self.assertEqual(event["schema"], 1)
        self.assertEqual(event["event_type"], "candidate_active_selection")
        self.assertEqual(event["status"], "ok")
        self.assertEqual(event["mode"], "active")
        self.assertTrue(event["conservation_check"])
        self.assertFalse(event["m01_second_selection_applied"])
        self.assertIsNone(event["error"])

        if scenario == "quality":
            for field in (
                "evidence_frame_id",
                "confidence_version",
                "confidence_source_frame_id",
            ):
                self.assertIs(type(event[field]), int, field)
            for field in ("confidence_min", "confidence_max", "scale_x", "scale_y"):
                self.assertIs(type(event[field]), float, field)
            for field in ("source_resolution", "confidence_resolution"):
                self.assertIs(type(event[field]), list, field)
                self.assertEqual(len(event[field]), 2)
                self.assertTrue(all(type(value) is int for value in event[field]))
            self.assertIs(type(event["confidence_sampling_method"]), str)
            self.assertIs(type(event["selected_indices_fingerprint"]), str)
            self.assertIsNone(event["fallback_reason"])
            self.assertTrue(event["selected_indices_created"])
            self.assertFalse(event["fallback"])
        elif scenario == "fallback":
            self.assertIs(type(event["fallback_reason"]), dict)
            self.assertEqual(
                set(event["fallback_reason"]), {"code", "error"}
            )
            self.assertIs(type(event["fallback_reason"]["code"]), str)
            self.assertIs(type(event["fallback_reason"]["error"]), dict)
            self.assertTrue(event["selected_indices_created"])
            self.assertTrue(event["fallback"])
            for field in (
                "evidence_frame_id",
                "confidence_version",
                "confidence_source_frame_id",
                "confidence_sampling_method",
                "source_resolution",
                "confidence_resolution",
                "scale_x",
                "scale_y",
                "confidence_min",
                "confidence_max",
            ):
                self.assertIsNone(event[field], field)
        else:
            self.assertFalse(event["selected_indices_created"])
            self.assertFalse(event["fallback"])
            self.assertEqual(event["selected_indices_count"], 0)
            for field in (
                "evidence_frame_id",
                "confidence_version",
                "confidence_source_frame_id",
                "confidence_sampling_method",
                "source_resolution",
                "confidence_resolution",
                "scale_x",
                "scale_y",
                "confidence_min",
                "confidence_max",
                "selected_indices_fingerprint",
                "fallback_reason",
            ):
                self.assertIsNone(event[field], field)

    def test_complete_event_contract_for_all_active_states(self):
        scenarios = []

        selector, _, camera, candidates = make_active_fixture(603)
        result = select_active(selector, camera, candidates)
        scenarios.append(("quality", selector, result.token, 600, 611))

        selector, _, camera, candidates = make_active_fixture(603)
        selector._confidence_snapshot_getter = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("confidence is not valid")
        )
        result = select_active(selector, camera, candidates)
        scenarios.append(("fallback", selector, result.token, 600, 611))

        selector, _, camera, candidates = make_active_fixture(603)
        result = select_active(selector, camera, candidates, init=True)
        scenarios.append(("init", selector, result.token, 603, 614))

        selector, _, camera, candidates = make_active_fixture(4)
        result = select_active(selector, camera, candidates)
        scenarios.append(("keep_all", selector, result.token, 4, 15))

        selector, _, camera, _ = make_active_fixture(0)
        token = selector.observe_active_empty(
            camera=camera,
            mapper_update_id=23,
            init=False,
            gaussian_before=11,
        )
        scenarios.append(("empty", selector, token, 0, 11))

        for scenario, selector, token, admitted, gaussian_after in scenarios:
            with self.subTest(scenario=scenario), redirect_stdout(io.StringIO()):
                summary = selector.record_active_after_extend(
                    token,
                    admitted_candidate_count=admitted,
                    dropped_candidate_count=0,
                    gaussian_after_extend=gaussian_after,
                    m01_second_selection_applied=False,
                )
                self._assert_complete_event_contract(summary.fields, scenario)

    def test_active_log_contract_and_post_extend_conservation(self):
        selector, _, camera, candidates = make_active_fixture(603)
        result = select_active(selector, camera, candidates)
        stream = io.StringIO()
        with redirect_stdout(stream):
            summary = selector.record_active_after_extend(
                result.token,
                admitted_candidate_count=600,
                dropped_candidate_count=0,
                gaussian_after_extend=611,
                m01_second_selection_applied=False,
            )
        event = json.loads(stream.getvalue().split(" ", 1)[1])
        self.assertTrue(stream.getvalue().startswith(LOG_PREFIX + " "))
        self.assertEqual(event["event_type"], "candidate_active_selection")
        self.assertEqual(event["mode"], "active")
        self.assertEqual(event["retained_candidate_count"], 600)
        self.assertEqual(event["actual_dropped_count"], 3)
        self.assertEqual(event["selected_indices_count"], 600)
        self.assertTrue(event["conservation_check"])
        self.assertFalse(event["m01_second_selection_applied"])
        self.assertEqual(summary.fields["status"], "ok")

    def test_second_m01_selection_is_detected_as_error(self):
        selector, _, camera, candidates = make_active_fixture(603)
        result = select_active(selector, camera, candidates)
        with redirect_stdout(io.StringIO()):
            summary = selector.record_active_after_extend(
                result.token,
                admitted_candidate_count=599,
                dropped_candidate_count=1,
                gaussian_after_extend=610,
                m01_second_selection_applied=True,
            )
        self.assertEqual(summary.fields["status"], "error")
        self.assertTrue(summary.fields["m01_second_selection_applied"])
        self.assertFalse(summary.fields["conservation_check"])

    def test_empty_point_cloud_has_independent_active_event(self):
        selector, getter, camera, _ = make_active_fixture(0)
        token = selector.observe_active_empty(
            camera=camera,
            mapper_update_id=23,
            init=False,
            gaussian_before=11,
        )
        self.assertEqual(getter.calls, 0)
        self.assertEqual(token.fields["selector"], "empty_candidate_bypass")
        with redirect_stdout(io.StringIO()):
            summary = selector.record_active_after_extend(
                token,
                admitted_candidate_count=0,
                dropped_candidate_count=0,
                gaussian_after_extend=11,
                m01_second_selection_applied=False,
            )
        self.assertTrue(summary.fields["conservation_check"])

    def test_static_hook_is_before_m01_and_runtime_rejects_second_selector(self):
        root = Path(__file__).resolve().parents[1]
        source = (
            root / "src/gaussian_splatting/scene/gaussian_model.py"
        ).read_text(encoding="utf-8")
        function = source[source.index("def extend_from_pcd_seq") :]
        self.assertLess(
            function.index("candidate_selector_v1.select_before_extend"),
            function.index("# OURS-M01"),
        )
        self.assertIn(
            "active_selection is not None and self.resource_admission is not None",
            function,
        )

    def test_default_yaml_is_off_and_no_active_topk_alias_exists(self):
        root = Path(__file__).resolve().parents[1]
        config = (root / "configs/mapping/base.yaml").read_text(encoding="utf-8")
        block = config[config.index("candidate_selector_v1:") :]
        self.assertLess(block.index('mode: "off"'), block.index("budget: 600"))
        self.assertNotIn("active_topk600", block.split("camera_scheduler:", 1)[0])


class GaussianModelActiveHookSpyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extend_from_pcd_seq = staticmethod(
            load_gaussian_model_extend_from_pcd_seq()
        )

    def _run_hook(self, harness, selector, camera, *, init=False):
        with redirect_stdout(io.StringIO()):
            self.extend_from_pcd_seq(
                harness,
                camera,
                camera.uid,
                init=init,
                candidate_selector_v1=selector,
                mapper_update_id=23,
            )

    def test_gaussian_model_quality_path_selects_once_and_extends_once(self):
        selector, getter, camera, candidates = make_active_fixture(603)
        harness = GaussianModelHookHarness(candidates)
        with mock.patch.object(
            active_module,
            "stable_quality_topk_indices",
            wraps=stable_quality_topk_indices,
        ) as quality, mock.patch.object(
            active_module,
            "deterministic_uniform_indices",
            wraps=deterministic_uniform_indices,
        ) as fallback:
            self._run_hook(harness, selector, camera)
        self.assertEqual(quality.call_count, 1)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(getter.calls, 1)
        self.assertEqual(harness.m01_calls, 0)
        self.assertEqual(harness.extend_calls, 1)
        self.assertEqual(harness.last_extended_count, 600)

    def test_gaussian_model_fallback_selects_once_and_extends_once(self):
        selector, getter, camera, candidates = make_active_fixture(603)
        harness = GaussianModelHookHarness(candidates)

        def missing(*args, **kwargs):
            getter.calls += 1
            raise RuntimeError("confidence is not valid")

        selector._confidence_snapshot_getter = missing
        with mock.patch.object(
            active_module,
            "stable_quality_topk_indices",
            wraps=stable_quality_topk_indices,
        ) as quality, mock.patch.object(
            active_module,
            "deterministic_uniform_indices",
            wraps=deterministic_uniform_indices,
        ) as fallback:
            self._run_hook(harness, selector, camera)
        self.assertEqual(quality.call_count, 0)
        self.assertEqual(fallback.call_count, 1)
        self.assertEqual(getter.calls, 1)
        self.assertEqual(harness.m01_calls, 0)
        self.assertEqual(harness.extend_calls, 1)
        self.assertEqual(harness.last_extended_count, 600)

    def test_gaussian_model_init_bypasses_confidence_and_selection(self):
        selector, getter, camera, candidates = make_active_fixture(603)
        harness = GaussianModelHookHarness(candidates)
        with mock.patch.object(
            active_module,
            "stable_quality_topk_indices",
            wraps=stable_quality_topk_indices,
        ) as quality, mock.patch.object(
            active_module,
            "deterministic_uniform_indices",
            wraps=deterministic_uniform_indices,
        ) as fallback:
            self._run_hook(harness, selector, camera, init=True)
        self.assertEqual(quality.call_count, 0)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(getter.calls, 0)
        self.assertEqual(harness.m01_calls, 0)
        self.assertEqual(harness.extend_calls, 1)
        self.assertEqual(harness.last_extended_count, 603)

    def test_gaussian_model_empty_bypasses_confidence_and_extend(self):
        selector, getter, camera, _ = make_active_fixture(0)
        harness = GaussianModelHookHarness(None)
        with mock.patch.object(
            active_module,
            "stable_quality_topk_indices",
            wraps=stable_quality_topk_indices,
        ) as quality, mock.patch.object(
            active_module,
            "deterministic_uniform_indices",
            wraps=deterministic_uniform_indices,
        ) as fallback:
            self._run_hook(harness, selector, camera)
        self.assertEqual(quality.call_count, 0)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(getter.calls, 0)
        self.assertEqual(harness.m01_calls, 0)
        self.assertEqual(harness.extend_calls, 0)


if __name__ == "__main__":
    unittest.main()
