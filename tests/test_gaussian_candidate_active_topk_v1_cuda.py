"""CUDA-only correctness gate for GCS-v1 Active Top-k600.

This module intentionally fails instead of skipping when CUDA is unavailable.
The server preflight must reject an unsuitable environment before invoking it.
It uses synthetic tensors only and never starts DROID-Splat or reads a dataset.
"""

from contextlib import redirect_stdout
import hashlib
import io
import json
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

import src.candidate_selection.gaussian_candidate_active_topk_v1 as active_module
from src.candidate_selection.gaussian_candidate_active_topk_v1 import (
    ACTIVE_BUDGET,
    GaussianCandidateActiveTopKV1,
    stable_quality_topk_indices,
)
from src.candidate_selection.gaussian_candidate_selector_v1 import (
    build_gaussian_candidate_selector_v1,
)
from src.resource_management.m01_resource_admission.resource_admission import (
    deterministic_uniform_indices,
)
from tests.test_gaussian_candidate_active_topk_v1 import (
    ActiveCamera,
    ActiveSnapshotGetter,
    GaussianModelHookHarness,
    load_gaussian_model_extend_from_pcd_seq,
    make_candidates,
)


CUDA_DEVICE = torch.device("cuda:0")


def setUpModule() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA correctness gate requires torch.cuda.is_available() == True; "
            "refusing to skip."
        )
    if torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA correctness gate requires at least one GPU.")
    torch.cuda.set_device(CUDA_DEVICE)
    probe = torch.ones(1, device=CUDA_DEVICE)
    if probe.device != CUDA_DEVICE:
        raise RuntimeError("CUDA probe tensor was not created on cuda:0.")
    torch.cuda.synchronize(CUDA_DEVICE)


def _cuda_candidates(
    count: int,
    *,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, ...]:
    tensors = tuple(
        value.to(device=CUDA_DEVICE, dtype=dtype)
        for value in make_candidates(count, dtype=dtype)
    )
    if requires_grad:
        tensors = tuple(value.requires_grad_() for value in tensors)
    return tensors


def _cuda_fixture(
    count: int,
    scores: torch.Tensor | None = None,
    *,
    candidate_dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
):
    depth = torch.ones(1, max(count, 1), dtype=candidate_dtype, device=CUDA_DEVICE)
    camera = ActiveCamera(depth)
    camera._pose = camera._pose.to(device=CUDA_DEVICE, dtype=candidate_dtype)
    camera.depth = camera.depth.to(device=CUDA_DEVICE, dtype=candidate_dtype)
    camera.depth_prior = camera.depth_prior.to(
        device=CUDA_DEVICE, dtype=candidate_dtype
    )
    if scores is None:
        scores = torch.arange(
            max(count, 1), dtype=candidate_dtype, device=CUDA_DEVICE
        )
    else:
        scores = scores.to(device=CUDA_DEVICE)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    getter = ActiveSnapshotGetter(scores)
    selector = GaussianCandidateActiveTopKV1(
        confidence_snapshot_getter=getter,
        logging_enabled=True,
        device=CUDA_DEVICE,
        active_budget=ACTIVE_BUDGET,
    )
    candidates = _cuda_candidates(
        count,
        dtype=candidate_dtype,
        requires_grad=requires_grad,
    )
    return selector, getter, camera, candidates


def _select(selector, camera, candidates, *, init: bool = False):
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


def _reference_winners(scores: torch.Tensor, k: int = ACTIVE_BUDGET) -> list[int]:
    """Independent CPU oracle: (-quality, original index), then source order."""

    values = [float(value) for value in scores.detach().cpu().reshape(-1).tolist()]
    ranked = sorted(range(len(values)), key=lambda index: (-values[index], index))
    return sorted(ranked[: min(len(values), k)])


def _outputs(result) -> tuple[torch.Tensor, ...]:
    return (
        result.xyz,
        result.features,
        result.scales,
        result.rotations,
        result.opacities,
    )


def _index_fingerprint(indices: torch.Tensor) -> str:
    payload = json.dumps(
        [int(value) for value in indices.detach().cpu().tolist()],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ConfidenceAccessTrap:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("confidence getter must not be called")


class CudaGateTestCase(unittest.TestCase):
    def tearDown(self) -> None:
        torch.cuda.synchronize(CUDA_DEVICE)


class StableCudaArgsortTests(CudaGateTestCase):
    def test_cuda_descending_stable_argsort_prefers_lower_tie_index(self):
        confidence = torch.tensor(
            [1.0, 4.0, 4.0, -2.0, 4.0], device=CUDA_DEVICE
        )
        ranked = torch.argsort(confidence, descending=True, stable=True)
        torch.cuda.synchronize(CUDA_DEVICE)
        self.assertEqual(ranked.cpu().tolist(), [1, 2, 4, 0, 3])

    def test_production_stable_topk_matches_independent_oracle(self):
        scores = torch.tensor(
            [2.0, -1.0, 9.0, 9.0, 3.0, 9.0], device=CUDA_DEVICE
        )
        selected = stable_quality_topk_indices(
            scores,
            candidate_count=6,
            requested_k=3,
            device=CUDA_DEVICE,
        )
        torch.cuda.synchronize(CUDA_DEVICE)
        self.assertEqual(selected.cpu().tolist(), _reference_winners(scores, 3))


class BypassAndFastPathCudaTests(CudaGateTestCase):
    def test_init_bypasses_confidence_and_selection(self):
        selector, _, camera, candidates = _cuda_fixture(601)
        trap = ConfidenceAccessTrap()
        selector._confidence_snapshot_getter = trap
        with mock.patch.object(
            active_module,
            "stable_quality_topk_indices",
            wraps=stable_quality_topk_indices,
        ) as quality, mock.patch.object(
            active_module,
            "deterministic_uniform_indices",
            wraps=deterministic_uniform_indices,
        ) as fallback:
            result = _select(selector, camera, candidates, init=True)
        self.assertEqual(trap.calls, 0)
        self.assertEqual(quality.call_count, 0)
        self.assertEqual(fallback.call_count, 0)
        self.assertIsNone(result.selected_indices)
        for original, output in zip(candidates, _outputs(result)):
            self.assertIs(original, output)

    def test_empty_candidates_bypass_confidence(self):
        selector, _, camera, candidates = _cuda_fixture(0)
        trap = ConfidenceAccessTrap()
        selector._confidence_snapshot_getter = trap
        result = _select(selector, camera, candidates)
        self.assertEqual(trap.calls, 0)
        self.assertIsNone(result.selected_indices)
        self.assertEqual(result.retained_candidate_count, 0)
        for original, output in zip(candidates, _outputs(result)):
            self.assertIs(original, output)

    def test_n_le_600_keeps_five_original_cuda_objects(self):
        for count in (1, 599, 600):
            with self.subTest(count=count):
                selector, _, camera, candidates = _cuda_fixture(
                    count, requires_grad=True
                )
                trap = ConfidenceAccessTrap()
                selector._confidence_snapshot_getter = trap
                with mock.patch.object(
                    active_module,
                    "stable_quality_topk_indices",
                    side_effect=AssertionError("fast path must not sort"),
                ) as quality:
                    result = _select(selector, camera, candidates)
                self.assertEqual(quality.call_count, 0)
                self.assertEqual(trap.calls, 0)
                self.assertIsNone(result.selected_indices)
                for original, output in zip(candidates, _outputs(result)):
                    self.assertIs(original, output)
                    self.assertEqual(output.device, CUDA_DEVICE)
                    self.assertEqual(output.dtype, original.dtype)
                    self.assertEqual(output.shape, original.shape)
                    self.assertTrue(output.requires_grad)


class ActiveTopKCorrectnessCudaTests(CudaGateTestCase):
    def _assert_selection(self, scores: torch.Tensor) -> None:
        count = int(scores.numel())
        selector, _, camera, candidates = _cuda_fixture(count, scores)
        result = _select(selector, camera, candidates)
        expected_list = _reference_winners(scores)
        expected = torch.tensor(expected_list, dtype=torch.long, device=CUDA_DEVICE)
        self.assertTrue(torch.equal(result.selected_indices, expected))
        self.assertEqual(result.selected_indices.shape, (ACTIVE_BUDGET,))
        self.assertTrue(
            bool(torch.all(result.selected_indices[1:] > result.selected_indices[:-1]))
        )
        for original, output in zip(candidates, _outputs(result)):
            self.assertEqual(output.device, CUDA_DEVICE)
            self.assertEqual(output.dtype, original.dtype)
            self.assertEqual(output.shape[0], ACTIVE_BUDGET)
            self.assertEqual(output.shape[1:], original.shape[1:])
            self.assertTrue(torch.equal(output, original.index_select(0, expected)))

    def test_n_601_boundary(self):
        scores = torch.arange(601, dtype=torch.float32, device=CUDA_DEVICE)
        self._assert_selection(scores)

    def test_all_equal_confidence(self):
        scores = torch.ones(601, dtype=torch.float32, device=CUDA_DEVICE)
        self._assert_selection(scores)

    def test_local_ties_and_mixed_signed_confidence(self):
        scores = torch.linspace(-5.0, 5.0, 601, device=CUDA_DEVICE)
        scores[50:75] = 3.0
        scores[300:330] = -1.0
        scores[590:601] = 6.0
        self._assert_selection(scores)

    def test_tie_crossing_rank_600_boundary(self):
        scores = torch.arange(601, 0, -1, dtype=torch.float32, device=CUDA_DEVICE)
        scores[599] = 1.0
        scores[600] = 1.0
        self._assert_selection(scores)
        selector, _, camera, candidates = _cuda_fixture(601, scores)
        result = _select(selector, camera, candidates)
        self.assertIn(599, result.selected_indices.cpu().tolist())
        self.assertNotIn(600, result.selected_indices.cpu().tolist())

    def test_mixed_candidate_dtypes_are_preserved_per_field(self):
        scores = torch.arange(601, dtype=torch.float32, device=CUDA_DEVICE)
        selector, _, camera, candidates = _cuda_fixture(601, scores)
        candidates = (
            candidates[0].to(torch.float64),
            candidates[1].to(torch.float32),
            candidates[2].to(torch.float64),
            candidates[3].to(torch.float32),
            candidates[4].to(torch.float64),
        )
        camera._pose = camera._pose.to(torch.float64)
        camera.depth = camera.depth.to(torch.float64)
        camera.depth_prior = camera.depth_prior.to(torch.float64)
        result = _select(selector, camera, candidates)
        for original, output in zip(candidates, _outputs(result)):
            self.assertEqual(output.dtype, original.dtype)
            self.assertEqual(output.device, original.device)

    def test_outer_factory_active_cuda_integration_and_gates(self):
        scores = torch.arange(601, dtype=torch.float32, device=CUDA_DEVICE)
        _, getter, camera, candidates = _cuda_fixture(601, scores)
        config = {
            "mode": "active",
            "budget": 600,
            "logging": {"enabled": True},
        }
        selector = build_gaussian_candidate_selector_v1(
            config,
            candidate_observer=None,
            resource_admission_mode="disabled",
            confidence_snapshot_getter=getter,
            device=CUDA_DEVICE,
        )
        self.assertIsInstance(selector, GaussianCandidateActiveTopKV1)
        self.assertTrue(selector.is_active)
        self.assertEqual(selector.device, CUDA_DEVICE)
        self.assertEqual(selector.active_budget, ACTIVE_BUDGET)

        result = _select(selector, camera, candidates)
        expected = torch.tensor(
            _reference_winners(scores), dtype=torch.long, device=CUDA_DEVICE
        )
        self.assertTrue(torch.equal(result.selected_indices, expected))
        for original, output in zip(candidates, _outputs(result)):
            self.assertEqual(output.device, CUDA_DEVICE)
            self.assertTrue(torch.equal(output, original.index_select(0, expected)))

        with self.assertRaisesRegex(ValueError, "candidate_observer.mode=off"):
            build_gaussian_candidate_selector_v1(
                config,
                candidate_observer=SimpleNamespace(voxel_size=0.05),
                resource_admission_mode="disabled",
                confidence_snapshot_getter=getter,
                device=CUDA_DEVICE,
            )
        for m01_mode in ("observe", "fixed_budget"):
            with self.subTest(m01_mode=m01_mode), self.assertRaisesRegex(
                ValueError, "requires M01"
            ):
                build_gaussian_candidate_selector_v1(
                    config,
                    candidate_observer=None,
                    resource_admission_mode=m01_mode,
                    confidence_snapshot_getter=getter,
                    device=CUDA_DEVICE,
                )


class DeterminismAndAutogradCudaTests(CudaGateTestCase):
    def test_one_hundred_tied_runs_are_identical(self):
        count = 1200
        scores = torch.ones(count, dtype=torch.float32, device=CUDA_DEVICE)
        selector, _, camera, candidates = _cuda_fixture(count, scores)
        first = _select(selector, camera, candidates)
        expected = torch.arange(ACTIVE_BUDGET, device=CUDA_DEVICE)
        self.assertTrue(torch.equal(first.selected_indices, expected))
        first_outputs = tuple(value.clone() for value in _outputs(first))
        first_fingerprint = _index_fingerprint(first.selected_indices)
        for _ in range(99):
            result = _select(selector, camera, candidates)
            self.assertTrue(torch.equal(result.selected_indices, expected))
            self.assertEqual(
                _index_fingerprint(result.selected_indices), first_fingerprint
            )
            for baseline, output in zip(first_outputs, _outputs(result)):
                self.assertTrue(torch.equal(output, baseline))
        torch.cuda.synchronize(CUDA_DEVICE)
        self.assertEqual(
            first.token.fields["selected_indices_fingerprint"],
            _select(selector, camera, candidates).token.fields[
                "selected_indices_fingerprint"
            ],
        )

    def test_n_gt_600_index_select_retains_backward_relation(self):
        count = 601
        scores = torch.arange(count, dtype=torch.float32, device=CUDA_DEVICE)
        selector, _, camera, candidates = _cuda_fixture(
            count, scores, requires_grad=True
        )
        result = _select(selector, camera, candidates)
        self.assertTrue(result.xyz.requires_grad)
        row_weights = torch.arange(
            1, ACTIVE_BUDGET + 1, dtype=result.xyz.dtype, device=CUDA_DEVICE
        )
        loss = (result.xyz[:, 0] * row_weights).sum()
        loss.backward()
        expected = torch.zeros(count, dtype=result.xyz.dtype, device=CUDA_DEVICE)
        expected[result.selected_indices] = row_weights
        self.assertTrue(torch.equal(candidates[0].grad[:, 0], expected))
        self.assertTrue(torch.equal(candidates[0].grad[:, 1:], torch.zeros_like(
            candidates[0].grad[:, 1:]
        )))

    def test_n_le_600_identity_path_retains_backward_relation(self):
        selector, _, camera, candidates = _cuda_fixture(600, requires_grad=True)
        result = _select(selector, camera, candidates)
        self.assertIs(result.xyz, candidates[0])
        result.xyz.sum().backward()
        self.assertTrue(torch.equal(candidates[0].grad, torch.ones_like(candidates[0])))


class FallbackAndFailClosedCudaTests(CudaGateTestCase):
    def _assert_fixed_fallback(self, selector, camera, candidates, reason: str):
        with mock.patch.object(
            active_module,
            "stable_quality_topk_indices",
            wraps=stable_quality_topk_indices,
        ) as quality, mock.patch.object(
            active_module,
            "deterministic_uniform_indices",
            wraps=deterministic_uniform_indices,
        ) as fallback:
            result = _select(selector, camera, candidates)
        expected = deterministic_uniform_indices(
            candidate_count=601,
            fixed_budget=ACTIVE_BUDGET,
            device=CUDA_DEVICE,
        )
        self.assertEqual(quality.call_count, 0)
        self.assertEqual(fallback.call_count, 1)
        self.assertTrue(torch.equal(result.selected_indices, expected))
        self.assertEqual(result.selected_indices.device, CUDA_DEVICE)
        self.assertEqual(result.token.fields["fallback_reason"]["code"], reason)
        return result

    def test_nan_and_inf_are_allowed_fixed600_fallbacks(self):
        for value in (torch.nan, torch.inf, -torch.inf):
            with self.subTest(value=value):
                scores = torch.arange(601, dtype=torch.float32, device=CUDA_DEVICE)
                scores[17] = value
                selector, _, camera, candidates = _cuda_fixture(601, scores)
                self._assert_fixed_fallback(
                    selector, camera, candidates, "nonfinite_confidence"
                )

    def test_defined_provenance_failure_is_allowed_fallback(self):
        selector, getter, camera, candidates = _cuda_fixture(601)
        getter.overrides.update({"is_current": False, "is_stale": True})
        self._assert_fixed_fallback(
            selector, camera, candidates, "stale_confidence"
        )

    def test_confidence_shape_and_dtype_are_allowed_fallbacks(self):
        selector, getter, camera, candidates = _cuda_fixture(601)
        getter.confidence = torch.ones(1, 1, 601, device=CUDA_DEVICE)
        self._assert_fixed_fallback(
            selector, camera, candidates, "confidence_shape_mismatch"
        )

        selector, getter, camera, candidates = _cuda_fixture(601)
        getter.confidence = torch.ones(
            1, 601, dtype=torch.int64, device=CUDA_DEVICE
        )
        self._assert_fixed_fallback(
            selector, camera, candidates, "confidence_dtype_mismatch"
        )

    def test_candidate_first_dimension_mismatch_fails_closed(self):
        selector, _, camera, candidates = _cuda_fixture(601)
        invalid = list(candidates)
        invalid[1] = invalid[1][:-1]
        with mock.patch.object(
            active_module, "deterministic_uniform_indices"
        ) as fallback, self.assertRaisesRegex(ValueError, "first-dimension mismatch"):
            _select(selector, camera, tuple(invalid))
        self.assertEqual(fallback.call_count, 0)

    def test_one_cpu_candidate_field_among_cuda_fields_fails_closed(self):
        selector, _, camera, candidates = _cuda_fixture(601)
        invalid = list(candidates)
        invalid[4] = invalid[4].cpu()
        with mock.patch.object(
            active_module, "deterministic_uniform_indices"
        ) as fallback, self.assertRaisesRegex(ValueError, "device mismatch"):
            _select(selector, camera, tuple(invalid))
        self.assertEqual(fallback.call_count, 0)

    def test_candidate_shape_contract_fails_closed(self):
        selector, _, camera, candidates = _cuda_fixture(601)
        invalid = list(candidates)
        invalid[3] = torch.zeros(601, 3, device=CUDA_DEVICE)
        with mock.patch.object(
            active_module, "deterministic_uniform_indices"
        ) as fallback, self.assertRaisesRegex(ValueError, "shape contract"):
            _select(selector, camera, tuple(invalid))
        self.assertEqual(fallback.call_count, 0)

    def test_unknown_runtime_error_is_not_converted_to_fixed600(self):
        selector, _, camera, candidates = _cuda_fixture(601)

        def programmer_bug(*args, **kwargs):
            raise RuntimeError("injected non-whitelisted implementation error")

        selector._confidence_snapshot_getter = programmer_bug
        with mock.patch.object(
            active_module, "deterministic_uniform_indices"
        ) as fallback, self.assertRaisesRegex(RuntimeError, "non-whitelisted"):
            _select(selector, camera, candidates)
        self.assertEqual(fallback.call_count, 0)


class GaussianModelCudaCallCountTests(CudaGateTestCase):
    @classmethod
    def setUpClass(cls):
        cls.extend_from_pcd_seq = staticmethod(
            load_gaussian_model_extend_from_pcd_seq()
        )

    def _run_model(self, harness, selector, camera, *, init: bool = False):
        with redirect_stdout(io.StringIO()):
            self.extend_from_pcd_seq(
                harness,
                camera,
                camera.uid,
                init=init,
                candidate_selector_v1=selector,
                mapper_update_id=23,
            )

    def test_model_active_success_call_counts_on_cuda(self):
        selector, getter, camera, candidates = _cuda_fixture(601)
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
            self._run_model(harness, selector, camera)
        self.assertEqual(quality.call_count, 1)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(getter.calls, 1)
        self.assertEqual(harness.m01_calls, 0)
        self.assertEqual(harness.extend_calls, 1)
        self.assertEqual(harness.last_extended_count, 600)

    def test_model_allowed_fallback_call_counts_on_cuda(self):
        selector, getter, camera, candidates = _cuda_fixture(601)
        harness = GaussianModelHookHarness(candidates)

        def stale(*args, **kwargs):
            getter.calls += 1
            raise RuntimeError("confidence is stale")

        selector._confidence_snapshot_getter = stale
        with mock.patch.object(
            active_module,
            "stable_quality_topk_indices",
            wraps=stable_quality_topk_indices,
        ) as quality, mock.patch.object(
            active_module,
            "deterministic_uniform_indices",
            wraps=deterministic_uniform_indices,
        ) as fallback:
            self._run_model(harness, selector, camera)
        self.assertEqual(quality.call_count, 0)
        self.assertEqual(fallback.call_count, 1)
        self.assertEqual(getter.calls, 1)
        self.assertEqual(harness.m01_calls, 0)
        self.assertEqual(harness.extend_calls, 1)
        self.assertEqual(harness.last_extended_count, 600)

    def test_model_init_call_counts_on_cuda(self):
        selector, getter, camera, candidates = _cuda_fixture(601)
        harness = GaussianModelHookHarness(candidates)
        with mock.patch.object(
            active_module, "stable_quality_topk_indices"
        ) as quality, mock.patch.object(
            active_module, "deterministic_uniform_indices"
        ) as fallback:
            self._run_model(harness, selector, camera, init=True)
        self.assertEqual(quality.call_count, 0)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(getter.calls, 0)
        self.assertEqual(harness.m01_calls, 0)
        self.assertEqual(harness.extend_calls, 1)
        self.assertEqual(harness.last_extended_count, 601)

    def test_model_empty_call_counts_on_cuda(self):
        selector, getter, camera, _ = _cuda_fixture(0)
        harness = GaussianModelHookHarness(None)
        with mock.patch.object(
            active_module, "stable_quality_topk_indices"
        ) as quality, mock.patch.object(
            active_module, "deterministic_uniform_indices"
        ) as fallback:
            self._run_model(harness, selector, camera)
        self.assertEqual(quality.call_count, 0)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(getter.calls, 0)
        self.assertEqual(harness.m01_calls, 0)
        self.assertEqual(harness.extend_calls, 0)


if __name__ == "__main__":
    unittest.main()
