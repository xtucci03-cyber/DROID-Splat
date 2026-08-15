"""No-dataset CUDA gates for V2 observe and fixed-K active modes.

A valid server run executes every test with zero skips.  The active tests use
real CUDA tensors; the production-hook test also forwards through the real
production renderer and Gaussian rasterizer fixture.
"""

from contextlib import redirect_stdout
import io
import inspect
import json
import math
from types import MethodType, SimpleNamespace
import unittest
from unittest import mock

import torch

from src.candidate_selection.gaussian_candidate_selector_v2 import (
    NUMERICAL_EPSILON,
    GaussianCandidateActiveFixedKV2,
    GaussianCandidateActiveDynamicKV2,
    GaussianCandidateDynamicKObserveV2,
    GaussianCandidateSelectorV2,
    build_gaussian_candidate_selector_v2,
)
from src.candidate_selection.gaussian_candidate_dynamic_budget_v1 import (
    DynamicBudgetObserveConfigV1,
    GaussianCandidateDynamicBudgetObserverV1,
)
from src.candidate_selection.preinsert_render_evidence_v1 import (
    PreinsertRenderEvidenceObserverV1,
    PreinsertRenderEvidenceV1,
)
import tests.test_preinsert_render_evidence_v1_cuda as raster_fixture
import tests.test_preinsert_render_evidence_v1_mapper_cuda as mapper_fixture


CUDA_DEVICE = torch.device("cuda:0")
SQRT_2 = math.sqrt(2.0)


def _contains_tensor(value, seen=None):
    if isinstance(value, torch.Tensor):
        return True
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    if isinstance(value, dict):
        return any(_contains_tensor(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item, seen) for item in value)
    return False


class _CurrentSnapshotGetter:
    def __init__(self, confidence, *, error=None):
        self.confidence = confidence
        self.error = error
        self.calls = 0

    def __call__(self, camera, require_current=True, *, upsampled=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        if not require_current or upsampled is not False:
            raise AssertionError("V2 must request the current low-resolution snapshot.")
        return SimpleNamespace(
            buffer_index=camera.buffer_index,
            source_frame_id=camera.source_frame_id,
            source_timestamp=camera.source_timestamp,
            confidence_source_frame_id=camera.source_frame_id,
            confidence_version=9,
            confidence_up_version=3,
            is_current=True,
            is_stale=False,
            confidence=self.confidence,
            shape=tuple(self.confidence.shape),
            dtype=str(self.confidence.dtype),
            device=str(self.confidence.device),
            requires_grad=bool(self.confidence.requires_grad),
        )


@unittest.skipUnless(
    torch.cuda.is_available(),
    "GCS-v2 V2-A CUDA gate not run: CUDA is unavailable.",
)
class GaussianCandidateSelectorV2CudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.device_count() < 1:
            raise RuntimeError("GCS-v2 V2-A CUDA gate requires cuda:0.")
        torch.cuda.set_device(CUDA_DEVICE)

        # Any production extension import/kernel failure is an ERROR, not a skip.
        raster_fixture.PreinsertRenderEvidenceRealRasterizerCudaTests.setUpClass()
        mapper_fixture.GaussianMapperPreinsertRealCudaTests.setUpClass()
        cls.raster = raster_fixture.PreinsertRenderEvidenceRealRasterizerCudaTests(
            methodName="runTest"
        )
        cls.mapper = mapper_fixture.GaussianMapperPreinsertRealCudaTests(
            methodName="runTest"
        )
        torch.cuda.synchronize(CUDA_DEVICE)

    @staticmethod
    def _candidates(xyz):
        count = int(xyz.shape[0])
        return (
            xyz,
            torch.arange(count * 3, dtype=torch.float32, device=CUDA_DEVICE).reshape(
                count, 3, 1
            ),
            torch.full((count, 3), -2.0, device=CUDA_DEVICE),
            torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]] * count,
                dtype=torch.float32,
                device=CUDA_DEVICE,
            ),
            torch.zeros(count, 1, device=CUDA_DEVICE),
        )

    @staticmethod
    def _selector(confidence, *, logging_enabled=False):
        getter = _CurrentSnapshotGetter(confidence)
        selector = GaussianCandidateSelectorV2(
            confidence_snapshot_getter=getter,
            logging_enabled=logging_enabled,
            device=CUDA_DEVICE,
        )
        return selector, getter

    @staticmethod
    def _active_selector(
        confidence,
        *,
        strategy,
        fixed_k=2,
        diversity_strength=0.1,
        confidence_error=None,
        logging_enabled=False,
    ):
        getter = _CurrentSnapshotGetter(confidence, error=confidence_error)
        selector = GaussianCandidateActiveFixedKV2(
            confidence_snapshot_getter=getter,
            logging_enabled=logging_enabled,
            device=CUDA_DEVICE,
            strategy=strategy,
            fixed_k=fixed_k,
            diversity_strength=diversity_strength,
        )
        return selector, getter

    @staticmethod
    def _dynamic_selector(confidence, *, logging_enabled=False):
        getter = _CurrentSnapshotGetter(confidence)
        selector = GaussianCandidateDynamicKObserveV2(
            confidence_snapshot_getter=getter,
            logging_enabled=logging_enabled,
            device=CUDA_DEVICE,
            dynamic_budget_config=DynamicBudgetObserveConfigV1(
                observe_histogram_bins=32,
                k_max_reference=600,
            ),
        )
        return selector, getter

    @staticmethod
    def _dynamic_active_selector(
        confidence, *, bins=32, threshold_bin=1, k_min=256, k_max=600,
        logging_enabled=False,
    ):
        getter = _CurrentSnapshotGetter(confidence)
        selector = GaussianCandidateActiveDynamicKV2(
            confidence_snapshot_getter=getter,
            logging_enabled=logging_enabled,
            device=CUDA_DEVICE,
            dynamic_budget_config=DynamicBudgetObserveConfigV1(
                observe_histogram_bins=bins,
                k_max_reference=600,
                threshold_bin=threshold_bin,
                k_min=k_min,
                k_max=k_max,
            ),
        )
        return selector, getter

    def _active_scene(
        self,
        *,
        alpha,
        confidence,
        rgb_l1=None,
        old_depth=None,
        gaussian_before=11,
    ):
        alpha_values = torch.as_tensor(
            alpha,
            dtype=torch.float32,
            device=CUDA_DEVICE,
        ).reshape(-1)
        count = int(alpha_values.shape[0])
        height, width = 2, count
        camera = SimpleNamespace(
            uid=7,
            buffer_index=17,
            source_frame_id=107,
            source_timestamp=7.25,
            image_height=height,
            image_width=width,
            device=str(CUDA_DEVICE),
            pose=torch.eye(4, dtype=torch.float32, device=CUDA_DEVICE),
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
            depth=torch.full(
                (height, width),
                2.0,
                dtype=torch.float32,
                device=CUDA_DEVICE,
            ),
            depth_prior=torch.full(
                (height, width),
                2.0,
                dtype=torch.float32,
                device=CUDA_DEVICE,
            ),
            original_image=torch.zeros(
                3,
                height,
                width,
                dtype=torch.float32,
                device=CUDA_DEVICE,
            ),
            mask=torch.ones(
                1,
                height,
                width,
                dtype=torch.bool,
                device=CUDA_DEVICE,
            ),
        )
        xyz = torch.tensor(
            [[2.0 * index, 0.0, 2.0] for index in range(count)],
            dtype=torch.float32,
            device=CUDA_DEVICE,
        )
        candidates = self._candidates(xyz)
        alpha_accum = torch.zeros(
            1,
            height,
            width,
            dtype=torch.float32,
            device=CUDA_DEVICE,
        )
        alpha_accum[0, 0, :] = alpha_values
        rgb_values = torch.zeros_like(alpha_values)
        if rgb_l1 is not None:
            rgb_values = torch.as_tensor(
                rgb_l1,
                dtype=torch.float32,
                device=CUDA_DEVICE,
            ).reshape(-1)
        render_rgb = torch.zeros(
            3,
            height,
            width,
            dtype=torch.float32,
            device=CUDA_DEVICE,
        )
        render_rgb[:, 0, :] = rgb_values.unsqueeze(0)
        depth_values = 2.0 * torch.ones_like(alpha_values)
        if old_depth is not None:
            depth_values = torch.as_tensor(
                old_depth,
                dtype=torch.float32,
                device=CUDA_DEVICE,
            ).reshape(-1)
        depth_accum = torch.zeros_like(alpha_accum)
        depth_accum[0, 0, :] = alpha_values * depth_values
        evidence = PreinsertRenderEvidenceV1(
            render_rgb=render_rgb,
            depth_accum=depth_accum,
            alpha_accum=alpha_accum,
            source_camera_id=camera.uid,
            source_frame_id=camera.source_frame_id,
            source_buffer_index=camera.buffer_index,
            source_timestamp=camera.source_timestamp,
            height=height,
            width=width,
            gaussian_count_before_render=gaussian_before,
            available=True,
            unavailable_reason=None,
            mapper_update_id=23,
        )
        confidence_tensor = torch.as_tensor(
            confidence,
            dtype=torch.float32,
            device=CUDA_DEVICE,
        )
        if confidence_tensor.ndim == 1:
            confidence_tensor = confidence_tensor.unsqueeze(0)
        return camera, candidates, evidence, confidence_tensor

    @staticmethod
    def _active_select(
        selector,
        candidates,
        camera,
        evidence,
        *,
        gaussian_before=11,
    ):
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
            init=False,
            gaussian_before=gaussian_before,
            preinsert_render_evidence=evidence,
        )

    @staticmethod
    def _observe(selector, candidates, camera, evidence, *, gaussian_before=11):
        return selector.observe_before_extend(
            xyz=candidates[0],
            features=candidates[1],
            scales=candidates[2],
            rotations=candidates[3],
            opacities=candidates[4],
            camera=camera,
            depthmap=None,
            depth_source="estimated_clean_depth",
            mapper_update_id=23,
            init=False,
            gaussian_before=gaussian_before,
            preinsert_render_evidence=evidence,
        )

    def test_cuda_candidate_component_numerics_and_no_mutation(self):
        height, width = 2, 4
        camera = SimpleNamespace(
            uid=7,
            buffer_index=17,
            source_frame_id=107,
            source_timestamp=7.25,
            image_height=height,
            image_width=width,
            device=str(CUDA_DEVICE),
            pose=torch.eye(4, device=CUDA_DEVICE),
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
            depth=torch.full((height, width), 2.0, device=CUDA_DEVICE),
            depth_prior=torch.full((height, width), 2.0, device=CUDA_DEVICE),
            original_image=torch.zeros(3, height, width, device=CUDA_DEVICE),
            mask=torch.ones(1, height, width, dtype=torch.bool, device=CUDA_DEVICE),
        )
        camera.original_image[:, 0, 1] = 255.0
        candidates = self._candidates(torch.tensor(
            [[0.0, 0.0, 2.0], [2.0, 0.0, 2.0],
             [4.0, 0.0, 2.0], [6.0, 0.0, 2.0]], device=CUDA_DEVICE
        ))
        confidence = torch.tensor([[-SQRT_2, 2.0 * SQRT_2]], device=CUDA_DEVICE)
        alpha = torch.tensor(
            [[[0.0, NUMERICAL_EPSILON / 2.0, 0.5, 1.0],
              [0.0, 0.0, 0.0, 0.0]]],
            device=CUDA_DEVICE,
        )
        render_rgb = torch.zeros(3, height, width, device=CUDA_DEVICE)
        render_rgb[:, 0, :] = torch.tensor(
            [0.0, 0.25, 0.5, 1.0], device=CUDA_DEVICE
        )
        evidence = PreinsertRenderEvidenceV1(
            render_rgb=render_rgb,
            depth_accum=2.0 * alpha,
            alpha_accum=alpha,
            source_camera_id=camera.uid,
            source_frame_id=camera.source_frame_id,
            source_buffer_index=camera.buffer_index,
            source_timestamp=camera.source_timestamp,
            height=height,
            width=width,
            gaussian_count_before_render=11,
            available=True,
            unavailable_reason=None,
            mapper_update_id=23,
        )
        selector, getter = self._selector(confidence, logging_enabled=True)
        self.assertTrue(all(value.device == CUDA_DEVICE for value in candidates))
        self.assertEqual(confidence.device, CUDA_DEVICE)
        self.assertEqual(tuple(evidence.render_rgb.shape), (3, height, width))
        self.assertEqual(tuple(evidence.depth_accum.shape), (1, height, width))
        self.assertEqual(tuple(evidence.alpha_accum.shape), (1, height, width))
        states = [
            (value, value.data_ptr(), value._version, tuple(value.shape),
             value.dtype, value.device, value.detach().clone())
            for value in candidates
        ]
        token = self._observe(selector, candidates, camera, evidence)
        self.assertEqual(getter.calls, 1)
        self.assertTrue(token.fields["diagnostic_gpu_to_cpu_sync_expected"])
        self.assertTrue(math.isfinite(token.fields["host_wall_elapsed_ms"]))
        self.assertGreaterEqual(token.fields["host_wall_elapsed_ms"], 0.0)
        self.assertEqual(token.fields["confidence_norm"]["min"], 0.0)
        self.assertEqual(token.fields["confidence_norm"]["max"], 1.0)
        self.assertAlmostEqual(token.fields["confidence_norm"]["mean"], 0.5)
        self.assertEqual(token.fields["alpha_valid_count"], 4)
        self.assertEqual(token.fields["coverage_gap_valid_count"], 4)
        self.assertEqual(token.fields["alpha"]["min"], 0.0)
        self.assertEqual(token.fields["alpha"]["max"], 1.0)
        self.assertEqual(token.fields["coverage_gap"]["min"], 0.0)
        self.assertEqual(token.fields["coverage_gap"]["max"], 1.0)
        self.assertEqual(token.fields["depth_relative_error_valid_count"], 2)
        self.assertEqual(token.fields["depth_relative_error"]["max"], 0.0)
        self.assertAlmostEqual(token.fields["rgb_l1"]["max"], 1.0)
        self.assertAlmostEqual(token.fields["rgb_l1"]["mean"], 0.5625)
        self.assertEqual(token.fields["multiplicity"]["min"], 2.0)
        self.assertEqual(token.fields["cell_redundancy"]["max"], 0.5)
        self.assertFalse(token.fields["selected_indices_created"])
        observe_source = inspect.getsource(GaussianCandidateSelectorV2)
        self.assertNotIn("index_select", observe_source)
        self.assertFalse(hasattr(selector, "select_before_extend"))
        self.assertFalse(_contains_tensor(token.fields))
        for candidate, state in zip(candidates, states):
            actual, pointer, version, shape, dtype, device, values = state
            self.assertIs(actual, candidate)
            self.assertEqual(actual.data_ptr(), pointer)
            self.assertEqual(actual._version, version)
            self.assertEqual(tuple(actual.shape), shape)
            self.assertEqual(actual.dtype, dtype)
            self.assertEqual(actual.device, device)
            torch.testing.assert_close(actual, values, atol=0.0, rtol=0.0)

        one_token = self._observe(
            selector, tuple(value[:1] for value in candidates), camera, evidence
        )
        self.assertEqual(one_token.fields["multiplicity"]["min"], 1.0)
        self.assertEqual(one_token.fields["cell_redundancy"]["max"], 0.0)

        stream = io.StringIO()
        with redirect_stdout(stream):
            summary = selector.record_after_extend(
                token,
                admitted_candidate_count=4,
                dropped_candidate_count=0,
                gaussian_after_extend=15,
            )
        event = json.loads(stream.getvalue().split(" ", 1)[1])
        self.assertEqual(event, summary.to_event())
        self.assertFalse(_contains_tensor(event))
        self.assertNotIn("selected_indices", event)

    def test_real_rasterizer_evidence_flows_into_v2_without_second_render(self):
        camera, model, pipeline, background = self.raster._make_scene()
        camera.depth = torch.full((raster_fixture.HEIGHT, raster_fixture.WIDTH),
                                  raster_fixture.GAUSSIAN_CAMERA_Z,
                                  device=CUDA_DEVICE)
        camera.mask = torch.ones(1, raster_fixture.HEIGHT, raster_fixture.WIDTH,
                                 dtype=torch.bool, device=CUDA_DEVICE)
        observer = PreinsertRenderEvidenceObserverV1(device=CUDA_DEVICE)
        calls = 0

        def real_renderer(*args, **kwargs):
            nonlocal calls
            calls += 1
            return self.raster.production_render(*args, **kwargs)

        evidence = observer.capture(
            camera=camera,
            gaussians=model,
            renderer=real_renderer,
            pipeline_params=pipeline,
            background=background,
            mapper_update_id=23,
        )
        candidates = self._candidates(torch.tensor([[0.0, 0.0, 2.0]],
                                                    device=CUDA_DEVICE))
        selector, _ = self._selector(torch.tensor([[SQRT_2]], device=CUDA_DEVICE))
        token = self._observe(
            selector, candidates, camera, evidence, gaussian_before=len(model)
        )
        torch.cuda.synchronize(CUDA_DEVICE)
        self.assertEqual(calls, 1)
        self.assertTrue(token.fields["evidence_available"])
        for tensor, shape in (
            (evidence.render_rgb, (3, raster_fixture.HEIGHT, raster_fixture.WIDTH)),
            (evidence.depth_accum, (1, raster_fixture.HEIGHT, raster_fixture.WIDTH)),
            (evidence.alpha_accum, (1, raster_fixture.HEIGHT, raster_fixture.WIDTH)),
        ):
            self.assertEqual(tuple(tensor.shape), shape)
            self.assertEqual(tensor.device, CUDA_DEVICE)
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)
            self.assertTrue(bool(torch.isfinite(tensor).all()))
        for component in ("alpha", "coverage_gap", "rgb_l1", "depth_relative_error"):
            stats = token.fields[component]
            self.assertGreater(stats["count"], 0)
            for key in ("min", "mean", "p50", "p90", "p95", "max"):
                self.assertTrue(math.isfinite(stats[key]))
        self.assertFalse(any(isinstance(value, PreinsertRenderEvidenceV1)
                             for value in vars(selector).values()))

    def test_bound_mapper_runs_v2_before_real_extend_and_records_conservation(self):
        camera, model, observer, mapper, events = self.mapper._scene()
        tokens = []
        summaries = []
        candidate_states = []
        selector, getter = self._selector(
            torch.tensor([[SQRT_2]], device=CUDA_DEVICE)
        )

        original_observe = selector.observe_before_extend
        original_record = selector.record_after_extend

        def observe_spy(**kwargs):
            events.append("v2_observe_before")
            candidate_states.extend(
                (value, value.data_ptr(), value._version, value.detach().clone())
                for value in (
                    kwargs["xyz"], kwargs["features"], kwargs["scales"],
                    kwargs["rotations"], kwargs["opacities"]
                )
            )
            token = original_observe(**kwargs)
            tokens.append(token)
            return token

        def record_spy(token, **kwargs):
            events.append("v2_record_after")
            summary = original_record(token, **kwargs)
            summaries.append(summary)
            return summary

        selector.observe_before_extend = observe_spy
        selector.record_after_extend = record_spy
        mapper.candidate_selector_v2 = selector
        render_calls = 0
        captured_evidence = []
        original_capture = observer.capture

        def capture_spy(**kwargs):
            evidence = original_capture(**kwargs)
            captured_evidence.append(evidence)
            return evidence

        def renderer_spy(*args, **kwargs):
            nonlocal render_calls
            render_calls += 1
            events.append("render")
            return self.mapper.production_render(*args, **kwargs)

        self.assertIs(mapper.add_new_gaussians.__func__,
                      self.mapper.GaussianMapper.add_new_gaussians)
        self.assertIs(model.extend_from_pcd_seq.__func__,
                      self.mapper.GaussianModel.extend_from_pcd_seq)
        before = len(model)
        with mock.patch.object(observer, "capture", side_effect=capture_spy), \
             mock.patch.object(self.mapper.mapper_module, "render",
                               side_effect=renderer_spy):
            returned = mapper.add_new_gaussians([camera])
        torch.cuda.synchronize(CUDA_DEVICE)
        self.assertIs(returned, camera)
        self.assertEqual(render_calls, 1)
        self.assertEqual(
            events,
            [
                "render",
                "candidate_generation",
                "v2_observe_before",
                "extend",
                "v2_record_after",
            ],
        )
        self.assertEqual(getter.calls, 1)
        self.assertEqual(len(captured_evidence), 1)
        self.assertEqual(captured_evidence[0].gaussian_count_before_render, before)
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].fields["gaussian_before"], before)
        self.assertEqual(len(model), before + 6)
        self.assertEqual(summaries[0].fields["status"], "ok")
        self.assertTrue(summaries[0].fields["conservation_check"])
        for value, pointer, version, frozen in candidate_states:
            self.assertEqual(value.data_ptr(), pointer)
            self.assertEqual(value._version, version)
            torch.testing.assert_close(value, frozen, atol=0.0, rtol=0.0)
        evidence_ptrs = {
            value.data_ptr()
            for value in (
                captured_evidence[0].render_rgb,
                captured_evidence[0].depth_accum,
                captured_evidence[0].alpha_accum,
            )
        }
        for owner in (mapper, camera, model, observer, selector):
            self.assertFalse(any(isinstance(value, PreinsertRenderEvidenceV1)
                                 for value in vars(owner).values()))
            self.assertFalse(any(isinstance(value, torch.Tensor)
                                 and value.data_ptr() in evidence_ptrs
                                 for value in vars(owner).values()))

    def test_dynamic_k_observe_cuda_histogram_matches_cpu_oracle_without_full_q_copy(self):
        threshold = 1.0 / 32.0
        delta = 1.0 / 4096.0
        quality = torch.tensor(
            [
                0.0,
                threshold - delta,
                threshold,
                threshold + delta,
                1.0,
                float("nan"),
                -0.1,
                1.1,
            ],
            dtype=torch.float32,
            device=CUDA_DEVICE,
        )
        valid = torch.ones(quality.shape[0], dtype=torch.bool, device=CUDA_DEVICE)
        observer = GaussianCandidateDynamicBudgetObserverV1(
            DynamicBudgetObserveConfigV1(
                observe_histogram_bins=32,
                k_max_reference=600,
            )
        )
        result = observer.observe(quality, valid)
        cpu_values = torch.tensor(
            [0.0, threshold - delta, threshold, threshold + delta, 1.0],
            dtype=torch.float32,
        )
        cpu_bins = torch.clamp(
            torch.floor(cpu_values * 32).to(torch.long),
            min=0,
            max=31,
        )
        cpu_histogram = torch.bincount(cpu_bins, minlength=32).to(torch.int64)
        expected = tuple(int(cpu_histogram[index]) for index in range(32))
        self.assertEqual(result.q_histogram_counts, expected)
        self.assertEqual(result.q_valid_count, 5)
        self.assertEqual(result.q_invalid_count, 3)
        self.assertEqual(sum(result.q_histogram_counts[1:]), 3)
        self.assertTrue(result.diagnostic_gpu_to_cpu_sync_expected)
        source = inspect.getsource(GaussianCandidateDynamicBudgetObserverV1.observe)
        self.assertEqual(source.count(".cpu()"), 1)
        self.assertNotIn("quality.cpu", source)
        self.assertNotIn(".tolist(", source)
        self.assertNotIn("torch.quantile", source)

    def test_dynamic_k_observe_cuda_preserves_candidate_versions_and_autograd(self):
        camera, base_candidates, evidence, confidence = self._active_scene(
            alpha=[0.0, 0.25, 0.5, 1.0],
            confidence=[SQRT_2, SQRT_2],
            rgb_l1=[0.0, 0.25, 0.5, 1.0],
        )
        candidates = tuple(
            value.detach().clone().requires_grad_(True) for value in base_candidates
        )
        states = tuple(
            (
                value.data_ptr(),
                value._version,
                tuple(value.shape),
                value.dtype,
                value.device,
                value.detach().clone(),
            )
            for value in candidates
        )
        selector, getter = self._dynamic_selector(confidence)
        with mock.patch.object(
            torch,
            "index_select",
            side_effect=AssertionError("dynamic observe must not index candidates"),
        ):
            token = selector.observe_before_extend(
                xyz=candidates[0],
                features=candidates[1],
                scales=candidates[2],
                rotations=candidates[3],
                opacities=candidates[4],
                camera=camera,
                depthmap=None,
                depth_source="estimated_clean_depth",
                mapper_update_id=23,
                init=False,
                gaussian_before=11,
                preinsert_render_evidence=evidence,
            )
        self.assertEqual(getter.calls, 1)
        self.assertFalse(token.fields["selection_applied"])
        self.assertFalse(token.fields["selected_indices_created"])
        self.assertEqual(token.fields["q_histogram_bins"], 32)
        self.assertEqual(sum(token.fields["q_histogram_counts"]), 4)
        self.assertTrue(token.fields["diagnostic_gpu_to_cpu_sync_expected"])
        self.assertFalse(_contains_tensor(token.fields))
        for value, state in zip(candidates, states):
            pointer, version, shape, dtype, device, frozen = state
            self.assertEqual(value.data_ptr(), pointer)
            self.assertEqual(value._version, version)
            self.assertEqual(tuple(value.shape), shape)
            self.assertEqual(value.dtype, dtype)
            self.assertEqual(value.device, device)
            self.assertTrue(value.requires_grad)
            torch.testing.assert_close(value, frozen, atol=0.0, rtol=0.0)

    def test_dynamic_k_observe_real_mapper_hook_renders_once_and_keeps_all(self):
        camera, model, observer, mapper, events = self.mapper._scene()
        confidence = torch.tensor([[SQRT_2]], device=CUDA_DEVICE)
        getter = _CurrentSnapshotGetter(confidence)
        selector = build_gaussian_candidate_selector_v2(
            {
                "mode": "dynamic_k_observe",
                "strategy": "evidence",
                "dynamic_budget": {
                    "observe_histogram_bins": 32,
                    "k_max_reference": 600,
                },
                "logging": {"enabled": True},
            },
            candidate_selector_v1=None,
            resource_admission_mode="disabled",
            preinsert_render_evidence_v1=observer,
            confidence_snapshot_getter=getter,
            device=CUDA_DEVICE,
        )
        self.assertIsInstance(selector, GaussianCandidateDynamicKObserveV2)
        observe_calls = 0
        render_calls = 0
        summaries = []
        original_observe = selector.observe_before_extend
        original_record = selector.record_after_extend

        def observe_spy(**kwargs):
            nonlocal observe_calls
            observe_calls += 1
            events.append("v2_dynamic_observe")
            return original_observe(**kwargs)

        def record_spy(token, **kwargs):
            events.append("v2_dynamic_record")
            summary = original_record(token, **kwargs)
            summaries.append(summary)
            return summary

        def renderer_spy(*args, **kwargs):
            nonlocal render_calls
            render_calls += 1
            events.append("render")
            return self.mapper.production_render(*args, **kwargs)

        selector.observe_before_extend = observe_spy
        selector.record_after_extend = record_spy
        mapper.candidate_selector_v2 = selector
        before = len(model)
        with mock.patch.object(
            self.mapper.mapper_module,
            "render",
            side_effect=renderer_spy,
        ):
            returned = mapper.add_new_gaussians([camera])
        torch.cuda.synchronize(CUDA_DEVICE)
        self.assertIs(returned, camera)
        self.assertEqual(render_calls, 1)
        self.assertEqual(observe_calls, 1)
        self.assertEqual(getter.calls, 1)
        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertTrue(summary.fields["all_candidates_forwarded"])
        self.assertTrue(summary.fields["conservation_check"])
        self.assertFalse(summary.fields["selection_applied"])
        self.assertEqual(len(model), before + summary.fields["candidate_count"])
        self.assertLess(events.index("render"), events.index("candidate_generation"))
        self.assertLess(
            events.index("candidate_generation"),
            events.index("v2_dynamic_observe"),
        )
        self.assertLess(events.index("v2_dynamic_observe"), events.index("extend"))
        self.assertLess(events.index("extend"), events.index("v2_dynamic_record"))

    def test_active_fixed_k_coverage_cuda_selects_exact_k_and_aligns_five_tensors(self):
        camera, candidates, evidence, confidence = self._active_scene(
            alpha=[0.9, 0.1, 0.8, 0.2, 0.7],
            confidence=[0.0],
        )
        selector, getter = self._active_selector(
            confidence,
            strategy="coverage",
            fixed_k=2,
        )
        frozen = tuple(value.detach().clone() for value in candidates)
        result = self._active_select(selector, candidates, camera, evidence)
        expected = torch.tensor([1, 3], dtype=torch.long, device=CUDA_DEVICE)
        self.assertEqual(getter.calls, 0)
        self.assertEqual(result.selected_indices.device, CUDA_DEVICE)
        self.assertEqual(result.selected_indices.dtype, torch.long)
        self.assertTrue(torch.equal(result.selected_indices, expected))
        self.assertEqual(result.token.fields["selected_count"], 2)
        self.assertEqual(result.token.fields["reason"], "coverage_fixed_k")
        for actual, original, original_frozen in zip(
            (
                result.xyz,
                result.features,
                result.scales,
                result.rotations,
                result.opacities,
            ),
            candidates,
            frozen,
        ):
            self.assertEqual(actual.device, CUDA_DEVICE)
            self.assertEqual(actual.dtype, original.dtype)
            torch.testing.assert_close(
                actual,
                torch.index_select(original_frozen, 0, expected),
                atol=0.0,
                rtol=0.0,
            )
            torch.testing.assert_close(
                original,
                original_frozen,
                atol=0.0,
                rtol=0.0,
            )

    def test_active_fixed_k_evidence_cuda_matches_cpu_reference_and_alpha_continuity(self):
        camera, candidates, evidence, confidence = self._active_scene(
            alpha=[0.5, 0.5, 0.5, 0.5],
            confidence=[SQRT_2, SQRT_2, SQRT_2, SQRT_2],
            rgb_l1=[0.0, 0.0, 1.0, 0.0],
            old_depth=[2.0, 4.0, 2.0, 2.0],
        )
        selector, _ = self._active_selector(
            confidence,
            strategy="evidence",
            fixed_k=2,
        )
        projection = selector._active_project(candidates[0], camera)
        render_components, evidence_reason = selector._active_render_components(
            xyz=candidates[0],
            camera=camera,
            gaussian_before=11,
            mapper_update_id=23,
            evidence=evidence,
            projection=projection,
        )
        confidence_components = selector._active_confidence_components(
            xyz=candidates[0],
            camera=camera,
            depthmap=None,
            depth_source="estimated_clean_depth",
            projection=projection,
        )
        quality, valid, rgb_valid, depth_valid = (
            selector._weighted_evidence_quality(
                render_components,
                confidence_components,
            )
        )
        self.assertEqual(evidence_reason, "available")
        self.assertTrue(bool(valid.all()))
        self.assertTrue(bool(rgb_valid.all()))
        self.assertTrue(bool(depth_valid.all()))

        alpha_cpu = torch.full((4,), 0.5, dtype=torch.float32)
        confidence_cpu = torch.ones(4, dtype=torch.float32)
        coverage_cpu = 1.0 - alpha_cpu
        rgb_cpu = torch.tensor([0.0, 0.0, 1.0, 0.0])
        depth_relative_cpu = torch.tensor([0.0, 1.0, 0.0, 0.0])
        depth_saturated_cpu = depth_relative_cpu / (1.0 + depth_relative_cpu)
        reference_quality = (
            coverage_cpu
            + alpha_cpu * confidence_cpu * rgb_cpu
            + alpha_cpu * confidence_cpu * depth_saturated_cpu
        ) / (1.0 + alpha_cpu + alpha_cpu)
        torch.testing.assert_close(
            quality.cpu(),
            reference_quality,
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        expected = torch.sort(
            torch.argsort(-reference_quality, stable=True)[:2]
        ).values.to(device=CUDA_DEVICE)
        result = self._active_select(selector, candidates, camera, evidence)
        self.assertTrue(torch.equal(result.selected_indices, expected))
        self.assertTrue(
            torch.equal(
                result.selected_indices,
                torch.tensor([1, 2], dtype=torch.long, device=CUDA_DEVICE),
            )
        )

        alpha = torch.tensor(
            [
                0.0,
                NUMERICAL_EPSILON / 2.0,
                NUMERICAL_EPSILON,
                NUMERICAL_EPSILON * 2.0,
                0.8,
            ],
            dtype=torch.float32,
            device=CUDA_DEVICE,
        )
        count = int(alpha.shape[0])
        continuous_quality, continuous_valid, _, _ = (
            selector._weighted_evidence_quality(
                {
                    "alpha": alpha,
                    "coverage": 1.0 - alpha,
                    "coverage_valid": torch.ones(
                        count, dtype=torch.bool, device=CUDA_DEVICE
                    ),
                    "rgb_l1": torch.ones_like(alpha),
                    "rgb_valid": torch.ones(
                        count, dtype=torch.bool, device=CUDA_DEVICE
                    ),
                    "depth_saturated": torch.zeros_like(alpha),
                    "depth_valid": torch.zeros(
                        count, dtype=torch.bool, device=CUDA_DEVICE
                    ),
                },
                {
                    "normalized": torch.ones_like(alpha),
                    "valid": torch.ones(
                        count, dtype=torch.bool, device=CUDA_DEVICE
                    ),
                },
            )
        )
        torch.testing.assert_close(
            continuous_quality.cpu(),
            torch.reciprocal(1.0 + alpha.cpu()),
            atol=1.0e-7,
            rtol=1.0e-7,
        )
        self.assertTrue(bool(continuous_valid.all()))
        self.assertEqual(continuous_quality[0].item(), 1.0)
        self.assertLess(
            abs(continuous_quality[1].item() - continuous_quality[0].item()),
            NUMERICAL_EPSILON,
        )

    def test_active_fixed_k_evidence_projected_cell_cuda_executes_grouping_and_ties(self):
        camera, candidates, evidence, confidence = self._active_scene(
            alpha=[0.10, 0.102, 0.11, 1.0],
            confidence=[0.0, SQRT_2],
        )
        evidence_selector, _ = self._active_selector(
            confidence,
            strategy="evidence",
            fixed_k=2,
        )
        projected_selector, _ = self._active_selector(
            confidence,
            strategy="evidence_projected_cell",
            fixed_k=2,
        )
        evidence_result = self._active_select(
            evidence_selector, candidates, camera, evidence
        )
        real_argsort = torch.argsort
        real_cummax = torch.cummax
        with mock.patch.object(torch, "argsort", side_effect=real_argsort) as argsort_spy, \
             mock.patch.object(torch, "cummax", side_effect=real_cummax) as cummax_spy:
            projected_result = self._active_select(
                projected_selector, candidates, camera, evidence
            )
        self.assertGreaterEqual(argsort_spy.call_count, 3)
        self.assertTrue(
            any(call.kwargs.get("stable") is True for call in argsort_spy.call_args_list)
        )
        self.assertEqual(cummax_spy.call_count, 1)
        self.assertTrue(
            torch.equal(
                evidence_result.selected_indices,
                torch.tensor([0, 1], dtype=torch.long, device=CUDA_DEVICE),
            )
        )
        self.assertTrue(
            torch.equal(
                projected_result.selected_indices,
                torch.tensor([0, 2], dtype=torch.long, device=CUDA_DEVICE),
            )
        )
        self.assertEqual(
            projected_result.token.fields["selected_unique_cell_count"], 2
        )

        tie_camera, tie_candidates, tie_evidence, tie_confidence = self._active_scene(
            alpha=[0.5, 0.5, 0.5, 0.5],
            confidence=[SQRT_2, SQRT_2],
        )
        tie_selector, _ = self._active_selector(
            tie_confidence,
            strategy="evidence_projected_cell",
            fixed_k=1,
        )
        tie_result = self._active_select(
            tie_selector,
            tie_candidates,
            tie_camera,
            tie_evidence,
        )
        self.assertTrue(
            torch.equal(
                tie_result.selected_indices,
                torch.tensor([0], dtype=torch.long, device=CUDA_DEVICE),
            )
        )

    def test_active_fixed_k_cuda_is_deterministic_across_twenty_runs(self):
        camera, candidates, evidence, confidence = self._active_scene(
            alpha=[0.10, 0.102, 0.11, 1.0],
            confidence=[0.0, SQRT_2],
        )
        selector, _ = self._active_selector(
            confidence,
            strategy="evidence_projected_cell",
            fixed_k=2,
        )
        reference = None
        for _ in range(20):
            result = self._active_select(selector, candidates, camera, evidence)
            if reference is None:
                reference = result.selected_indices.detach().clone()
            else:
                self.assertTrue(torch.equal(result.selected_indices, reference))
        self.assertTrue(
            torch.equal(
                reference,
                torch.tensor([0, 2], dtype=torch.long, device=CUDA_DEVICE),
            )
        )

    def test_active_fixed_k_cuda_preserves_inputs_versions_and_autograd(self):
        camera, base_candidates, evidence, confidence = self._active_scene(
            alpha=[0.9, 0.1, 0.8, 0.2, 0.7],
            confidence=[0.0],
        )
        candidates = tuple(
            value.detach().clone().requires_grad_(True) for value in base_candidates
        )
        states = tuple(
            (
                value.data_ptr(),
                value._version,
                tuple(value.shape),
                value.dtype,
                value.device,
                value.detach().clone(),
            )
            for value in candidates
        )
        selector, _ = self._active_selector(
            confidence,
            strategy="coverage",
            fixed_k=2,
        )
        result = self._active_select(selector, candidates, camera, evidence)
        outputs = (
            result.xyz,
            result.features,
            result.scales,
            result.rotations,
            result.opacities,
        )
        for source, output, state in zip(candidates, outputs, states):
            pointer, version, shape, dtype, device, values = state
            self.assertEqual(source.data_ptr(), pointer)
            self.assertEqual(source._version, version)
            self.assertEqual(tuple(source.shape), shape)
            self.assertEqual(source.dtype, dtype)
            self.assertEqual(source.device, device)
            torch.testing.assert_close(source, values, atol=0.0, rtol=0.0)
            self.assertEqual(output.dtype, dtype)
            self.assertEqual(output.device, device)
            self.assertTrue(output.requires_grad)
            self.assertIsNotNone(output.grad_fn)
        sum(value.sum() for value in outputs).backward()
        for source in candidates:
            self.assertIsNotNone(source.grad)
            self.assertTrue(bool(torch.isfinite(source.grad).all()))

    def test_active_fixed_k_cuda_fallbacks_remain_bounded(self):
        camera, candidates, evidence, confidence = self._active_scene(
            alpha=[0.9, 0.1, 0.8, 0.2, 0.7],
            confidence=[0.0],
        )
        missing = RuntimeError("confidence is not valid")
        coverage_selector, _ = self._active_selector(
            confidence,
            strategy="evidence",
            fixed_k=2,
            confidence_error=missing,
        )
        coverage_result = self._active_select(
            coverage_selector, candidates, camera, evidence
        )
        self.assertEqual(
            coverage_result.token.fields["reason"], "coverage_only_fallback"
        )
        self.assertTrue(
            torch.equal(
                coverage_result.selected_indices,
                torch.tensor([1, 3], dtype=torch.long, device=CUDA_DEVICE),
            )
        )
        self.assertEqual(coverage_result.selected_indices.numel(), 2)

        unavailable = PreinsertRenderEvidenceObserverV1(
            device=CUDA_DEVICE
        ).unavailable(
            camera=camera,
            gaussian_count_before_render=11,
            mapper_update_id=23,
            reason="empty_old_map",
        )
        uniform_selector, _ = self._active_selector(
            confidence,
            strategy="evidence",
            fixed_k=2,
            confidence_error=RuntimeError("confidence is not valid"),
        )
        uniform_result = self._active_select(
            uniform_selector, candidates, camera, unavailable
        )
        self.assertEqual(
            uniform_result.token.fields["reason"],
            "deterministic_uniform_fallback",
        )
        self.assertTrue(
            torch.equal(
                uniform_result.selected_indices,
                torch.tensor([0, 4], dtype=torch.long, device=CUDA_DEVICE),
            )
        )
        self.assertEqual(uniform_result.selected_indices.numel(), 2)
        self.assertLessEqual(
            uniform_result.token.fields["selected_count"],
            uniform_selector.fixed_k,
        )

    def test_active_fixed_k_factory_and_real_mapper_hook_select_once_without_second_render(self):
        camera, model, observer, mapper, events = self.mapper._scene()
        confidence = torch.tensor([[SQRT_2]], device=CUDA_DEVICE)
        getter = _CurrentSnapshotGetter(confidence)
        selector = build_gaussian_candidate_selector_v2(
            {
                "mode": "active_fixed_k",
                "strategy": "coverage",
                "fixed_k": 2,
                "diversity_strength": 0.1,
                "logging": {"enabled": True},
            },
            candidate_selector_v1=None,
            resource_admission_mode="disabled",
            preinsert_render_evidence_v1=observer,
            confidence_snapshot_getter=getter,
            device=CUDA_DEVICE,
        )
        self.assertIsInstance(selector, GaussianCandidateActiveFixedKV2)
        self.assertEqual(selector.mode, "active_fixed_k")
        self.assertTrue(selector.is_active)
        self.assertIsNone(mapper.candidate_selector_v1)
        self.assertIsNone(model.resource_admission)
        self.assertEqual(observer.mode, "observe")

        select_calls = 0
        render_calls = 0
        summaries = []
        original_select = selector.select_before_extend
        original_record = selector.record_active_after_extend

        def select_spy(**kwargs):
            nonlocal select_calls
            select_calls += 1
            events.append("v2_active_select")
            return original_select(**kwargs)

        def record_spy(token, **kwargs):
            events.append("v2_active_record")
            summary = original_record(token, **kwargs)
            summaries.append(summary)
            return summary

        def renderer_spy(*args, **kwargs):
            nonlocal render_calls
            render_calls += 1
            events.append("render")
            return self.mapper.production_render(*args, **kwargs)

        selector.select_before_extend = select_spy
        selector.record_active_after_extend = record_spy
        mapper.candidate_selector_v2 = selector
        before = len(model)
        with mock.patch.object(
            self.mapper.mapper_module,
            "render",
            side_effect=renderer_spy,
        ):
            returned = mapper.add_new_gaussians([camera])
        torch.cuda.synchronize(CUDA_DEVICE)
        self.assertIs(returned, camera)
        self.assertEqual(render_calls, 1)
        self.assertEqual(select_calls, 1)
        self.assertEqual(getter.calls, 0)
        self.assertEqual(
            events,
            [
                "render",
                "candidate_generation",
                "v2_active_select",
                "extend",
                "v2_active_record",
            ],
        )
        self.assertEqual(len(model), before + 2)
        self.assertEqual(len(summaries), 1)
        self.assertTrue(summaries[0].fields["conservation_check"])
        self.assertFalse(
            summaries[0].fields["m01_second_selection_applied"]
        )

    def test_exact_off_and_conservation_failure_remain_fail_closed(self):
        trap = object()
        self.assertIsNone(
            build_gaussian_candidate_selector_v2(
                {"mode": "off", "logging": {"enabled": True}},
                candidate_selector_v1=trap,
                resource_admission_mode="fixed_budget",
                preinsert_render_evidence_v1=trap,
                confidence_snapshot_getter=trap,
                device="not-a-device",
            )
        )
        camera, model, _, mapper, events = self.mapper._scene()
        mapper.preinsert_render_evidence_v1 = None
        mapper.candidate_selector_v2 = None
        generation_flags = []
        controlled_generation = model.create_pcd_from_image

        def generation_spy(
            model_self, *args, collect_candidate_metadata=False, **kwargs
        ):
            generation_flags.append(bool(collect_candidate_metadata))
            return controlled_generation(
                *args,
                collect_candidate_metadata=collect_candidate_metadata,
                **kwargs,
            )

        model.create_pcd_from_image = MethodType(generation_spy, model)
        before = len(model)
        mapper.add_new_gaussians([camera])
        self.assertEqual(generation_flags, [False])
        self.assertEqual(events, ["candidate_generation", "extend"])
        self.assertEqual(len(model), before + 6)

        selector, _ = self._selector(torch.tensor([[SQRT_2]], device=CUDA_DEVICE))
        evidence = PreinsertRenderEvidenceObserverV1(
            device=CUDA_DEVICE
        ).unavailable(
            camera=camera,
            gaussian_count_before_render=len(model),
            mapper_update_id=23,
            reason="empty_old_map",
        )
        candidates = self._candidates(torch.tensor([[0.0, 0.0, 2.0]],
                                                    device=CUDA_DEVICE))
        token = self._observe(
            selector, candidates, camera, evidence, gaussian_before=len(model)
        )
        summary = selector.record_after_extend(
            token,
            admitted_candidate_count=1,
            dropped_candidate_count=0,
            gaussian_after_extend=len(model),
        )
        self.assertEqual(summary.fields["status"], "error")
        self.assertEqual(
            summary.fields["reason"], "observe_only_forwarding_contract_failed"
        )
        self.assertFalse(summary.fields["conservation_check"])
        self.assertEqual(summary.fields["error"]["type"], "ForwardingContractError")

    def test_active_dynamic_k_cuda_histogram_selects_256_and_600_boundaries(self):
        cases = (
            ([0.0] * 100 + [1.0] * 600, 100, 256),
            ([0.0] * 700, 700, 600),
        )
        for alpha, demand, target in cases:
            with self.subTest(target=target):
                camera, candidates, evidence, confidence = self._active_scene(
                    alpha=alpha,
                    confidence=[SQRT_2] * 700,
                )
                selector, _ = self._dynamic_active_selector(confidence)
                result = self._active_select(selector, candidates, camera, evidence)
                fields = result.token.fields
                self.assertEqual(fields["demand_count"], demand)
                self.assertEqual(fields["target_k"], target)
                self.assertEqual(fields["selected_count"], target)
                self.assertEqual(result.selected_indices.device, CUDA_DEVICE)
                self.assertEqual(result.selected_indices.dtype, torch.long)
                self.assertTrue(
                    torch.equal(
                        result.selected_indices,
                        torch.arange(target, device=CUDA_DEVICE),
                    )
                )
                self.assertEqual(sum(fields["q_histogram_counts"]), 700)

    def test_active_dynamic_k_cuda_is_deterministic_across_twenty_runs(self):
        camera, candidates, evidence, confidence = self._active_scene(
            alpha=[0.5] * 300,
            confidence=[SQRT_2] * 300,
        )
        selector, _ = self._dynamic_active_selector(confidence)
        reference = None
        for _ in range(20):
            result = self._active_select(selector, candidates, camera, evidence)
            if reference is None:
                reference = result.selected_indices.detach().clone()
            else:
                self.assertTrue(torch.equal(result.selected_indices, reference))
        self.assertTrue(
            torch.equal(reference, torch.arange(256, device=CUDA_DEVICE))
        )

    def test_active_dynamic_k_cuda_preserves_inputs_versions_and_autograd(self):
        camera, base_candidates, evidence, confidence = self._active_scene(
            alpha=[0.0] * 300,
            confidence=[SQRT_2] * 300,
        )
        candidates = tuple(
            value.detach().clone().requires_grad_(True) for value in base_candidates
        )
        states = tuple(
            (value.data_ptr(), value._version, value.detach().clone())
            for value in candidates
        )
        selector, _ = self._dynamic_active_selector(confidence)
        result = self._active_select(selector, candidates, camera, evidence)
        outputs = (
            result.xyz, result.features, result.scales,
            result.rotations, result.opacities,
        )
        for source, output, state in zip(candidates, outputs, states):
            pointer, version, frozen = state
            self.assertEqual(source.data_ptr(), pointer)
            self.assertEqual(source._version, version)
            torch.testing.assert_close(source, frozen, atol=0.0, rtol=0.0)
            self.assertEqual(output.device, CUDA_DEVICE)
            self.assertTrue(output.requires_grad)
            self.assertIsNotNone(output.grad_fn)

    def test_active_dynamic_k_cuda_real_mapper_hook_selects_once_and_renders_once(self):
        camera, model, observer, mapper, events = self.mapper._scene()
        confidence = torch.tensor([[SQRT_2]], device=CUDA_DEVICE)
        selector = build_gaussian_candidate_selector_v2(
            {
                "mode": "active_dynamic_k",
                "strategy": "evidence",
                "dynamic_budget": {
                    "observe_histogram_bins": 4,
                    "k_max_reference": 600,
                    "threshold_bin": 1,
                    "k_min": 2,
                    "k_max": 3,
                },
                "logging": {"enabled": True},
            },
            candidate_selector_v1=None,
            resource_admission_mode="disabled",
            preinsert_render_evidence_v1=observer,
            confidence_snapshot_getter=_CurrentSnapshotGetter(confidence),
            device=CUDA_DEVICE,
        )
        mapper.candidate_selector_v2 = selector
        select_calls = render_calls = 0
        summaries = []
        original_select = selector.select_before_extend
        original_record = selector.record_active_after_extend

        def select_spy(**kwargs):
            nonlocal select_calls
            select_calls += 1
            events.append("v2_dynamic_select")
            return original_select(**kwargs)

        def record_spy(token, **kwargs):
            summary = original_record(token, **kwargs)
            summaries.append(summary)
            events.append("v2_dynamic_record")
            return summary

        def renderer_spy(*args, **kwargs):
            nonlocal render_calls
            render_calls += 1
            events.append("render")
            return self.mapper.production_render(*args, **kwargs)

        selector.select_before_extend = select_spy
        selector.record_active_after_extend = record_spy
        before = len(model)
        with mock.patch.object(
            self.mapper.mapper_module, "render", side_effect=renderer_spy
        ):
            mapper.add_new_gaussians([camera])
        self.assertEqual(render_calls, 1)
        self.assertEqual(select_calls, 1)
        self.assertEqual(len(summaries), 1)
        self.assertTrue(summaries[0].fields["conservation_check"])
        self.assertEqual(
            len(model), before + summaries[0].fields["selected_count"]
        )
        self.assertLess(events.index("render"), events.index("candidate_generation"))
        self.assertLess(
            events.index("candidate_generation"), events.index("v2_dynamic_select")
        )
        self.assertLess(events.index("v2_dynamic_select"), events.index("extend"))

    def test_active_dynamic_k_cuda_source_has_no_extra_render_or_full_q_copy(self):
        source = inspect.getsource(GaussianCandidateActiveDynamicKV2)
        for forbidden in (
            "quality.cpu", ".tolist(", ".numpy(", "torch.quantile",
            "cuda.synchronize", "gaussian_renderer", "render(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertEqual(
            inspect.getsource(GaussianCandidateDynamicBudgetObserverV1.observe).count(
                ".cpu()"
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
