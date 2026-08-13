"""Four no-dataset CUDA gates for V2-A; a valid server run has zero skips."""

from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
from unittest import mock

import torch

from src.candidate_selection.gaussian_candidate_selector_v2 import (
    NUMERICAL_EPSILON,
    GaussianCandidateSelectorV2,
    build_gaussian_candidate_selector_v2,
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
    def __init__(self, confidence):
        self.confidence = confidence
        self.calls = 0

    def __call__(self, camera, require_current=True, *, upsampled=None):
        self.calls += 1
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
        source = (Path(__file__).resolve().parents[1] /
                  "src/candidate_selection/gaussian_candidate_selector_v2.py").read_text(
                      encoding="utf-8"
                  )
        self.assertNotIn("index_select", source)
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


if __name__ == "__main__":
    unittest.main()
