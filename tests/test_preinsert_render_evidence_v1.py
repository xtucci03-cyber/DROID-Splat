from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from src.candidate_selection.gaussian_candidate_active_topk_v1 import (
    ACTIVE_BUDGET,
    GaussianCandidateActiveTopKV1,
)
from src.candidate_selection.preinsert_render_evidence_v1 import (
    EVIDENCE_SEMANTICS_VERSION,
    RASTERIZER_BINARY_PROVENANCE,
    RASTERIZER_SOURCE_GITLINK,
    PreinsertRenderEvidenceError,
    PreinsertRenderEvidenceObserverV1,
    PreinsertRenderEvidenceV1,
    build_preinsert_render_evidence_v1,
)
from tests.test_gaussian_candidate_active_topk_v1 import (
    GaussianModelHookHarness,
    load_gaussian_model_extend_from_pcd_seq,
    make_active_fixture,
)


ROOT = Path(__file__).resolve().parents[1]


class EvidenceCamera:
    def __init__(self, uid: int = 17, *, height: int = 4, width: int = 6):
        self.uid = uid
        self.buffer_index = uid + 100
        self.source_frame_id = uid + 1000
        self.source_timestamp = float(uid) + 0.25
        self.image_height = height
        self.image_width = width
        self.device = "cpu"


class FakeGaussians:
    def __init__(self, count: int, events=None, insert_count: int = 7):
        self.count = count
        self.events = [] if events is None else events
        self.insert_count = insert_count
        self.extend_calls = 0
        self.generation_calls = 0
        self.received_evidence = []

    def __len__(self):
        return self.count

    def extend_from_pcd_seq(self, camera, kf_id, *, init, **kwargs):
        self.events.append(("generate", camera.uid, self.count))
        self.generation_calls += 1
        self.events.append(("select", camera.uid, self.count))
        self.received_evidence.append(kwargs.get("preinsert_render_evidence"))
        self.events.append(("extend", camera.uid, self.count))
        self.extend_calls += 1
        self.count += self.insert_count


class RecordingEvidenceObserver:
    def __init__(self, events):
        self.events = events
        self.gaussian_counts = []

    def unavailable(self, *, camera, gaussian_count_before_render, mapper_update_id, reason):
        self.events.append(("unavailable", camera.uid, gaussian_count_before_render))
        return SimpleNamespace(
            available=False,
            source_camera_id=camera.uid,
            gaussian_count_before_render=gaussian_count_before_render,
        )

    def capture(
        self,
        *,
        camera,
        gaussians,
        renderer,
        pipeline_params,
        background,
        mapper_update_id,
    ):
        self.gaussian_counts.append(len(gaussians))
        renderer(camera, gaussians, pipeline_params, background, device="cpu")
        return SimpleNamespace(
            available=True,
            source_camera_id=camera.uid,
            gaussian_count_before_render=len(gaussians),
        )


def load_mapper_candidate_event_methods(renderer):
    source_path = ROOT / "src/gaussian_mapping.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GaussianMapper"
    )
    wanted = {
        "_capture_preinsert_render_evidence",
        "add_new_gaussians",
    }
    methods = [
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *methods,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"render": renderer}
    exec(compile(module, str(source_path), "exec"), namespace)
    return {name: namespace[name] for name in wanted}


class MapperHarness:
    def __init__(self, *, observer, renderer, initialized=True, gaussian_count=10):
        methods = load_mapper_candidate_event_methods(renderer)
        self._capture_preinsert_render_evidence = methods[
            "_capture_preinsert_render_evidence"
        ].__get__(self)
        self.add_new_gaussians = methods["add_new_gaussians"].__get__(self)
        self.preinsert_render_evidence_v1 = observer
        self.candidate_observer = None
        self.candidate_selector = None
        self.candidate_selector_v1 = None
        self.resource_admission = None
        events = observer.events if observer is not None else []
        self.gaussians = FakeGaussians(gaussian_count, events=events)
        self.pipeline_params = object()
        self.background = torch.zeros(3)
        self.count = 23
        self.initialized = initialized
        self.info_messages = []

    def info(self, message):
        self.info_messages.append(message)


def make_evidence(camera, *, count=11, mapper_update_id=23, fill=(1.0, 2.0, 3.0)):
    height = camera.image_height
    width = camera.image_width
    return PreinsertRenderEvidenceV1(
        render_rgb=torch.full((3, height, width), fill[0]),
        depth_accum=torch.full((1, height, width), fill[1]),
        alpha_accum=torch.full((1, height, width), fill[2]),
        source_camera_id=camera.uid,
        source_frame_id=camera.source_frame_id,
        source_buffer_index=camera.buffer_index,
        source_timestamp=camera.source_timestamp,
        height=height,
        width=width,
        gaussian_count_before_render=count,
        available=True,
        unavailable_reason=None,
        mapper_update_id=mapper_update_id,
    )


class PreinsertRenderEvidenceContractTests(unittest.TestCase):
    def test_factory_default_off_and_illegal_mode_fail_closed(self):
        self.assertIsNone(build_preinsert_render_evidence_v1(None, device="cpu"))
        self.assertIsNone(
            build_preinsert_render_evidence_v1({"mode": "off"}, device="cpu")
        )
        observer = build_preinsert_render_evidence_v1(
            {"mode": "observe"}, device="cpu"
        )
        self.assertIsInstance(observer, PreinsertRenderEvidenceObserverV1)
        with self.assertRaises(ValueError):
            build_preinsert_render_evidence_v1({"mode": "active"}, device="cpu")
        with self.assertRaises(ValueError):
            build_preinsert_render_evidence_v1(
                {"mode": "observe", "logging": True}, device="cpu"
            )
        config = (ROOT / "configs/mapping/base.yaml").read_text(encoding="utf-8")
        block = config.split("preinsert_render_evidence_v1:", 1)[1]
        self.assertEqual(block.lstrip().splitlines()[0].strip(), 'mode: "off"')

    def test_capture_is_one_no_grad_render_without_clone_or_cpu_copy(self):
        camera = EvidenceCamera()
        observer = PreinsertRenderEvidenceObserverV1(device="cpu")
        gaussians = FakeGaussians(11)
        package = {}
        calls = []

        def renderer(view, model, pipeline, background, *, device):
            calls.append((view, model, torch.is_grad_enabled(), device))
            package.update(
                {
                    "render": torch.full((3, 4, 6), 1.0),
                    "depth": torch.full((1, 4, 6), 2.0),
                    "opacity": torch.full((1, 4, 6), 3.0),
                }
            )
            return package

        evidence = observer.capture(
            camera=camera,
            gaussians=gaussians,
            renderer=renderer,
            pipeline_params=object(),
            background=torch.zeros(3),
            mapper_update_id=23,
        )
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], camera)
        self.assertIs(calls[0][1], gaussians)
        self.assertFalse(calls[0][2])
        self.assertEqual(calls[0][3], "cpu")
        self.assertTrue(evidence.available)
        self.assertEqual(evidence.render_rgb.data_ptr(), package["render"].data_ptr())
        self.assertEqual(evidence.depth_accum.data_ptr(), package["depth"].data_ptr())
        self.assertEqual(evidence.alpha_accum.data_ptr(), package["opacity"].data_ptr())
        self.assertEqual(float(evidence.depth_accum[0, 0, 0]), 2.0)
        self.assertEqual(float(evidence.alpha_accum[0, 0, 0]), 3.0)
        self.assertFalse(evidence.render_rgb.requires_grad)
        self.assertIsNone(evidence.render_rgb.grad_fn)
        self.assertEqual(evidence.evidence_semantics_version, EVIDENCE_SEMANTICS_VERSION)
        self.assertEqual(evidence.rasterizer_source_gitlink, RASTERIZER_SOURCE_GITLINK)
        self.assertEqual(
            evidence.rasterizer_binary_provenance,
            RASTERIZER_BINARY_PROVENANCE,
        )

    def test_empty_old_map_is_unavailable_without_calling_renderer(self):
        observer = PreinsertRenderEvidenceObserverV1(device="cpu")
        camera = EvidenceCamera()

        def renderer(*args, **kwargs):
            raise AssertionError("renderer must not be called for an empty map")

        evidence = observer.capture(
            camera=camera,
            gaussians=FakeGaussians(0),
            renderer=renderer,
            pipeline_params=object(),
            background=torch.zeros(3),
            mapper_update_id=23,
        )
        self.assertFalse(evidence.available)
        self.assertEqual(evidence.unavailable_reason, "empty_old_map")
        self.assertIsNone(evidence.render_rgb)
        self.assertIsNone(evidence.depth_accum)
        self.assertIsNone(evidence.alpha_accum)

    def test_missing_render_field_fails_instead_of_fabricating_evidence(self):
        observer = PreinsertRenderEvidenceObserverV1(device="cpu")
        camera = EvidenceCamera()

        def renderer(*args, **kwargs):
            return {
                "render": torch.zeros(3, 4, 6),
                "depth": torch.zeros(1, 4, 6),
            }

        with self.assertRaisesRegex(
            PreinsertRenderEvidenceError,
            "missing fields",
        ):
            observer.capture(
                camera=camera,
                gaussians=FakeGaussians(1),
                renderer=renderer,
                pipeline_params=object(),
                background=torch.zeros(3),
                mapper_update_id=23,
            )

    def test_shape_camera_identity_and_stale_count_fail_closed(self):
        camera = EvidenceCamera()
        evidence = make_evidence(camera)
        evidence.validate_for_event(
            camera=camera,
            gaussian_count_current=11,
            mapper_update_id=23,
        )
        wrong_camera = EvidenceCamera(uid=18)
        with self.assertRaisesRegex(
            PreinsertRenderEvidenceError, "identity mismatch"
        ):
            evidence.validate_for_event(
                camera=wrong_camera,
                gaussian_count_current=11,
                mapper_update_id=23,
            )
        with self.assertRaisesRegex(PreinsertRenderEvidenceError, "stale"):
            evidence.validate_for_event(
                camera=camera,
                gaussian_count_current=12,
                mapper_update_id=23,
            )
        bad_shape = PreinsertRenderEvidenceV1(
            **{
                **evidence.__dict__,
                "depth_accum": torch.ones(1, 3, 6),
            }
        )
        with self.assertRaisesRegex(PreinsertRenderEvidenceError, "shape mismatch"):
            bad_shape.validate_for_event(
                camera=camera,
                gaussian_count_current=11,
                mapper_update_id=23,
            )

    def test_unavailable_evidence_cannot_disguise_zero_tensors(self):
        camera = EvidenceCamera()
        evidence = PreinsertRenderEvidenceV1(
            **{
                **make_evidence(camera).__dict__,
                "available": False,
                "unavailable_reason": "empty_old_map",
            }
        )
        with self.assertRaisesRegex(
            PreinsertRenderEvidenceError,
            "must not contain render tensors",
        ):
            evidence.validate_for_event(
                camera=camera,
                gaussian_count_current=11,
                mapper_update_id=23,
            )

    def test_production_module_contains_no_cpu_or_scalar_sync_calls(self):
        source = (
            ROOT / "src/candidate_selection/preinsert_render_evidence_v1.py"
        ).read_text(encoding="utf-8")
        for forbidden in (".cpu(", ".item(", ".tolist(", ".numpy("):
            self.assertNotIn(forbidden, source)


class MapperPreinsertRenderOrderingTests(unittest.TestCase):
    def test_off_mode_calls_no_renderer_and_extends_once(self):
        def renderer(*args, **kwargs):
            raise AssertionError("off mode must not render")

        harness = MapperHarness(observer=None, renderer=renderer)
        camera = EvidenceCamera()
        harness.add_new_gaussians([camera])
        self.assertEqual(harness.gaussians.extend_calls, 1)
        self.assertEqual(harness.gaussians.received_evidence, [None])

    def test_init_uses_unavailable_evidence_without_render_and_keeps_all(self):
        events = []
        observer = RecordingEvidenceObserver(events)

        def renderer(*args, **kwargs):
            raise AssertionError("init must not render")

        harness = MapperHarness(
            observer=observer,
            renderer=renderer,
            initialized=False,
            gaussian_count=0,
        )
        camera = EvidenceCamera()
        harness.add_new_gaussians([camera])
        self.assertEqual(
            events,
            [
                ("unavailable", camera.uid, 0),
                ("generate", camera.uid, 0),
                ("select", camera.uid, 0),
                ("extend", camera.uid, 0),
            ],
        )
        self.assertEqual(harness.gaussians.count, 7)

    def test_multi_camera_is_render_select_extend_then_next_render(self):
        events = []
        observer = RecordingEvidenceObserver(events)

        def renderer(camera, gaussians, pipeline, background, *, device):
            events.append(("render", camera.uid, len(gaussians)))
            return {
                "render": torch.zeros(3, 4, 6),
                "depth": torch.zeros(1, 4, 6),
                "opacity": torch.zeros(1, 4, 6),
            }

        harness = MapperHarness(observer=observer, renderer=renderer)
        cameras = [EvidenceCamera(uid=17), EvidenceCamera(uid=18)]
        harness.add_new_gaussians(cameras)
        self.assertEqual(
            events,
            [
                ("render", 17, 10),
                ("generate", 17, 10),
                ("select", 17, 10),
                ("extend", 17, 10),
                ("render", 18, 17),
                ("generate", 18, 17),
                ("select", 18, 17),
                ("extend", 18, 17),
            ],
        )
        self.assertEqual(observer.gaussian_counts, [10, 17])
        self.assertEqual(harness.gaussians.generation_calls, 2)
        self.assertEqual(harness.gaussians.extend_calls, 2)

    def test_mapper_never_persists_event_render_evidence(self):
        source = (ROOT / "src/gaussian_mapping.py").read_text(encoding="utf-8")
        self.assertNotIn("self.preinsert_render_evidence =", source)
        self.assertIn("preinsert_render_evidence = None", source)


class ActivePreinsertEvidenceEquivalenceTests(unittest.TestCase):
    @staticmethod
    def _select(selector, camera, candidates, *, init=False, evidence=None):
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
            preinsert_render_evidence=evidence,
        )

    def test_evidence_context_does_not_change_selected_indices_or_outputs(self):
        selector_off, _, camera_off, candidates_off = make_active_fixture(603)
        selector_on, _, camera_on, candidates_on = make_active_fixture(603)
        result_off = selector_off.select_before_extend(
            xyz=candidates_off[0],
            features=candidates_off[1],
            scales=candidates_off[2],
            rotations=candidates_off[3],
            opacities=candidates_off[4],
            camera=camera_off,
            depthmap=None,
            depth_source="estimated_clean_depth",
            mapper_update_id=23,
            init=False,
            gaussian_before=11,
        )
        result_on = selector_on.select_before_extend(
            xyz=candidates_on[0],
            features=candidates_on[1],
            scales=candidates_on[2],
            rotations=candidates_on[3],
            opacities=candidates_on[4],
            camera=camera_on,
            depthmap=None,
            depth_source="estimated_clean_depth",
            mapper_update_id=23,
            init=False,
            gaussian_before=11,
            preinsert_render_evidence=object(),
        )
        self.assertTrue(
            torch.equal(result_off.selected_indices, result_on.selected_indices)
        )
        for field in ("xyz", "features", "scales", "rotations", "opacities"):
            self.assertTrue(
                torch.equal(getattr(result_off, field), getattr(result_on, field))
            )

    def test_context_is_equivalent_for_all_frozen_active_paths(self):
        cases = (
            ("init", 603, None, True, None),
            ("n_less_than_k", 599, None, False, None),
            ("n_equal_k", 600, None, False, None),
            (
                "all_equal_ties",
                603,
                torch.ones(603, dtype=torch.float32),
                False,
                None,
            ),
            (
                "nonfinite_fallback",
                603,
                torch.cat(
                    (
                        torch.tensor([float("nan")]),
                        torch.arange(602, dtype=torch.float32),
                    )
                ),
                False,
                "fixed600_fallback_applied",
            ),
        )
        for name, count, scores, init, expected_reason in cases:
            with self.subTest(name=name):
                selector_off, _, camera_off, candidates_off = make_active_fixture(
                    count,
                    scores=scores,
                )
                selector_on, _, camera_on, candidates_on = make_active_fixture(
                    count,
                    scores=scores,
                )
                result_off = self._select(
                    selector_off,
                    camera_off,
                    candidates_off,
                    init=init,
                )
                result_on = self._select(
                    selector_on,
                    camera_on,
                    candidates_on,
                    init=init,
                    evidence=object(),
                )
                if result_off.selected_indices is None:
                    self.assertIsNone(result_on.selected_indices)
                else:
                    self.assertTrue(
                        torch.equal(
                            result_off.selected_indices,
                            result_on.selected_indices,
                        )
                    )
                if expected_reason is not None:
                    self.assertEqual(
                        result_off.token.fields["reason"], expected_reason
                    )
                    self.assertEqual(
                        result_on.token.fields["reason"], expected_reason
                    )
                for field in (
                    "xyz",
                    "features",
                    "scales",
                    "rotations",
                    "opacities",
                ):
                    off_value = getattr(result_off, field)
                    on_value = getattr(result_on, field)
                    self.assertEqual(off_value.shape, on_value.shape)
                    self.assertEqual(off_value.dtype, on_value.dtype)
                    self.assertEqual(off_value.device, on_value.device)
                    self.assertEqual(off_value.requires_grad, on_value.requires_grad)
                    self.assertTrue(torch.equal(off_value, on_value))

    def test_gaussian_model_validates_evidence_then_preserves_active_output(self):
        extend_from_pcd_seq = load_gaussian_model_extend_from_pcd_seq()
        selector_off, _, camera_off, candidates_off = make_active_fixture(603)
        selector_on, _, camera_on, candidates_on = make_active_fixture(603)
        camera_on.device = "cpu"
        harness_off = GaussianModelHookHarness(candidates_off)
        harness_on = GaussianModelHookHarness(candidates_on)
        evidence = make_evidence(
            camera_on,
            count=11,
            mapper_update_id=23,
        )
        extend_from_pcd_seq(
            harness_off,
            camera_off,
            camera_off.uid,
            init=False,
            candidate_selector_v1=selector_off,
            mapper_update_id=23,
        )
        extend_from_pcd_seq(
            harness_on,
            camera_on,
            camera_on.uid,
            init=False,
            candidate_selector_v1=selector_on,
            mapper_update_id=23,
            preinsert_render_evidence=evidence,
        )
        self.assertEqual(harness_off.extend_calls, 1)
        self.assertEqual(harness_on.extend_calls, 1)
        self.assertEqual(harness_off.last_extended_count, ACTIVE_BUDGET)
        self.assertEqual(harness_on.last_extended_count, ACTIVE_BUDGET)

    def test_real_gaussian_model_generation_and_five_extend_inputs_are_equal(self):
        extend_from_pcd_seq = load_gaussian_model_extend_from_pcd_seq()
        selector_off, _, camera_off, candidates_off = make_active_fixture(603)
        selector_on, _, camera_on, candidates_on = make_active_fixture(603)
        camera_on.device = "cpu"

        class RecordingHarness(GaussianModelHookHarness):
            def __init__(self, candidates):
                super().__init__(candidates)
                self.generation_calls = 0
                self.extended = None

            def create_pcd_from_image(self, *args, **kwargs):
                self.generation_calls += 1
                return super().create_pcd_from_image(*args, **kwargs)

            def extend_from_pcd(self, *values):
                self.extended = values[:-1]
                return super().extend_from_pcd(*values)

        harness_off = RecordingHarness(candidates_off)
        harness_on = RecordingHarness(candidates_on)
        evidence = make_evidence(camera_on, count=11, mapper_update_id=23)
        extend_from_pcd_seq(
            harness_off,
            camera_off,
            camera_off.uid,
            init=False,
            candidate_selector_v1=selector_off,
            mapper_update_id=23,
        )
        extend_from_pcd_seq(
            harness_on,
            camera_on,
            camera_on.uid,
            init=False,
            candidate_selector_v1=selector_on,
            mapper_update_id=23,
            preinsert_render_evidence=evidence,
        )
        self.assertEqual(harness_off.generation_calls, 1)
        self.assertEqual(harness_on.generation_calls, 1)
        self.assertEqual(len(harness_off.extended), 5)
        self.assertEqual(len(harness_on.extended), 5)
        for off_value, on_value in zip(harness_off.extended, harness_on.extended):
            self.assertEqual(off_value.shape, on_value.shape)
            self.assertEqual(off_value.dtype, on_value.dtype)
            self.assertEqual(off_value.device, on_value.device)
            self.assertEqual(off_value.requires_grad, on_value.requires_grad)
            self.assertTrue(torch.equal(off_value, on_value))

    def test_stale_evidence_is_rejected_before_candidate_generation(self):
        extend_from_pcd_seq = load_gaussian_model_extend_from_pcd_seq()
        selector, _, camera, candidates = make_active_fixture(603)
        camera.device = "cpu"
        harness = GaussianModelHookHarness(candidates)
        evidence = make_evidence(camera, count=10, mapper_update_id=23)
        with self.assertRaisesRegex(PreinsertRenderEvidenceError, "stale"):
            extend_from_pcd_seq(
                harness,
                camera,
                camera.uid,
                init=False,
                candidate_selector_v1=selector,
                mapper_update_id=23,
                preinsert_render_evidence=evidence,
            )
        self.assertEqual(harness.extend_calls, 0)


if __name__ == "__main__":
    unittest.main()
