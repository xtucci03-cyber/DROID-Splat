"""CPU contract gate for the real GaussianMapper preinsert wiring.

The local CPU environment does not ship the project's optional GUI, Open3D,
or compiled CUDA dependencies.  Import-only placeholders below make the real
``src.gaussian_mapping`` module importable; no Mapper method is extracted,
compiled, copied, or replaced.  Every test invokes the bound production
``GaussianMapper.add_new_gaussians`` method.  Candidate creation and rendering
are controlled boundary spies, while GaussianModel admission and extension
continue through the production methods.
"""

from dataclasses import replace
import importlib
import math
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import torch

from src.candidate_selection.gaussian_candidate_active_topk_v1 import (
    ACTIVE_BUDGET,
    GaussianCandidateActiveTopKV1,
)
from src.candidate_selection.preinsert_render_evidence_v1 import (
    PreinsertRenderEvidenceError,
    PreinsertRenderEvidenceObserverV1,
    PreinsertRenderEvidenceV1,
)
from src.gaussian_candidate_observer import (
    CandidateGenerationMetadata,
    CandidateGenerationResult,
)


MAPPER_UPDATE_ID = 23


def _module(name, **attributes):
    value = ModuleType(name)
    for attribute, item in attributes.items():
        setattr(value, attribute, item)
    return value


def _load_production_runtime():
    """Import the real classes without importing unavailable CPU extensions."""

    termcolor = _module("termcolor", colored=lambda value, *args, **kwargs: value)
    tqdm = _module("tqdm", tqdm=lambda values, *args, **kwargs: values)
    omegaconf = _module("omegaconf", DictConfig=dict)
    plyfile = _module("plyfile", PlyData=object, PlyElement=object)
    simple_knn = _module("simple_knn")
    simple_knn.__path__ = []
    simple_knn_cuda = _module("simple_knn._C", distCUDA2=lambda value: value)
    matplotlib = _module("matplotlib")
    matplotlib.__path__ = []
    matplotlib_pyplot = _module("matplotlib.pyplot")
    renderer = _module(
        "src.gaussian_splatting.gaussian_renderer",
        render=lambda *args, **kwargs: None,
    )
    eval_utils = _module(
        "src.gaussian_splatting.eval_utils",
        EvaluatePacket=object,
    )
    gui_utils = _module(
        "src.gaussian_splatting.gui.gui_utils",
        GaussianPacket=object,
    )
    geom = _module("src.geom", lie_to_matrix=lambda value: value)
    trajectory_filler = _module(
        "src.trajectory_filler",
        PoseTrajectoryFiller=object,
    )
    stubs = {
        "ipdb": _module("ipdb"),
        "termcolor": termcolor,
        "tqdm": tqdm,
        "cv2": _module("cv2"),
        "matplotlib": matplotlib,
        "matplotlib.pyplot": matplotlib_pyplot,
        "omegaconf": omegaconf,
        "open3d": _module("open3d"),
        "lietorch": _module("lietorch"),
        "plyfile": plyfile,
        "simple_knn": simple_knn,
        "simple_knn._C": simple_knn_cuda,
        "src.gaussian_splatting.gaussian_renderer": renderer,
        "src.gaussian_splatting.eval_utils": eval_utils,
        "src.gaussian_splatting.gui.gui_utils": gui_utils,
        "src.geom": geom,
        "src.trajectory_filler": trajectory_filler,
    }
    production_modules = (
        "src.gaussian_mapping",
        "src.gaussian_splatting.scene.gaussian_model",
        "src.gaussian_splatting.camera_utils",
    )
    saved_modules = {
        name: sys.modules.pop(name)
        for name in production_modules
        if name in sys.modules
    }
    imported_modules = {}
    try:
        with mock.patch.dict(sys.modules, stubs, clear=False):
            imported_modules[production_modules[2]] = importlib.import_module(
                production_modules[2]
            )
            imported_modules[production_modules[1]] = importlib.import_module(
                production_modules[1]
            )
            imported_modules[production_modules[0]] = importlib.import_module(
                production_modules[0]
            )
    finally:
        for name in production_modules:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
    camera_module = imported_modules[production_modules[2]]
    model_module = imported_modules[production_modules[1]]
    mapper_module = imported_modules[production_modules[0]]
    return SimpleNamespace(
        mapper_module=mapper_module,
        GaussianMapper=mapper_module.GaussianMapper,
        GaussianModel=model_module.GaussianModel,
        Camera=camera_module.Camera,
    )


def _make_candidates(count):
    index = torch.arange(count, dtype=torch.float32)
    xyz = torch.stack((index, torch.zeros_like(index), torch.ones_like(index)), 1)
    features = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3, 1)
    scales = torch.full((count, 3), -2.0, dtype=torch.float32)
    rotations = torch.zeros(count, 4, dtype=torch.float32)
    if count:
        rotations[:, 0] = 1.0
    opacities = torch.zeros(count, 1, dtype=torch.float32)
    return xyz, features, scales, rotations, opacities


class _ConfidenceSnapshotGetter:
    def __init__(self, confidence):
        self.confidence = confidence
        self.calls = 0

    def __call__(self, camera, require_current=True, *, upsampled=None):
        self.calls += 1
        return SimpleNamespace(
            buffer_index=camera.buffer_index,
            source_frame_id=camera.source_frame_id,
            source_timestamp=camera.source_timestamp,
            confidence_source_frame_id=camera.source_frame_id,
            confidence_version=9,
            confidence_up_version=3,
            is_current=True,
            is_stale=False,
            confidence=self.confidence.clone(),
            shape=tuple(self.confidence.shape),
            dtype=str(self.confidence.dtype),
            device=str(self.confidence.device),
            requires_grad=bool(self.confidence.requires_grad),
        )


class _RejectingAdmission:
    def __init__(self):
        self.calls = 0

    def admit_before_extend(self, **kwargs):
        self.calls += 1
        raise AssertionError("M01 must not run after Active selection.")


class GaussianMapperPreinsertWiringCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = _load_production_runtime()
        if (
            cls.runtime.GaussianMapper.add_new_gaussians.__module__
            != cls.runtime.mapper_module.__name__
        ):
            raise AssertionError("GaussianMapper method did not come from production module.")

    def _camera(self, *, uid=7, width=8):
        height = 1
        depth = torch.ones(height, width, dtype=torch.float32)
        return self.runtime.Camera(
            uid=uid,
            color=torch.zeros(3, height, width, dtype=torch.float32),
            depth_est=depth,
            depth_gt=depth.clone(),
            pose_w2c=torch.eye(4, dtype=torch.float32),
            projection_matrix=torch.eye(4, dtype=torch.float32),
            intrinsics=(1.0, 1.0, 0.0, 0.0),
            fov=(math.pi / 2.0, math.pi / 2.0),
            img_size=(height, width),
            device="cpu",
            buffer_index=uid + 100,
            source_frame_id=uid + 1000,
            source_timestamp=float(uid) + 0.25,
        )

    @staticmethod
    def _training_args():
        return SimpleNamespace(
            percent_dense=0.01,
            position_lr_init=1.0e-4,
            position_lr_final=1.0e-6,
            position_lr_delay_mult=1.0,
            position_lr_max_steps=10,
            feature_lr=1.0e-3,
            opacity_lr=1.0e-2,
            scaling_lr=1.0e-3,
            rotation_lr=1.0e-3,
        )

    def _model(self, candidates_by_uid, *, initial_count=3, audit=None):
        audit = audit if audit is not None else SimpleNamespace()
        audit.events = getattr(audit, "events", [])
        audit.generation_calls = getattr(audit, "generation_calls", 0)
        audit.extend_calls = getattr(audit, "extend_calls", 0)
        audit.evidence = getattr(audit, "evidence", [])
        audit.extended = getattr(audit, "extended", [])
        production_model = self.runtime.GaussianModel

        class ControlledGaussianModel(production_model):
            def extend_from_pcd_seq(inner_self, *args, **kwargs):
                evidence = kwargs.get("preinsert_render_evidence")
                audit.evidence.append(
                    None
                    if evidence is None
                    else {
                        "available": evidence.available,
                        "reason": evidence.unavailable_reason,
                        "gaussian_count": evidence.gaussian_count_before_render,
                        "mapper_update_id": evidence.mapper_update_id,
                        "camera_id": evidence.source_camera_id,
                    }
                )
                return super().extend_from_pcd_seq(*args, **kwargs)

            def create_pcd_from_image(
                inner_self,
                camera,
                *args,
                collect_candidate_metadata=False,
                **kwargs,
            ):
                audit.generation_calls += 1
                audit.events.append(f"generate:{camera.uid}")
                candidates = candidates_by_uid[camera.uid]
                if collect_candidate_metadata:
                    count = int(candidates[0].shape[0])
                    return CandidateGenerationResult(
                        candidates=candidates,
                        metadata=CandidateGenerationMetadata(
                            depth_source="estimated_clean_depth",
                            depth_pixel_count=int(camera.depth.numel()),
                            valid_depth_count=int(camera.depth.numel()),
                            pre_downsample_point_count=count,
                            post_downsample_point_count=count,
                        ),
                    )
                return candidates

            def extend_from_pcd(inner_self, *values):
                audit.extend_calls += 1
                kf_id = values[-1]
                audit.events.append(f"extend:{kf_id}")
                audit.extended.append(tuple(value.detach().clone() for value in values[:-1]))
                return super().extend_from_pcd(*values)

        model = ControlledGaussianModel(sh_degree=0, config=None, device="cpu")
        old_xyz = torch.zeros(initial_count, 3, dtype=torch.float32)
        if initial_count:
            old_xyz[:, 2] = 2.0
        model._xyz = torch.nn.Parameter(old_xyz)
        model._features_dc = torch.nn.Parameter(
            torch.zeros(initial_count, 1, 3, dtype=torch.float32)
        )
        model._features_rest = torch.nn.Parameter(
            torch.empty(initial_count, 0, 3, dtype=torch.float32)
        )
        model._scaling = torch.nn.Parameter(
            torch.full((initial_count, 3), -2.0, dtype=torch.float32)
        )
        rotation = torch.zeros(initial_count, 4, dtype=torch.float32)
        if initial_count:
            rotation[:, 0] = 1.0
        model._rotation = torch.nn.Parameter(rotation)
        model._opacity = torch.nn.Parameter(
            torch.zeros(initial_count, 1, dtype=torch.float32)
        )
        model.unique_kfIDs = torch.zeros(initial_count, dtype=torch.int32)
        model.n_obs = torch.zeros(initial_count, dtype=torch.int32)
        model.n_optimized = torch.zeros(initial_count, dtype=torch.int32)
        model.init_lr(1.0)
        model.training_setup(self._training_args())
        return model, audit

    def _mapper(self, model, *, observer=None, selector=None, initialized=True):
        mapper = self.runtime.GaussianMapper.__new__(self.runtime.GaussianMapper)
        mapper.gaussians = model
        mapper.preinsert_render_evidence_v1 = observer
        mapper.candidate_observer = None
        mapper.candidate_selector = None
        mapper.candidate_selector_v1 = selector
        mapper.pipeline_params = SimpleNamespace(
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )
        mapper.background = torch.zeros(3, dtype=torch.float32)
        mapper.count = MAPPER_UPDATE_ID
        mapper.initialized = initialized
        mapper.info = lambda *args, **kwargs: None
        self.assertIs(
            mapper.add_new_gaussians.__func__,
            self.runtime.GaussianMapper.add_new_gaussians,
        )
        return mapper

    @staticmethod
    def _renderer(audit):
        def render(camera, gaussians, pipeline, background, **kwargs):
            audit.events.append(f"render:{camera.uid}")
            height, width = camera.image_height, camera.image_width
            return {
                "render": torch.full((3, height, width), 0.25),
                "depth": torch.full((1, height, width), 1.5),
                "opacity": torch.full((1, height, width), 0.5),
            }

        return render

    def _active_selector(self, count, audit):
        confidence = torch.arange(count, dtype=torch.float32).reshape(1, count)
        getter = _ConfidenceSnapshotGetter(confidence)

        class RecordingActiveSelector(GaussianCandidateActiveTopKV1):
            def select_before_extend(inner_self, **kwargs):
                result = super().select_before_extend(**kwargs)
                audit.selected_indices = (
                    None
                    if result.selected_indices is None
                    else result.selected_indices.detach().clone()
                )
                return result

        selector = RecordingActiveSelector(
            confidence_snapshot_getter=getter,
            logging_enabled=False,
            device="cpu",
            active_budget=ACTIVE_BUDGET,
        )
        return selector, getter

    def test_off_mode_calls_bound_mapper_without_preinsert_render(self):
        camera = self._camera()
        model, audit = self._model({camera.uid: _make_candidates(7)})
        mapper = self._mapper(model, observer=None)
        with mock.patch.object(
            self.runtime.mapper_module,
            "render",
            side_effect=AssertionError("off mode must not render"),
        ) as renderer:
            result = mapper.add_new_gaussians([camera])
        self.assertIs(result, camera)
        self.assertEqual(renderer.call_count, 0)
        self.assertEqual(audit.generation_calls, 1)
        self.assertEqual(audit.extend_calls, 1)

    def test_init_bypass_never_renders_and_still_extends_once(self):
        camera = self._camera()
        observer = PreinsertRenderEvidenceObserverV1(device="cpu")
        model, audit = self._model({camera.uid: _make_candidates(7)})
        mapper = self._mapper(model, observer=observer, initialized=False)
        with mock.patch.object(observer, "capture", wraps=observer.capture) as capture, mock.patch.object(
            self.runtime.mapper_module,
            "render",
            side_effect=AssertionError("init bypass must not render"),
        ) as renderer:
            mapper.add_new_gaussians([camera])
        self.assertEqual(capture.call_count, 0)
        self.assertEqual(renderer.call_count, 0)
        self.assertEqual(audit.extend_calls, 1)
        self.assertEqual(audit.evidence[0]["reason"], "init_bypass")
        self.assertFalse(audit.evidence[0]["available"])

    def test_empty_old_map_produces_unavailable_evidence_without_render(self):
        camera = self._camera()
        observer = PreinsertRenderEvidenceObserverV1(device="cpu")
        model, audit = self._model(
            {camera.uid: _make_candidates(7)}, initial_count=0
        )
        mapper = self._mapper(model, observer=observer)
        with mock.patch.object(observer, "capture", wraps=observer.capture) as capture, mock.patch.object(
            self.runtime.mapper_module,
            "render",
            side_effect=AssertionError("empty old map must not render"),
        ) as renderer:
            mapper.add_new_gaussians([camera])
        self.assertEqual(capture.call_count, 1)
        self.assertEqual(renderer.call_count, 0)
        self.assertEqual(audit.evidence[0]["reason"], "empty_old_map")
        self.assertEqual(audit.evidence[0]["gaussian_count"], 0)
        self.assertEqual(len(model), 7)

    def test_noninit_capture_is_once_before_generation_and_real_extend(self):
        camera = self._camera()
        observer = PreinsertRenderEvidenceObserverV1(device="cpu")
        model, audit = self._model({camera.uid: _make_candidates(7)}, initial_count=3)
        mapper = self._mapper(model, observer=observer)
        with mock.patch.object(observer, "capture", wraps=observer.capture) as capture, mock.patch.object(
            self.runtime.mapper_module,
            "render",
            side_effect=self._renderer(audit),
        ) as renderer:
            mapper.add_new_gaussians([camera])
        self.assertEqual(capture.call_count, 1)
        self.assertEqual(renderer.call_count, 1)
        self.assertEqual(audit.events, ["render:7", "generate:7", "extend:7"])
        self.assertEqual(audit.extend_calls, 1)
        self.assertEqual(audit.evidence[0]["gaussian_count"], 3)
        self.assertEqual(len(model), 10)

    def test_two_cameras_use_strict_per_event_render_extend_order(self):
        cameras = [self._camera(uid=7), self._camera(uid=8)]
        candidates = {camera.uid: _make_candidates(7) for camera in cameras}
        observer = PreinsertRenderEvidenceObserverV1(device="cpu")
        model, audit = self._model(candidates, initial_count=3)
        mapper = self._mapper(model, observer=observer)
        with mock.patch.object(observer, "capture", wraps=observer.capture) as capture, mock.patch.object(
            self.runtime.mapper_module,
            "render",
            side_effect=self._renderer(audit),
        ) as renderer:
            mapper.add_new_gaussians(cameras)
        self.assertEqual(capture.call_count, 2)
        self.assertEqual(renderer.call_count, 2)
        self.assertEqual(
            audit.events,
            [
                "render:7",
                "generate:7",
                "extend:7",
                "render:8",
                "generate:8",
                "extend:8",
            ],
        )
        self.assertEqual(
            [item["gaussian_count"] for item in audit.evidence], [3, 10]
        )

    def _run_active(self, *, evidence_enabled):
        count = 603
        camera = self._camera(width=count)
        candidates = _make_candidates(count)
        model, audit = self._model({camera.uid: candidates}, initial_count=3)
        selector, getter = self._active_selector(count, audit)
        observer = (
            PreinsertRenderEvidenceObserverV1(device="cpu")
            if evidence_enabled
            else None
        )
        mapper = self._mapper(model, observer=observer, selector=selector)
        renderer = self._renderer(audit)
        with mock.patch.object(
            self.runtime.mapper_module,
            "render",
            side_effect=renderer,
        ) as render_spy:
            mapper.add_new_gaussians([camera])
        return SimpleNamespace(
            candidates=candidates,
            model=model,
            audit=audit,
            getter=getter,
            render_calls=render_spy.call_count,
        )

    def test_evidence_on_off_preserves_active_selection_and_five_tensors(self):
        disabled = self._run_active(evidence_enabled=False)
        enabled = self._run_active(evidence_enabled=True)
        self.assertEqual(disabled.render_calls, 0)
        self.assertEqual(enabled.render_calls, 1)
        self.assertEqual(disabled.getter.calls, 1)
        self.assertEqual(enabled.getter.calls, 1)
        self.assertTrue(
            torch.equal(
                disabled.audit.selected_indices,
                enabled.audit.selected_indices,
            )
        )
        self.assertEqual(int(enabled.audit.selected_indices.shape[0]), 600)
        for left, right in zip(disabled.audit.extended[0], enabled.audit.extended[0]):
            torch.testing.assert_close(left, right, atol=0.0, rtol=0.0)
        for original, frozen in zip(disabled.candidates, _make_candidates(603)):
            torch.testing.assert_close(original, frozen, atol=0.0, rtol=0.0)
        self.assertEqual(len(disabled.model), len(enabled.model))
        self.assertEqual(len(enabled.model), 603)

    def test_stale_identity_update_and_count_fail_before_candidate_generation(self):
        defects = ("camera_identity", "mapper_update_id", "gaussian_count")
        for defect in defects:
            with self.subTest(defect=defect):
                camera = self._camera()
                model, audit = self._model(
                    {camera.uid: _make_candidates(7)}, initial_count=3
                )

                class DefectiveObserver(PreinsertRenderEvidenceObserverV1):
                    def capture(inner_self, **kwargs):
                        evidence = super().capture(**kwargs)
                        if defect == "camera_identity":
                            camera.source_frame_id += 1
                            return evidence
                        if defect == "mapper_update_id":
                            return replace(
                                evidence,
                                mapper_update_id=evidence.mapper_update_id + 1,
                            )
                        return replace(
                            evidence,
                            gaussian_count_before_render=(
                                evidence.gaussian_count_before_render + 1
                            ),
                        )

                mapper = self._mapper(
                    model,
                    observer=DefectiveObserver(device="cpu"),
                )
                with mock.patch.object(
                    self.runtime.mapper_module,
                    "render",
                    side_effect=self._renderer(audit),
                ):
                    with self.assertRaises(PreinsertRenderEvidenceError):
                        mapper.add_new_gaussians([camera])
                self.assertEqual(audit.generation_calls, 0)
                self.assertEqual(audit.extend_calls, 0)

    def test_call_does_not_add_persistent_evidence_attributes(self):
        camera = self._camera()
        observer = PreinsertRenderEvidenceObserverV1(device="cpu")
        model, audit = self._model({camera.uid: _make_candidates(7)}, initial_count=3)
        mapper = self._mapper(model, observer=observer)
        mapper_keys = frozenset(vars(mapper))
        camera_keys = frozenset(vars(camera))
        model_keys = frozenset(vars(model))
        observer_keys = frozenset(vars(observer))
        render_tensors = []

        def renderer(*args, **kwargs):
            package = self._renderer(audit)(*args, **kwargs)
            render_tensors.extend(package.values())
            return package

        with mock.patch.object(
            self.runtime.mapper_module,
            "render",
            side_effect=renderer,
        ):
            mapper.add_new_gaussians([camera])
        self.assertEqual(frozenset(vars(mapper)), mapper_keys)
        self.assertEqual(frozenset(vars(camera)), camera_keys)
        self.assertEqual(frozenset(vars(model)), model_keys)
        self.assertEqual(frozenset(vars(observer)), observer_keys)
        evidence_ptrs = {tensor.data_ptr() for tensor in render_tensors}
        for owner in (mapper, camera, model, observer):
            self.assertFalse(
                any(
                    isinstance(value, PreinsertRenderEvidenceV1)
                    for value in vars(owner).values()
                )
            )
            self.assertFalse(
                any(
                    isinstance(value, torch.Tensor)
                    and value.data_ptr() in evidence_ptrs
                    for value in vars(owner).values()
                )
            )

    def test_active_and_m01_remain_fail_closed_and_never_double_select(self):
        count = 603
        camera = self._camera(width=count)
        model, audit = self._model(
            {camera.uid: _make_candidates(count)}, initial_count=3
        )
        admission = _RejectingAdmission()
        model.resource_admission = admission
        selector, getter = self._active_selector(count, audit)
        mapper = self._mapper(model, selector=selector)
        with self.assertRaisesRegex(RuntimeError, "possible second selection"):
            mapper.add_new_gaussians([camera])
        self.assertEqual(selector.active_budget, 600)
        self.assertEqual(getter.calls, 1)
        self.assertEqual(int(audit.selected_indices.shape[0]), 600)
        self.assertEqual(admission.calls, 0)
        self.assertEqual(audit.extend_calls, 0)


if __name__ == "__main__":
    unittest.main()
