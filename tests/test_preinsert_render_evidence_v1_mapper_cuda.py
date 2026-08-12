"""No-dataset CUDA gate for the production GaussianMapper preinsert path.

The CUDA-only project imports are delayed until ``setUpClass`` so CPU-only
discovery remains possible.  A valid server run must report two tests and zero
skips.  Renderer spies in this file always forward to the production renderer;
no CUDA output is mocked.
"""

from types import MethodType, SimpleNamespace
import unittest
from unittest import mock

import torch

from src.candidate_selection.preinsert_render_evidence_v1 import (
    PreinsertRenderEvidenceObserverV1,
    PreinsertRenderEvidenceV1,
)
from src.gaussian_candidate_observer import (
    CandidateGenerationMetadata,
    CandidateGenerationResult,
)


CUDA_DEVICE = torch.device("cuda:0")
HEIGHT = 32
WIDTH = 40
FX = 50.0
FY = 50.0
CX = WIDTH / 2.0
CY = HEIGHT / 2.0
OLD_GAUSSIAN_Z = 2.0
ATOL = 1.0e-5


@unittest.skipUnless(
    torch.cuda.is_available(),
    "Real GaussianMapper CUDA gate not run: CUDA is unavailable.",
)
class GaussianMapperPreinsertRealCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.device_count() < 1:
            raise RuntimeError("Real GaussianMapper gate requires cuda:0.")
        torch.cuda.set_device(CUDA_DEVICE)

        # These imports must fail the server gate if any production CUDA
        # dependency or extension is unavailable.
        import src.gaussian_mapping as mapper_module
        from src.gaussian_mapping import GaussianMapper
        from src.gaussian_splatting.camera_utils import Camera
        from src.gaussian_splatting.gaussian_renderer import render
        from src.gaussian_splatting.scene.gaussian_model import GaussianModel
        from src.gaussian_splatting.utils.general_utils import inverse_sigmoid
        from src.gaussian_splatting.utils.graphics_utils import (
            focal2fov,
            getProjectionMatrix2,
        )
        from src.gaussian_splatting.utils.sh_utils import RGB2SH

        cls.mapper_module = mapper_module
        cls.GaussianMapper = GaussianMapper
        cls.Camera = Camera
        cls.GaussianModel = GaussianModel
        cls.production_render = staticmethod(render)
        cls.inverse_sigmoid = staticmethod(inverse_sigmoid)
        cls.focal2fov = staticmethod(focal2fov)
        cls.getProjectionMatrix2 = staticmethod(getProjectionMatrix2)
        cls.RGB2SH = staticmethod(RGB2SH)
        torch.cuda.synchronize(CUDA_DEVICE)

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

    def _scene(self):
        depth = torch.full(
            (HEIGHT, WIDTH),
            OLD_GAUSSIAN_Z,
            dtype=torch.float32,
            device=CUDA_DEVICE,
        )
        projection = self.getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            cx=CX,
            cy=CY,
            fx=FX,
            fy=FY,
            W=WIDTH,
            H=HEIGHT,
        ).transpose(0, 1)
        camera = self.Camera(
            uid=7,
            color=torch.zeros(3, HEIGHT, WIDTH, device=CUDA_DEVICE),
            depth_est=depth,
            depth_gt=depth.clone(),
            pose_w2c=torch.eye(4, device=CUDA_DEVICE),
            projection_matrix=projection,
            intrinsics=(FX, FY, CX, CY),
            fov=(
                self.focal2fov(FX, WIDTH),
                self.focal2fov(FY, HEIGHT),
            ),
            img_size=(HEIGHT, WIDTH),
            device=str(CUDA_DEVICE),
            buffer_index=3,
            source_frame_id=101,
            source_timestamp=1.25,
        )

        model = self.GaussianModel(
            sh_degree=0,
            config=None,
            device=str(CUDA_DEVICE),
        )
        xyz = torch.tensor(
            [[0.0, 0.0, OLD_GAUSSIAN_Z]],
            dtype=torch.float32,
            device=CUDA_DEVICE,
        )
        color = torch.tensor(
            [[0.9, 0.2, 0.1]], dtype=torch.float32, device=CUDA_DEVICE
        )
        model._xyz = torch.nn.Parameter(xyz)
        model._features_dc = torch.nn.Parameter(self.RGB2SH(color).unsqueeze(1))
        model._features_rest = torch.nn.Parameter(
            torch.empty(1, 0, 3, dtype=torch.float32, device=CUDA_DEVICE)
        )
        model._scaling = torch.nn.Parameter(
            torch.log(
                torch.tensor(
                    [[0.08, 0.08, 0.08]],
                    dtype=torch.float32,
                    device=CUDA_DEVICE,
                )
            )
        )
        model._rotation = torch.nn.Parameter(
            torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
                device=CUDA_DEVICE,
            )
        )
        model._opacity = torch.nn.Parameter(
            self.inverse_sigmoid(
                torch.tensor([[0.9]], dtype=torch.float32, device=CUDA_DEVICE)
            )
        )
        model.unique_kfIDs = torch.zeros(1, dtype=torch.int32, device=CUDA_DEVICE)
        model.n_obs = torch.zeros(1, dtype=torch.int32, device=CUDA_DEVICE)
        model.n_optimized = torch.zeros(1, dtype=torch.int32, device=CUDA_DEVICE)
        model.init_lr(1.0)
        model.training_setup(self._training_args())

        candidates = (
            torch.tensor(
                [
                    [-0.04, 0.0, 2.0],
                    [-0.02, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                    [0.02, 0.0, 2.0],
                    [0.04, 0.0, 2.0],
                    [0.06, 0.0, 2.0],
                ],
                dtype=torch.float32,
                device=CUDA_DEVICE,
            ),
            torch.zeros(6, 3, 1, dtype=torch.float32, device=CUDA_DEVICE),
            torch.full((6, 3), -2.5, dtype=torch.float32, device=CUDA_DEVICE),
            torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]] * 6,
                dtype=torch.float32,
                device=CUDA_DEVICE,
            ),
            torch.zeros(6, 1, dtype=torch.float32, device=CUDA_DEVICE),
        )
        events = []

        def controlled_generation(
            model_self,
            event_camera,
            *args,
            collect_candidate_metadata=False,
            **kwargs,
        ):
            events.append("candidate_generation")
            if collect_candidate_metadata:
                return CandidateGenerationResult(
                    candidates=candidates,
                    metadata=CandidateGenerationMetadata(
                        depth_source="estimated_clean_depth",
                        depth_pixel_count=HEIGHT * WIDTH,
                        valid_depth_count=HEIGHT * WIDTH,
                        pre_downsample_point_count=6,
                        post_downsample_point_count=6,
                    ),
                )
            return candidates

        production_extend = model.extend_from_pcd

        def forwarding_extend(model_self, *values):
            events.append("extend")
            return production_extend(*values)

        model.create_pcd_from_image = MethodType(controlled_generation, model)
        model.extend_from_pcd = MethodType(forwarding_extend, model)

        observer = PreinsertRenderEvidenceObserverV1(device=CUDA_DEVICE)
        mapper = self.GaussianMapper.__new__(self.GaussianMapper)
        mapper.gaussians = model
        mapper.preinsert_render_evidence_v1 = observer
        mapper.candidate_observer = None
        mapper.candidate_selector = None
        mapper.candidate_selector_v1 = None
        mapper.pipeline_params = SimpleNamespace(
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )
        mapper.background = torch.zeros(3, device=CUDA_DEVICE)
        mapper.count = 23
        mapper.initialized = True
        mapper.info = lambda *args, **kwargs: None
        self.assertIs(
            mapper.add_new_gaussians.__func__,
            self.GaussianMapper.add_new_gaussians,
        )
        return camera, model, observer, mapper, events

    @staticmethod
    def _contains_evidence(owner):
        return any(
            isinstance(value, PreinsertRenderEvidenceV1)
            for value in vars(owner).values()
        )

    def _run_real_mapper_event(self):
        camera, model, observer, mapper, events = self._scene()
        evidence_values = []
        render_packages = []
        render_calls = 0
        original_capture = observer.capture

        def capture_spy(**kwargs):
            evidence = original_capture(**kwargs)
            evidence_values.append(evidence)
            return evidence

        def renderer_spy(*args, **kwargs):
            nonlocal render_calls
            render_calls += 1
            events.append("render")
            package = self.production_render(*args, **kwargs)
            render_packages.append(package)
            return package

        keys_before = {
            "mapper": frozenset(vars(mapper)),
            "camera": frozenset(vars(camera)),
            "model": frozenset(vars(model)),
            "observer": frozenset(vars(observer)),
        }
        old_count = len(model)
        with mock.patch.object(observer, "capture", side_effect=capture_spy), mock.patch.object(
            self.mapper_module,
            "render",
            side_effect=renderer_spy,
        ):
            returned_camera = mapper.add_new_gaussians([camera])
        torch.cuda.synchronize(CUDA_DEVICE)
        return SimpleNamespace(
            camera=camera,
            model=model,
            observer=observer,
            mapper=mapper,
            events=events,
            evidence=evidence_values,
            render_packages=render_packages,
            render_calls=render_calls,
            old_count=old_count,
            returned_camera=returned_camera,
            keys_before=keys_before,
        )

    def test_bound_mapper_real_render_precedes_real_gaussian_extend(self):
        result = self._run_real_mapper_event()
        self.assertIs(result.returned_camera, result.camera)
        self.assertEqual(result.render_calls, 1)
        self.assertEqual(result.events, ["render", "candidate_generation", "extend"])
        self.assertEqual(len(result.evidence), 1)
        evidence = result.evidence[0]
        package = result.render_packages[0]
        self.assertTrue(evidence.available)
        self.assertEqual(evidence.gaussian_count_before_render, result.old_count)
        self.assertEqual(result.old_count, 1)
        self.assertEqual(len(result.model), 7)
        for evidence_tensor, package_tensor, shape in (
            (evidence.render_rgb, package["render"], (3, HEIGHT, WIDTH)),
            (evidence.depth_accum, package["depth"], (1, HEIGHT, WIDTH)),
            (evidence.alpha_accum, package["opacity"], (1, HEIGHT, WIDTH)),
        ):
            self.assertEqual(evidence_tensor.data_ptr(), package_tensor.data_ptr())
            self.assertEqual(tuple(evidence_tensor.shape), shape)
            self.assertEqual(evidence_tensor.device, CUDA_DEVICE)
            self.assertFalse(evidence_tensor.requires_grad)
            self.assertIsNone(evidence_tensor.grad_fn)
            self.assertTrue(bool(torch.isfinite(evidence_tensor).all()))
        alpha = evidence.alpha_accum
        self.assertTrue(bool((alpha >= -ATOL).all()))
        self.assertTrue(bool((alpha <= 1.0 + ATOL).all()))
        covered = alpha > ATOL
        self.assertTrue(bool(covered.any()))
        normalized_depth = evidence.depth_accum[covered] / alpha[covered]
        torch.testing.assert_close(
            normalized_depth,
            torch.full_like(normalized_depth, OLD_GAUSSIAN_Z),
            atol=ATOL,
            rtol=ATOL,
        )
        background = result.mapper.background[:, None, None].expand_as(
            evidence.render_rgb
        )
        self.assertFalse(
            torch.allclose(evidence.render_rgb, background, atol=ATOL, rtol=ATOL)
        )

    def test_bound_mapper_leaves_no_persistent_evidence(self):
        result = self._run_real_mapper_event()
        owners = {
            "mapper": result.mapper,
            "camera": result.camera,
            "model": result.model,
            "observer": result.observer,
        }
        evidence_ptrs = {
            tensor.data_ptr()
            for evidence in result.evidence
            for tensor in (
                evidence.render_rgb,
                evidence.depth_accum,
                evidence.alpha_accum,
            )
        }
        for name, owner in owners.items():
            self.assertEqual(frozenset(vars(owner)), result.keys_before[name])
            self.assertFalse(self._contains_evidence(owner))
            self.assertFalse(
                any(
                    isinstance(value, torch.Tensor)
                    and value.data_ptr() in evidence_ptrs
                    for value in vars(owner).values()
                )
            )


if __name__ == "__main__":
    unittest.main()
