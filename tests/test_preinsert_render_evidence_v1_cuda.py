"""CUDA numerical gate for pre-insertion render evidence.

The gate uses a deterministic synthetic scene and the production Camera,
GaussianModel, renderer, and evidence observer.  CUDA-only project imports are
deliberately delayed until ``setUpClass`` so CPU-only unittest discovery can
import this module.  A server run is valid only when unittest reports zero
skips.
"""

from types import SimpleNamespace
import unittest

import torch

from src.candidate_selection.preinsert_render_evidence_v1 import (
    PreinsertRenderEvidenceObserverV1,
    PreinsertRenderEvidenceV1,
)


CUDA_DEVICE = torch.device("cuda:0")
HEIGHT = 32
WIDTH = 40
FX = 50.0
FY = 50.0
CX = WIDTH / 2.0
CY = HEIGHT / 2.0
GAUSSIAN_CAMERA_Z = 2.0
ATOL = 1.0e-5
RTOL = 1.0e-5
ALPHA_COVERAGE_EPS = 1.0e-6


@unittest.skipUnless(
    torch.cuda.is_available(),
    "CUDA numerical gate not run: torch.cuda.is_available() is false.",
)
class PreinsertRenderEvidenceRealRasterizerCudaTests(unittest.TestCase):
    """Connect the evidence interface to the actual CUDA rasterizer kernel."""

    @classmethod
    def setUpClass(cls) -> None:
        if torch.cuda.device_count() < 1:
            raise RuntimeError("CUDA numerical gate requires cuda:0.")
        torch.cuda.set_device(CUDA_DEVICE)

        # These imports transitively load the compiled CUDA extensions.  They
        # must fail the server gate, rather than being replaced or skipped, if
        # the production rasterizer is unavailable.
        from src.gaussian_splatting.camera_utils import Camera
        from src.gaussian_splatting.gaussian_renderer import render
        from src.gaussian_splatting.scene.gaussian_model import GaussianModel
        from src.gaussian_splatting.utils.general_utils import inverse_sigmoid
        from src.gaussian_splatting.utils.graphics_utils import (
            focal2fov,
            getProjectionMatrix2,
        )
        from src.gaussian_splatting.utils.sh_utils import RGB2SH

        cls.Camera = Camera
        cls.GaussianModel = GaussianModel
        cls.production_render = staticmethod(render)
        cls.inverse_sigmoid = staticmethod(inverse_sigmoid)
        cls.focal2fov = staticmethod(focal2fov)
        cls.getProjectionMatrix2 = staticmethod(getProjectionMatrix2)
        cls.RGB2SH = staticmethod(RGB2SH)

        probe = torch.ones(1, dtype=torch.float32, device=CUDA_DEVICE)
        if probe.device != CUDA_DEVICE:
            raise RuntimeError("CUDA probe tensor was not created on cuda:0.")
        torch.cuda.synchronize(CUDA_DEVICE)

    def _make_scene(self):
        color = torch.zeros(
            (3, HEIGHT, WIDTH), dtype=torch.float32, device=CUDA_DEVICE
        )
        depth = torch.full(
            (1, HEIGHT, WIDTH),
            GAUSSIAN_CAMERA_Z,
            dtype=torch.float32,
            device=CUDA_DEVICE,
        )
        pose_w2c = torch.eye(4, dtype=torch.float32, device=CUDA_DEVICE)
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
            color=color,
            depth_est=depth,
            depth_gt=depth.clone(),
            pose_w2c=pose_w2c,
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
            [[0.0, 0.0, GAUSSIAN_CAMERA_Z]],
            dtype=torch.float32,
            device=CUDA_DEVICE,
        )
        rgb = torch.tensor(
            [[0.9, 0.2, 0.1]], dtype=torch.float32, device=CUDA_DEVICE
        )
        features_dc = self.RGB2SH(rgb).unsqueeze(1)
        features_rest = torch.empty(
            (1, 0, 3), dtype=torch.float32, device=CUDA_DEVICE
        )
        scaling = torch.log(
            torch.tensor(
                [[0.08, 0.08, 0.08]],
                dtype=torch.float32,
                device=CUDA_DEVICE,
            )
        )
        rotation = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=CUDA_DEVICE,
        )
        opacity = self.inverse_sigmoid(
            torch.tensor([[0.9]], dtype=torch.float32, device=CUDA_DEVICE)
        )

        model._xyz = torch.nn.Parameter(xyz)
        model._features_dc = torch.nn.Parameter(features_dc)
        model._features_rest = torch.nn.Parameter(features_rest)
        model._scaling = torch.nn.Parameter(scaling)
        model._rotation = torch.nn.Parameter(rotation)
        model._opacity = torch.nn.Parameter(opacity)

        pipeline = SimpleNamespace(
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )
        background = torch.zeros(3, dtype=torch.float32, device=CUDA_DEVICE)
        return camera, model, pipeline, background

    @staticmethod
    def _tensor_state(tensor):
        return {
            "data_ptr": tensor.data_ptr(),
            "version": tensor._version,
            "shape": tuple(tensor.shape),
            "dtype": tensor.dtype,
            "device": tensor.device,
            "requires_grad": tensor.requires_grad,
            "values": tensor.detach().clone(),
        }

    @staticmethod
    def _semantic_tensor_state(tensor):
        # Renderer-facing properties such as full_proj_transform are computed
        # tensors, so each property access legitimately has a new data_ptr.
        return {
            "shape": tuple(tensor.shape),
            "dtype": tensor.dtype,
            "device": tensor.device,
            "requires_grad": tensor.requires_grad,
            "values": tensor.detach().clone(),
        }

    @classmethod
    def _direct_tensor_states(cls, value):
        tensors = {
            name: item
            for name, item in vars(value).items()
            if isinstance(item, torch.Tensor)
        }
        if isinstance(value, torch.nn.Module):
            tensors.update(value.named_parameters(recurse=False))
            tensors.update(value.named_buffers(recurse=False))
        return {
            name: cls._tensor_state(tensor)
            for name, tensor in sorted(tensors.items())
        }

    @classmethod
    def _scene_state(cls, camera, model):
        camera_renderer_tensors = {
            "world_view_transform": camera.world_view_transform,
            "projection_matrix": camera.projection_matrix,
            "full_proj_transform": camera.full_proj_transform,
            "camera_center": camera.camera_center,
            "cam_rot_delta": camera.cam_rot_delta,
            "cam_trans_delta": camera.cam_trans_delta,
        }
        model_renderer_tensors = {
            "means3D": model.get_xyz,
            "features": model.get_features,
            "opacity": model.get_opacity,
            "scaling": model.get_scaling,
            "rotation": model.get_rotation,
        }
        return {
            "camera_attribute_keys": frozenset(vars(camera)),
            "model_attribute_keys": frozenset(vars(model)),
            "camera_direct_tensors": cls._direct_tensor_states(camera),
            "model_direct_tensors": cls._direct_tensor_states(model),
            "camera_renderer_tensors": {
                name: cls._semantic_tensor_state(tensor)
                for name, tensor in camera_renderer_tensors.items()
            },
            "model_renderer_tensors": {
                name: cls._semantic_tensor_state(tensor)
                for name, tensor in model_renderer_tensors.items()
            },
            "camera_identity": {
                name: getattr(camera, name)
                for name in (
                    "uid",
                    "device",
                    "buffer_index",
                    "source_frame_id",
                    "source_timestamp",
                    "fx",
                    "fy",
                    "cx",
                    "cy",
                    "FoVx",
                    "FoVy",
                    "image_height",
                    "image_width",
                )
            },
            "model_identity": {
                name: getattr(model, name)
                for name in (
                    "active_sh_degree",
                    "max_sh_degree",
                    "device",
                    "max_scale",
                    "isotropic",
                )
            },
        }

    def _assert_tensor_states_equal(self, before, after, label):
        self.assertEqual(set(after), set(before), msg=f"{label}: tensor keys")
        for name in sorted(before):
            before_tensor = before[name]
            after_tensor = after[name]
            for field in before_tensor.keys() - {"values"}:
                self.assertEqual(
                    after_tensor[field],
                    before_tensor[field],
                    msg=f"{label}: {name}.{field}",
                )
            torch.testing.assert_close(
                after_tensor["values"],
                before_tensor["values"],
                atol=0.0,
                rtol=0.0,
                equal_nan=True,
                msg=f"{label}: {name}.values",
            )

    def _assert_scene_state_equal(self, before, after, label):
        for field in (
            "camera_attribute_keys",
            "model_attribute_keys",
            "camera_identity",
            "model_identity",
        ):
            self.assertEqual(after[field], before[field], msg=f"{label}: {field}")
        for field in (
            "camera_direct_tensors",
            "model_direct_tensors",
            "camera_renderer_tensors",
            "model_renderer_tensors",
        ):
            self._assert_tensor_states_equal(
                before[field], after[field], f"{label}: {field}"
            )

    def test_real_capture_matches_direct_production_render_numerically(self):
        camera, model, pipeline, background = self._make_scene()
        observer = PreinsertRenderEvidenceObserverV1(device=CUDA_DEVICE)
        render_calls = []

        def forwarding_spy(*args, **kwargs):
            package = self.production_render(*args, **kwargs)
            render_calls.append(package)
            return package

        scene_state_before = self._scene_state(camera, model)
        evidence = observer.capture(
            camera=camera,
            gaussians=model,
            renderer=forwarding_spy,
            pipeline_params=pipeline,
            background=background,
            mapper_update_id=23,
        )
        torch.cuda.synchronize(CUDA_DEVICE)
        scene_state_after_capture = self._scene_state(camera, model)
        self._assert_scene_state_equal(
            scene_state_before,
            scene_state_after_capture,
            "capture mutation",
        )

        self.assertEqual(len(render_calls), 1)
        captured_package = render_calls[0]
        self.assertIsInstance(evidence, PreinsertRenderEvidenceV1)
        self.assertTrue(evidence.available)
        for field in ("render", "depth", "opacity"):
            self.assertIn(field, captured_package)
        self.assertEqual(len(model), 1)
        self.assertGreater(float(model.get_xyz[0, 2]), 0.0)
        self.assertTrue(bool(captured_package["visibility_filter"][0]))
        self.assertGreater(float(captured_package["radii"][0]), 0.0)

        captured_pairs = (
            (evidence.render_rgb, captured_package["render"]),
            (evidence.depth_accum, captured_package["depth"]),
            (evidence.alpha_accum, captured_package["opacity"]),
        )
        for evidence_tensor, captured_tensor in captured_pairs:
            self.assertEqual(evidence_tensor.data_ptr(), captured_tensor.data_ptr())

        with torch.no_grad():
            oracle = self.production_render(
                camera,
                model,
                pipeline,
                background,
                device=str(CUDA_DEVICE),
            )
        torch.cuda.synchronize(CUDA_DEVICE)
        scene_state_after_oracle = self._scene_state(camera, model)
        self._assert_scene_state_equal(
            scene_state_before,
            scene_state_after_oracle,
            "oracle render mutation",
        )
        torch.testing.assert_close(
            evidence.render_rgb, oracle["render"], atol=ATOL, rtol=RTOL
        )
        torch.testing.assert_close(
            evidence.depth_accum, oracle["depth"], atol=ATOL, rtol=RTOL
        )
        torch.testing.assert_close(
            evidence.alpha_accum, oracle["opacity"], atol=ATOL, rtol=RTOL
        )

    def test_real_outputs_satisfy_cuda_shape_autograd_and_depth_semantics(self):
        camera, model, pipeline, background = self._make_scene()
        observer = PreinsertRenderEvidenceObserverV1(device=CUDA_DEVICE)
        calls = 0

        def forwarding_spy(*args, **kwargs):
            nonlocal calls
            calls += 1
            return self.production_render(*args, **kwargs)

        evidence = observer.capture(
            camera=camera,
            gaussians=model,
            renderer=forwarding_spy,
            pipeline_params=pipeline,
            background=background,
            mapper_update_id=23,
        )
        torch.cuda.synchronize(CUDA_DEVICE)
        self.assertEqual(calls, 1)

        expected = (
            (evidence.render_rgb, (3, HEIGHT, WIDTH)),
            (evidence.depth_accum, (1, HEIGHT, WIDTH)),
            (evidence.alpha_accum, (1, HEIGHT, WIDTH)),
        )
        for tensor, shape in expected:
            self.assertEqual(tuple(tensor.shape), shape)
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertEqual(tensor.device, CUDA_DEVICE)
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)
            self.assertTrue(bool(torch.isfinite(tensor).all()))

        alpha = evidence.alpha_accum
        self.assertTrue(bool((alpha >= -ATOL).all()))
        self.assertTrue(bool((alpha <= 1.0 + ATOL).all()))
        covered = alpha > ALPHA_COVERAGE_EPS
        self.assertTrue(bool(covered.any()))

        background_image = background[:, None, None].expand_as(evidence.render_rgb)
        self.assertFalse(
            torch.allclose(
                evidence.render_rgb,
                background_image,
                atol=ATOL,
                rtol=RTOL,
            )
        )
        covered_depth = evidence.depth_accum[covered]
        covered_alpha = alpha[covered]
        self.assertTrue(bool((covered_depth > 0.0).all()))
        normalized_depth = covered_depth / covered_alpha
        torch.testing.assert_close(
            normalized_depth,
            torch.full_like(normalized_depth, GAUSSIAN_CAMERA_Z),
            atol=ATOL,
            rtol=RTOL,
        )

    def test_capture_does_not_persist_evidence_on_observer(self):
        camera, model, pipeline, background = self._make_scene()
        observer = PreinsertRenderEvidenceObserverV1(device=CUDA_DEVICE)
        calls = 0
        observer_attribute_keys_before = frozenset(vars(observer))
        observer_state_before = dict(vars(observer))
        scene_state_before = self._scene_state(camera, model)

        def forwarding_spy(*args, **kwargs):
            nonlocal calls
            calls += 1
            return self.production_render(*args, **kwargs)

        evidence = observer.capture(
            camera=camera,
            gaussians=model,
            renderer=forwarding_spy,
            pipeline_params=pipeline,
            background=background,
            mapper_update_id=23,
        )
        torch.cuda.synchronize(CUDA_DEVICE)
        scene_state_after_capture = self._scene_state(camera, model)
        self.assertEqual(calls, 1)
        self.assertTrue(evidence.available)
        self.assertEqual(
            frozenset(vars(observer)), observer_attribute_keys_before
        )
        self.assertEqual(vars(observer), observer_state_before)
        self.assertEqual(set(observer.__dict__), {"device"})
        self.assertFalse(
            any(
                isinstance(value, PreinsertRenderEvidenceV1)
                for value in observer.__dict__.values()
            )
        )
        self._assert_scene_state_equal(
            scene_state_before,
            scene_state_after_capture,
            "capture persistence",
        )


if __name__ == "__main__":
    unittest.main()
