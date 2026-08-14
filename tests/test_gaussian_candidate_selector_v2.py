from __future__ import annotations

from contextlib import redirect_stdout
import io
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from src.candidate_selection.gaussian_candidate_selector_v2 import (
    LOG_PREFIX,
    NUMERICAL_EPSILON,
    GaussianCandidateActiveFixedKV2,
    GaussianCandidateSelectorV2,
    build_gaussian_candidate_selector_v2,
)
from src.candidate_selection.gaussian_candidate_active_topk_v1 import (
    GaussianCandidateActiveTopKV1,
)
from src.candidate_selection.preinsert_render_evidence_v1 import (
    PreinsertRenderEvidenceObserverV1,
    PreinsertRenderEvidenceV1,
)
from tests.test_gaussian_candidate_active_topk_v1 import (
    GaussianModelHookHarness,
    load_gaussian_model_extend_from_pcd_seq,
)
from tests.test_preinsert_render_evidence_v1_mapper import _load_production_runtime


ROOT = Path(__file__).resolve().parents[1]
SQRT_2 = math.sqrt(2.0)


class AttrConfig(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


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


class SnapshotGetter:
    def __init__(
        self,
        confidence: torch.Tensor,
        *,
        stale: bool = False,
        source_frame_delta: int = 0,
        error: Exception | None = None,
    ):
        self.confidence = confidence
        self.stale = stale
        self.source_frame_delta = source_frame_delta
        self.error = error
        self.calls = 0

    def __call__(self, camera, require_current=True, *, upsampled=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            buffer_index=camera.buffer_index,
            source_frame_id=camera.source_frame_id + self.source_frame_delta,
            source_timestamp=camera.source_timestamp,
            confidence_source_frame_id=camera.source_frame_id,
            confidence_version=9,
            confidence_up_version=3,
            is_current=not self.stale,
            is_stale=self.stale,
            confidence=self.confidence,
            shape=tuple(self.confidence.shape),
            dtype=str(self.confidence.dtype),
            device=str(self.confidence.device),
            requires_grad=bool(self.confidence.requires_grad),
        )


class GaussianCandidateSelectorV2Tests(unittest.TestCase):
    def setUp(self):
        self.height = 2
        self.width = 4
        self.camera = SimpleNamespace(
            uid=7,
            buffer_index=17,
            source_frame_id=107,
            source_timestamp=7.25,
            image_height=self.height,
            image_width=self.width,
            device="cpu",
            pose=torch.eye(4, dtype=torch.float32),
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
            depth=torch.full((self.height, self.width), 2.0),
            depth_prior=torch.full((self.height, self.width), 2.0),
            original_image=torch.zeros(3, self.height, self.width),
            mask=torch.ones(self.height, self.width, dtype=torch.bool),
        )
        # These camera-space/world-space points project to (u,v):
        # (0,0), (1,0), (2,0), (3,0).
        xyz = torch.tensor(
            [[0.0, 0.0, 2.0], [2.0, 0.0, 2.0], [4.0, 0.0, 2.0], [6.0, 0.0, 2.0]]
        )
        self.candidates = (
            xyz,
            torch.arange(12, dtype=torch.float32).reshape(4, 3, 1),
            torch.full((4, 3), -2.0),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4),
            torch.zeros(4, 1),
        )
        self.confidence = torch.tensor([[0.0, SQRT_2]], dtype=torch.float32)
        self.getter = SnapshotGetter(self.confidence)
        self.selector = GaussianCandidateSelectorV2(
            confidence_snapshot_getter=self.getter,
            logging_enabled=False,
            device="cpu",
        )

    def _evidence(
        self,
        *,
        available=True,
        reason=None,
        alpha=None,
        depth=None,
        rgb=None,
        frame_id=None,
    ):
        if not available:
            return PreinsertRenderEvidenceV1(
                render_rgb=None,
                depth_accum=None,
                alpha_accum=None,
                source_camera_id=self.camera.uid,
                source_frame_id=self.camera.source_frame_id,
                source_buffer_index=self.camera.buffer_index,
                source_timestamp=self.camera.source_timestamp,
                height=self.height,
                width=self.width,
                gaussian_count_before_render=11,
                available=False,
                unavailable_reason=reason or "empty_old_map",
                mapper_update_id=23,
            )
        alpha = (
            torch.tensor([[[0.0, 0.25, 0.5, 1.0], [0.0, 0.0, 0.0, 0.0]]])
            if alpha is None
            else alpha
        )
        depth = 2.0 * alpha if depth is None else depth
        rgb = (
            torch.zeros(3, self.height, self.width)
            if rgb is None
            else rgb
        )
        return PreinsertRenderEvidenceV1(
            render_rgb=rgb,
            depth_accum=depth,
            alpha_accum=alpha,
            source_camera_id=self.camera.uid,
            source_frame_id=(
                self.camera.source_frame_id if frame_id is None else frame_id
            ),
            source_buffer_index=self.camera.buffer_index,
            source_timestamp=self.camera.source_timestamp,
            height=self.height,
            width=self.width,
            gaussian_count_before_render=11,
            available=True,
            unavailable_reason=None,
            mapper_update_id=23,
        )

    def _observe(self, *, evidence=None, init=False, candidates=None):
        candidates = self.candidates if candidates is None else candidates
        return self.selector.observe_before_extend(
            xyz=candidates[0],
            features=candidates[1],
            scales=candidates[2],
            rotations=candidates[3],
            opacities=candidates[4],
            camera=self.camera,
            depthmap=None,
            depth_source="estimated_clean_depth",
            mapper_update_id=23,
            init=init,
            gaussian_before=11,
            preinsert_render_evidence=(
                self._evidence() if evidence is None else evidence
            ),
        )

    def _active_selector(self, strategy, *, getter=None, fixed_k=2):
        return GaussianCandidateActiveFixedKV2(
            confidence_snapshot_getter=self.getter if getter is None else getter,
            logging_enabled=False,
            device="cpu",
            strategy=strategy,
            fixed_k=fixed_k,
            diversity_strength=0.1,
        )

    def _select(
        self,
        strategy,
        *,
        evidence=None,
        getter=None,
        candidates=None,
        init=False,
        fixed_k=2,
    ):
        candidates = self.candidates if candidates is None else candidates
        selector = self._active_selector(
            strategy,
            getter=getter,
            fixed_k=fixed_k,
        )
        result = selector.select_before_extend(
            xyz=candidates[0],
            features=candidates[1],
            scales=candidates[2],
            rotations=candidates[3],
            opacities=candidates[4],
            camera=self.camera,
            depthmap=None,
            depth_source="estimated_clean_depth",
            mapper_update_id=23,
            init=init,
            gaussian_before=11,
            preinsert_render_evidence=(
                self._evidence() if evidence is None else evidence
            ),
        )
        return selector, result

    def _weighted_quality(
        self,
        *,
        alpha,
        rgb_l1,
        depth_saturated,
        rgb_valid,
        depth_valid,
        confidence=1.0,
    ):
        alpha = torch.as_tensor(alpha, dtype=torch.float32).reshape(-1)
        count = int(alpha.shape[0])
        render_components = {
            "alpha": alpha,
            "coverage": 1.0 - alpha,
            "coverage_valid": torch.ones(count, dtype=torch.bool),
            "rgb_l1": torch.as_tensor(rgb_l1, dtype=torch.float32).reshape(-1),
            "rgb_valid": torch.as_tensor(rgb_valid, dtype=torch.bool).reshape(-1),
            "depth_saturated": torch.as_tensor(
                depth_saturated,
                dtype=torch.float32,
            ).reshape(-1),
            "depth_valid": torch.as_tensor(
                depth_valid,
                dtype=torch.bool,
            ).reshape(-1),
        }
        confidence_components = {
            "normalized": torch.full(
                (count,),
                float(confidence),
                dtype=torch.float32,
            ),
            "valid": torch.ones(count, dtype=torch.bool),
        }
        return GaussianCandidateActiveFixedKV2._weighted_evidence_quality(
            render_components,
            confidence_components,
        )

    def _mapper_init_config(self, *, v1_mode="off", v2_mode="observe", evidence_mode="observe", m01_mode="disabled"):
        online_opt = AttrConfig(
            batch_mode=False,
            iters=1,
            n_last_frames=1,
            n_rand_frames=0,
            filter=AttrConfig(),
            optimize_poses=False,
        )
        mapping = AttrConfig(
            delay=0,
            warmup=0,
            online_opt=online_opt,
            opt_params=AttrConfig(init_lr=1.0),
            loss=AttrConfig(),
            pipeline_params=AttrConfig(),
            use_spherical_harmonics=False,
            input=AttrConfig(),
            refinement=AttrConfig(optimize_poses=False),
            feedback=AttrConfig(disps=False, poses=False),
            resource_admission=AttrConfig(
                mode=m01_mode,
                fixed_budget=600,
                selection="deterministic_uniform",
            ),
            candidate_observer=AttrConfig(enabled=False),
            candidate_selector=AttrConfig(mode="off"),
            candidate_selector_v1=AttrConfig(
                mode=v1_mode,
                budget=600,
                logging=AttrConfig(enabled=True),
            ),
            candidate_selector_v2=AttrConfig(
                mode=v2_mode,
                strategy="evidence",
                fixed_k=600,
                diversity_strength=0.1,
                logging=AttrConfig(enabled=True),
            ),
            preinsert_render_evidence_v1=AttrConfig(mode=evidence_mode),
        )
        return AttrConfig(
            device="cpu",
            mode="rgbd",
            evaluate=False,
            mapping=mapping,
            tracking=AttrConfig(upsample=True),
        )

    def test_factory_exact_off_returns_none_without_touching_dependencies(self):
        trap = object()
        self.assertIsNone(
            build_gaussian_candidate_selector_v2(
                None,
                candidate_selector_v1=trap,
                resource_admission_mode="fixed_budget",
                preinsert_render_evidence_v1=trap,
                confidence_snapshot_getter=trap,
                device="not-a-device",
            )
        )
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

    def test_factory_rejects_unknown_fields_and_non_observe_modes(self):
        kwargs = dict(
            candidate_selector_v1=None,
            resource_admission_mode="disabled",
            preinsert_render_evidence_v1=SimpleNamespace(mode="observe"),
            confidence_snapshot_getter=lambda *args, **kwargs: None,
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            build_gaussian_candidate_selector_v2(
                {"mode": "off", "surprise": True}, **kwargs
            )
        for mode in ("active", "dynamic"):
            with self.subTest(mode=mode), self.assertRaisesRegex(
                ValueError, "must be one of"
            ):
                build_gaussian_candidate_selector_v2({"mode": mode}, **kwargs)

    def test_factory_enforces_v1_m01_and_preinsert_mutual_exclusion(self):
        base = dict(
            config={"mode": "observe", "logging": {"enabled": True}},
            candidate_selector_v1=None,
            resource_admission_mode="disabled",
            preinsert_render_evidence_v1=SimpleNamespace(mode="observe"),
            confidence_snapshot_getter=lambda *args, **kwargs: None,
            device="cpu",
        )
        for key, value, message in (
            ("candidate_selector_v1", object(), "candidate_selector_v1.mode=off"),
            ("resource_admission_mode", "observe", "resource_admission.mode=disabled"),
            ("preinsert_render_evidence_v1", None, "preinsert_render_evidence_v1.mode=observe"),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                build_gaussian_candidate_selector_v2(**{**base, key: value})

    def test_gaussian_mapper_init_enforces_v2_configuration_conflicts(self):
        runtime = _load_production_runtime()
        mapper_module = runtime.mapper_module
        slam = SimpleNamespace(video=object(), output="unused")

        def v1_builder(config, **kwargs):
            del kwargs
            return object() if config.get("mode") != "off" else None

        def preinsert_builder(config, **kwargs):
            del kwargs
            return (
                SimpleNamespace(mode="observe")
                if config.get("mode") == "observe"
                else None
            )

        patches = (
            mock.patch.object(mapper_module, "ResourceAdmission", return_value=object()),
            mock.patch.object(mapper_module, "build_historical_camera_scheduler", return_value=None),
            mock.patch.object(mapper_module, "build_gaussian_candidate_observer", return_value=None),
            mock.patch.object(mapper_module, "build_gaussian_candidate_selector", return_value=None),
            mock.patch.object(mapper_module, "build_gaussian_candidate_selector_v1", side_effect=v1_builder),
            mock.patch.object(mapper_module, "build_preinsert_render_evidence_v1", side_effect=preinsert_builder),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        conflicts = (
            (
                self._mapper_init_config(m01_mode="fixed_budget"),
                "resource_admission.mode=disabled",
            ),
            (
                self._mapper_init_config(v1_mode="active"),
                "candidate_selector_v1.mode=off",
            ),
            (
                self._mapper_init_config(evidence_mode="off"),
                "preinsert_render_evidence_v1.mode=observe",
            ),
        )
        for config, message in conflicts:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                mapper = runtime.GaussianMapper.__new__(runtime.GaussianMapper)
                runtime.GaussianMapper.__init__(mapper, config, slam)

    def test_base_config_defaults_v2_off(self):
        config = (ROOT / "configs/mapping/base.yaml").read_text(encoding="utf-8")
        block = config.split("candidate_selector_v2:", 1)[1].split(
            "preinsert_render_evidence_v1:", 1
        )[0]
        self.assertIn('mode: "off"', block)
        self.assertIn('fixed_k: 600', block)
        self.assertIn('diversity_strength: 0.1', block)

    def test_init_bypass_preserves_candidates_without_confidence(self):
        references = list(self.candidates)
        before = [
            (value.data_ptr(), value._version, value.clone())
            for value in references
        ]
        token = self._observe(evidence=None, init=True)
        self.assertEqual(token.fields["reason"], "init_bypass")
        self.assertEqual(self.getter.calls, 0)
        self.assertFalse(token.fields["selected_indices_created"])
        for actual, reference, (pointer, version, frozen) in zip(
            self.candidates, references, before
        ):
            self.assertIs(actual, reference)
            self.assertEqual(actual.data_ptr(), pointer)
            self.assertEqual(actual._version, version)
            torch.testing.assert_close(actual, frozen, atol=0.0, rtol=0.0)

    def test_empty_candidate_is_structured_and_does_not_read_confidence(self):
        token = self.selector.observe_empty(
            camera=self.camera,
            mapper_update_id=23,
            init=False,
            gaussian_before=11,
            preinsert_render_evidence=self._evidence(),
        )
        summary = self.selector.record_after_extend(
            token,
            admitted_candidate_count=0,
            dropped_candidate_count=0,
            gaussian_after_extend=11,
        )
        self.assertEqual(summary.fields["reason"], "empty_candidate")
        self.assertEqual(summary.fields["candidate_count"], 0)
        self.assertTrue(summary.fields["all_candidates_forwarded"])
        self.assertEqual(self.getter.calls, 0)

    def test_evidence_unavailable_is_structured_and_candidates_continue(self):
        token = self._observe(evidence=self._evidence(available=False))
        self.assertFalse(token.fields["evidence_available"])
        self.assertEqual(token.fields["evidence_reason"], "empty_old_map")
        summary = self.selector.record_after_extend(
            token,
            admitted_candidate_count=4,
            dropped_candidate_count=0,
            gaussian_after_extend=15,
        )
        self.assertTrue(summary.fields["all_candidates_forwarded"])
        self.assertTrue(summary.fields["conservation_check"])

    def test_stale_confidence_is_structured_without_fixed600_fallback(self):
        getter = SnapshotGetter(self.confidence, stale=True)
        selector = GaussianCandidateSelectorV2(
            confidence_snapshot_getter=getter,
            logging_enabled=False,
            device="cpu",
        )
        token = selector.observe_before_extend(
            xyz=self.candidates[0],
            features=self.candidates[1],
            scales=self.candidates[2],
            rotations=self.candidates[3],
            opacities=self.candidates[4],
            camera=self.camera,
            depthmap=None,
            depth_source="estimated_clean_depth",
            mapper_update_id=23,
            init=False,
            gaussian_before=11,
            preinsert_render_evidence=self._evidence(),
        )
        self.assertEqual(token.fields["confidence_reason"], "stale_confidence")
        self.assertFalse(token.fields["confidence_available"])
        self.assertFalse(token.fields["selected_indices_created"])
        self.assertFalse(hasattr(selector, "select_before_extend"))
        self.assertTrue(token.fields["evidence_available"])
        self.assertEqual(token.fields["alpha_valid_count"], 4)
        self.assertEqual(token.fields["coverage_gap_valid_count"], 4)
        self.assertEqual(token.fields["rgb_l1_valid_count"], 4)
        self.assertEqual(token.fields["depth_relative_error_valid_count"], 3)

    def test_missing_and_identity_mismatched_confidence_are_structured(self):
        getters = (
            (
                SnapshotGetter(
                    self.confidence,
                    error=RuntimeError("confidence is not valid"),
                ),
                "missing_confidence",
            ),
            (
                SnapshotGetter(self.confidence, source_frame_delta=1),
                "confidence_identity_mismatch",
            ),
        )
        for getter, reason in getters:
            with self.subTest(reason=reason):
                selector = GaussianCandidateSelectorV2(
                    confidence_snapshot_getter=getter,
                    logging_enabled=False,
                    device="cpu",
                )
                token = selector.observe_before_extend(
                    xyz=self.candidates[0],
                    features=self.candidates[1],
                    scales=self.candidates[2],
                    rotations=self.candidates[3],
                    opacities=self.candidates[4],
                    camera=self.camera,
                    depthmap=None,
                    depth_source="estimated_clean_depth",
                    mapper_update_id=23,
                    init=False,
                    gaussian_before=11,
                    preinsert_render_evidence=self._evidence(),
                )
                self.assertFalse(token.fields["confidence_available"])
                self.assertEqual(token.fields["confidence_reason"], reason)
                self.assertFalse(token.fields["selected_indices_created"])

    def test_evidence_camera_identity_mismatch_is_structured(self):
        token = self._observe(evidence=self._evidence(frame_id=999))
        self.assertFalse(token.fields["evidence_available"])
        self.assertEqual(
            token.fields["evidence_reason"], "invalid_preinsert_render_evidence"
        )
        self.assertEqual(
            token.fields["evidence_error"]["type"], "PreinsertRenderEvidenceError"
        )

    def test_confidence_normalization_clamps_to_unit_interval(self):
        self.confidence.copy_(torch.tensor([[-1.0, 2.0 * SQRT_2]]))
        token = self._observe()
        stats = token.fields["confidence_norm"]
        self.assertTrue(token.fields["confidence_available"])
        self.assertEqual(stats["min"], 0.0)
        self.assertEqual(stats["max"], 1.0)
        self.assertAlmostEqual(stats["mean"], 0.5)

    def test_alpha_coverage_depth_and_rgb_use_frozen_domains(self):
        self.camera.original_image[:, 0, 1] = 255.0
        render_rgb = torch.zeros(3, self.height, self.width)
        render_rgb[:, 0, 1] = 0.5
        token = self._observe(evidence=self._evidence(rgb=render_rgb))
        fields = token.fields
        self.assertEqual(fields["alpha_valid_count"], 4)
        self.assertAlmostEqual(fields["alpha"]["min"], 0.0)
        self.assertAlmostEqual(fields["alpha"]["max"], 1.0)
        self.assertAlmostEqual(fields["coverage_gap"]["min"], 0.0)
        self.assertAlmostEqual(fields["coverage_gap"]["max"], 1.0)
        self.assertEqual(fields["depth_relative_error_valid_count"], 3)
        self.assertEqual(fields["depth_relative_error"]["max"], 0.0)
        self.assertAlmostEqual(fields["rgb_l1"]["max"], 0.5)

    def test_mask_excludes_rgb_and_depth_samples(self):
        self.camera.mask[0, 3] = False
        token = self._observe()
        self.assertEqual(token.fields["mask_valid_count"], 3)
        self.assertEqual(token.fields["rgb_l1_valid_count"], 3)
        self.assertEqual(token.fields["depth_relative_error_valid_count"], 2)

    def test_single_channel_mask_matches_two_dimensional_mask(self):
        token_2d = self._observe()
        self.camera.mask = self.camera.mask.unsqueeze(0)
        token_3d = self._observe()
        for field in (
            "mask_valid_count",
            "rgb_l1_valid_count",
            "depth_relative_error_valid_count",
            "rgb_l1",
            "depth_relative_error",
        ):
            self.assertEqual(token_3d.fields[field], token_2d.fields[field])

    def test_tiny_positive_alpha_keeps_coverage_but_invalidates_depth(self):
        alpha = torch.tensor(
            [[[1.0e-7, 0.25, 0.5, 1.0], [0.0, 0.0, 0.0, 0.0]]]
        )
        token = self._observe(evidence=self._evidence(alpha=alpha))
        self.assertEqual(token.fields["alpha_valid_count"], 4)
        self.assertEqual(token.fields["coverage_gap_valid_count"], 4)
        self.assertEqual(token.fields["depth_relative_error_valid_count"], 3)
        self.assertEqual(token.fields["depth_relative_error_invalid_count"], 1)
        self.assertEqual(token.fields["depth_relative_error"]["max"], 0.0)

    def test_lowres_cell_multiplicity_and_redundancy(self):
        token = self._observe()
        self.assertEqual(token.fields["confidence_sampling_method"], "lowres_cell_floor_v1")
        self.assertEqual(token.fields["multiplicity"]["min"], 2.0)
        self.assertEqual(token.fields["multiplicity"]["max"], 2.0)
        self.assertEqual(token.fields["cell_redundancy"]["min"], 0.5)
        self.assertEqual(token.fields["cell_redundancy"]["max"], 0.5)

    def test_single_candidate_cell_has_no_redundancy(self):
        candidates = tuple(value[:1] for value in self.candidates)
        token = self._observe(candidates=candidates)
        self.assertEqual(token.fields["multiplicity_valid_count"], 1)
        self.assertEqual(token.fields["multiplicity"]["min"], 1.0)
        self.assertEqual(token.fields["multiplicity"]["max"], 1.0)
        self.assertEqual(token.fields["cell_redundancy"]["min"], 0.0)
        self.assertEqual(token.fields["cell_redundancy"]["max"], 0.0)

    def test_observe_keeps_all_five_candidate_tensor_objects_unchanged(self):
        states = [
            (value.data_ptr(), value._version, value.clone())
            for value in self.candidates
        ]
        token = self._observe()
        self.assertFalse(token.fields["selected_indices_created"])
        self.assertTrue(token.fields["observer_no_mutation"])
        for value, (pointer, version, frozen) in zip(self.candidates, states):
            self.assertEqual(value.data_ptr(), pointer)
            self.assertEqual(value._version, version)
            torch.testing.assert_close(value, frozen, atol=0.0, rtol=0.0)

    def test_observer_has_no_index_select_or_renderer_api(self):
        source = inspect.getsource(GaussianCandidateSelectorV2)
        self.assertNotIn("index_select", source)
        self.assertNotIn("gaussian_renderer", source)
        self.assertNotIn("renderer(", source)
        self.assertFalse(hasattr(self.selector, "select_before_extend"))

    def test_observer_does_not_persist_evidence_or_snapshot_tensors(self):
        keys_before = frozenset(vars(self.selector))
        normal_token = self._observe()
        unavailable_token = self._observe(evidence=self._evidence(available=False))
        stale_selector = GaussianCandidateSelectorV2(
            confidence_snapshot_getter=SnapshotGetter(self.confidence, stale=True),
            logging_enabled=False,
            device="cpu",
        )
        stale_token = stale_selector.observe_before_extend(
            xyz=self.candidates[0],
            features=self.candidates[1],
            scales=self.candidates[2],
            rotations=self.candidates[3],
            opacities=self.candidates[4],
            camera=self.camera,
            depthmap=None,
            depth_source="estimated_clean_depth",
            mapper_update_id=23,
            init=False,
            gaussian_before=11,
            preinsert_render_evidence=self._evidence(),
        )
        self.assertEqual(frozenset(vars(self.selector)), keys_before)
        self.assertFalse(
            any(
                isinstance(value, PreinsertRenderEvidenceV1)
                for value in vars(self.selector).values()
            )
        )
        self.assertFalse(
            any(isinstance(value, torch.Tensor) for value in vars(self.selector).values())
        )
        for token in (normal_token, unavailable_token, stale_token):
            self.assertFalse(_contains_tensor(token.fields))

    def test_elapsed_and_sync_diagnostics_have_honest_semantics(self):
        normal = self._observe()
        init = self._observe(init=True)
        empty = self.selector.observe_empty(
            camera=self.camera,
            mapper_update_id=23,
            init=False,
            gaussian_before=11,
            preinsert_render_evidence=self._evidence(),
        )
        for token in (normal, init, empty):
            self.assertIn("host_wall_elapsed_ms", token.fields)
            self.assertNotIn("cpu_elapsed_ms", token.fields)
            self.assertIn("diagnostic_gpu_to_cpu_sync_expected", token.fields)
            self.assertNotIn("diagnostic_gpu_to_cpu_sync", token.fields)
            self.assertFalse(token.fields["diagnostic_gpu_to_cpu_sync_expected"])
        self.assertGreaterEqual(normal.fields["host_wall_elapsed_ms"], 0.0)
        expected = self.selector._diagnostic_gpu_to_cpu_sync_expected
        self.assertTrue(expected(device_type="cuda", init=False, candidate_count=4))
        self.assertFalse(expected(device_type="cuda", init=True, candidate_count=4))
        self.assertFalse(expected(device_type="cuda", init=False, candidate_count=0))
        self.assertFalse(expected(device_type="cpu", init=False, candidate_count=4))
        # Evidence availability does not alter the conservative CUDA path
        # expectation: confidence diagnostics are still attempted.
        unavailable = self._evidence(available=False)
        self.assertFalse(unavailable.available)
        self.assertTrue(expected(device_type="cuda", init=False, candidate_count=4))

    def test_record_after_extend_reports_forwarding_conservation_failure(self):
        token = self._observe()
        summary = self.selector.record_after_extend(
            token,
            admitted_candidate_count=4,
            dropped_candidate_count=0,
            gaussian_after_extend=14,
        )
        self.assertEqual(summary.fields["status"], "error")
        self.assertEqual(
            summary.fields["reason"], "observe_only_forwarding_contract_failed"
        )
        self.assertFalse(summary.fields["conservation_check"])
        self.assertEqual(summary.fields["error"]["type"], "ForwardingContractError")

    def test_logging_emits_only_aggregate_json_event(self):
        selector = GaussianCandidateSelectorV2(
            confidence_snapshot_getter=self.getter,
            logging_enabled=True,
            device="cpu",
        )
        token = selector.observe_before_extend(
            xyz=self.candidates[0],
            features=self.candidates[1],
            scales=self.candidates[2],
            rotations=self.candidates[3],
            opacities=self.candidates[4],
            camera=self.camera,
            depthmap=None,
            depth_source="estimated_clean_depth",
            mapper_update_id=23,
            init=False,
            gaussian_before=11,
            preinsert_render_evidence=self._evidence(),
        )
        stream = io.StringIO()
        with redirect_stdout(stream):
            summary = selector.record_after_extend(
                token,
                admitted_candidate_count=4,
                dropped_candidate_count=0,
                gaussian_after_extend=15,
            )
        line = stream.getvalue().strip()
        self.assertTrue(line.startswith(LOG_PREFIX + " "))
        event = json.loads(line.split(" ", 1)[1])
        self.assertEqual(event, summary.to_event())
        self.assertFalse(event["selected_indices_created"])
        self.assertTrue(event["all_candidates_forwarded"])
        self.assertNotIn("candidate_values", event)

    def test_preinsert_observer_capture_is_not_called_by_v2(self):
        class Trap(PreinsertRenderEvidenceObserverV1):
            def capture(self, **kwargs):
                raise AssertionError("V2 must reuse evidence and never render again.")

        trap = Trap(device="cpu")
        evidence = self._evidence()
        selector = build_gaussian_candidate_selector_v2(
            {"mode": "observe", "logging": {"enabled": True}},
            candidate_selector_v1=None,
            resource_admission_mode="disabled",
            preinsert_render_evidence_v1=trap,
            confidence_snapshot_getter=self.getter,
            device="cpu",
        )
        token = selector.observe_before_extend(
            xyz=self.candidates[0],
            features=self.candidates[1],
            scales=self.candidates[2],
            rotations=self.candidates[3],
            opacities=self.candidates[4],
            camera=self.camera,
            depthmap=None,
            depth_source="estimated_clean_depth",
            mapper_update_id=23,
            init=False,
            gaussian_before=11,
            preinsert_render_evidence=evidence,
        )
        self.assertTrue(token.fields["evidence_available"])

    def test_gaussian_model_exact_off_uses_original_metadata_path(self):
        extend_from_pcd_seq = load_gaussian_model_extend_from_pcd_seq()
        harness = GaussianModelHookHarness(self.candidates)
        calls = []
        original_generation = harness.create_pcd_from_image

        def generation(*args, collect_candidate_metadata=False, **kwargs):
            calls.append(collect_candidate_metadata)
            return original_generation(
                *args,
                collect_candidate_metadata=collect_candidate_metadata,
                **kwargs,
            )

        harness.create_pcd_from_image = generation
        extend_from_pcd_seq(
            harness,
            self.camera,
            self.camera.uid,
            init=False,
            candidate_selector_v2=None,
            mapper_update_id=23,
        )
        self.assertEqual(calls, [False])
        self.assertEqual(self.getter.calls, 0)
        self.assertEqual(harness.extend_calls, 1)
        self.assertEqual(harness.last_extended_count, 4)

    def test_gaussian_model_observe_hook_preserves_extend_contract(self):
        extend_from_pcd_seq = load_gaussian_model_extend_from_pcd_seq()
        harness = GaussianModelHookHarness(self.candidates)
        tokens = []
        summaries = []

        class RecordingSelector(GaussianCandidateSelectorV2):
            def observe_before_extend(inner_self, **kwargs):
                token = super().observe_before_extend(**kwargs)
                tokens.append(token)
                return token

            def record_after_extend(inner_self, token, **kwargs):
                summary = super().record_after_extend(token, **kwargs)
                summaries.append(summary)
                return summary

        selector = RecordingSelector(
            confidence_snapshot_getter=self.getter,
            logging_enabled=False,
            device="cpu",
        )
        pointers = [value.data_ptr() for value in self.candidates]
        extend_from_pcd_seq(
            harness,
            self.camera,
            self.camera.uid,
            init=False,
            candidate_selector_v2=selector,
            mapper_update_id=23,
            preinsert_render_evidence=self._evidence(),
        )
        self.assertEqual(self.getter.calls, 1)
        self.assertEqual(len(tokens), 1)
        self.assertEqual(len(summaries), 1)
        self.assertTrue(summaries[0].fields["all_candidates_forwarded"])
        self.assertEqual(harness.extend_calls, 1)
        self.assertEqual(harness.last_extended_count, 4)
        self.assertEqual([value.data_ptr() for value in self.candidates], pointers)

    def test_active_factory_contract_and_mutual_exclusion(self):
        config = {
            "mode": "active_fixed_k",
            "strategy": "evidence_projected_cell",
            "fixed_k": 600,
            "diversity_strength": 0.1,
            "logging": {"enabled": True},
        }
        kwargs = dict(
            candidate_selector_v1=None,
            resource_admission_mode="disabled",
            preinsert_render_evidence_v1=SimpleNamespace(mode="observe"),
            confidence_snapshot_getter=self.getter,
            device="cpu",
        )
        selector = build_gaussian_candidate_selector_v2(config, **kwargs)
        self.assertIsInstance(selector, GaussianCandidateActiveFixedKV2)
        self.assertTrue(selector.is_active)
        self.assertEqual(selector.fixed_k, 600)
        self.assertEqual(selector.strategy, "evidence_projected_cell")
        for key, value, message in (
            ("candidate_selector_v1", object(), "candidate_selector_v1.mode=off"),
            ("resource_admission_mode", "fixed_budget", "resource_admission.mode=disabled"),
            ("preinsert_render_evidence_v1", None, "preinsert_render_evidence_v1.mode=observe"),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                build_gaussian_candidate_selector_v2(
                    config,
                    **{**kwargs, key: value},
                )
        with self.assertRaisesRegex(ValueError, "strategy must be one of"):
            build_gaussian_candidate_selector_v2(
                {**config, "strategy": "unknown"},
                **kwargs,
            )

    def test_active_coverage_ranking_ignores_confidence(self):
        getter = SnapshotGetter(
            torch.tensor([[SQRT_2, 0.0]], dtype=torch.float32)
        )
        selector, result = self._select("coverage", getter=getter)
        self.assertEqual(result.selected_indices.tolist(), [0, 1])
        self.assertEqual(getter.calls, 0)
        self.assertEqual(result.token.fields["reason"], "coverage_fixed_k")
        self.assertEqual(result.token.fields["selected_count"], 2)
        self.assertIsNone(result.token.fields["fallback_reason"])
        self.assertEqual(selector.strategy, "coverage")

    def test_active_evidence_formula_ranks_valid_component_weighted_mean(self):
        confidence = torch.full((1, 4), SQRT_2, dtype=torch.float32)
        getter = SnapshotGetter(confidence)
        alpha = torch.full((1, self.height, self.width), 0.5)
        depth = 2.0 * alpha
        depth[0, 0, 1] = 4.0 * alpha[0, 0, 1]
        rgb = torch.zeros(3, self.height, self.width)
        rgb[:, 0, 2] = 1.0
        _, result = self._select(
            "evidence",
            getter=getter,
            evidence=self._evidence(alpha=alpha, depth=depth, rgb=rgb),
        )
        self.assertEqual(result.selected_indices.tolist(), [1, 2])
        self.assertEqual(result.token.fields["reason"], "evidence_fixed_k")
        self.assertEqual(result.token.fields["coverage_valid_count"], 4)
        self.assertEqual(result.token.fields["rgb_l1_valid_count"], 4)
        self.assertEqual(
            result.token.fields["depth_relative_error_valid_count"], 4
        )

    def test_active_projection_and_lowres_cells_match_frozen_v1_semantics(self):
        selector = self._active_selector("evidence")
        projection = selector._active_project(self.candidates[0], self.camera)
        frozen_u, frozen_v, frozen_z, frozen_valid = (
            GaussianCandidateActiveTopKV1.project_world_to_source_pixels(
                selector,
                self.candidates[0],
                self.camera,
            )
        )
        torch.testing.assert_close(projection["u"], frozen_u)
        torch.testing.assert_close(projection["v"], frozen_v)
        torch.testing.assert_close(projection["z"], frozen_z)
        torch.testing.assert_close(projection["valid"], frozen_valid)
        components = selector._active_confidence_components(
            xyz=self.candidates[0],
            camera=self.camera,
            depthmap=None,
            depth_source="estimated_clean_depth",
            projection=projection,
        )
        frozen_conf_u, frozen_conf_v, _, _ = (
            GaussianCandidateActiveTopKV1.map_source_pixels_to_confidence(
                frozen_u,
                frozen_v,
                source_resolution=(self.height, self.width),
                confidence_resolution=tuple(self.confidence.shape),
            )
        )
        self.assertTrue(
            torch.equal(
                components["cell_ids"],
                frozen_conf_v * self.confidence.shape[1] + frozen_conf_u,
            )
        )

    def test_projected_cell_diminishing_returns_changes_only_winner_identity(self):
        alpha = torch.tensor(
            [[[0.10, 0.102, 0.11, 1.0], [0.0, 0.0, 0.0, 0.0]]]
        )
        evidence = self._evidence(alpha=alpha, depth=2.0 * alpha)
        _, evidence_result = self._select("evidence", evidence=evidence)
        _, diversity_result = self._select(
            "evidence_projected_cell",
            evidence=evidence,
        )
        self.assertEqual(evidence_result.selected_indices.tolist(), [0, 1])
        self.assertEqual(diversity_result.selected_indices.tolist(), [0, 2])
        self.assertEqual(
            diversity_result.token.fields["selected_unique_cell_count"], 2
        )
        self.assertEqual(diversity_result.selected_indices.numel(), 2)

    def test_low_alpha_continuously_downweights_rgb_and_depth(self):
        alpha = torch.tensor(
            [[[
                NUMERICAL_EPSILON / 2.0,
                0.2,
                0.3,
                0.4,
            ], [0.0, 0.0, 0.0, 0.0]]]
        )
        baseline_depth = 2.0 * alpha
        extreme_depth = baseline_depth.clone()
        extreme_depth[0, 0, 0] = 1000.0
        baseline_rgb = torch.zeros(3, self.height, self.width)
        extreme_rgb = baseline_rgb.clone()
        extreme_rgb[:, 0, 0] = 1000.0
        _, baseline = self._select(
            "evidence",
            getter=SnapshotGetter(torch.full((1, 4), SQRT_2)),
            evidence=self._evidence(
                alpha=alpha,
                depth=baseline_depth,
                rgb=baseline_rgb,
            ),
        )
        _, extreme = self._select(
            "evidence",
            getter=SnapshotGetter(torch.full((1, 4), SQRT_2)),
            evidence=self._evidence(
                alpha=alpha,
                depth=extreme_depth,
                rgb=extreme_rgb,
            ),
        )
        self.assertTrue(torch.equal(baseline.selected_indices, extreme.selected_indices))
        self.assertEqual(extreme.token.fields["coverage_valid_count"], 4)
        self.assertEqual(extreme.token.fields["rgb_l1_valid_count"], 4)
        self.assertEqual(
            extreme.token.fields["depth_relative_error_valid_count"], 4
        )

    def test_alpha_zero_residuals_cannot_change_coverage_only_quality(self):
        quality, valid, rgb_valid, depth_valid = self._weighted_quality(
            alpha=[0.0],
            rgb_l1=[1.0],
            depth_saturated=[1.0],
            rgb_valid=[True],
            depth_valid=[True],
        )
        self.assertTrue(valid.item())
        self.assertTrue(rgb_valid.item())
        self.assertTrue(depth_valid.item())
        self.assertEqual(quality.item(), 1.0)

    def test_alpha_weighted_quality_is_continuous_around_numerical_epsilon(self):
        alpha = torch.tensor(
            [
                0.0,
                NUMERICAL_EPSILON / 2.0,
                NUMERICAL_EPSILON,
                NUMERICAL_EPSILON * 2.0,
                0.5,
            ],
            dtype=torch.float32,
        )
        quality, valid, _, _ = self._weighted_quality(
            alpha=alpha,
            rgb_l1=torch.ones_like(alpha),
            depth_saturated=torch.zeros_like(alpha),
            rgb_valid=torch.ones_like(alpha, dtype=torch.bool),
            depth_valid=torch.zeros_like(alpha, dtype=torch.bool),
        )
        expected = torch.reciprocal(1.0 + alpha)
        torch.testing.assert_close(quality, expected, atol=1.0e-7, rtol=1.0e-7)
        self.assertTrue(torch.all(valid))
        self.assertLess(
            abs(quality[1].item() - quality[2].item()),
            2.0 * NUMERICAL_EPSILON,
        )
        self.assertLess(
            abs(quality[2].item() - quality[3].item()),
            2.0 * NUMERICAL_EPSILON,
        )

    def test_large_alpha_valid_residuals_change_quality(self):
        quality, _, _, _ = self._weighted_quality(
            alpha=[0.8, 0.8],
            rgb_l1=[0.0, 1.0],
            depth_saturated=[0.0, 1.0],
            rgb_valid=[True, True],
            depth_valid=[True, True],
        )
        self.assertGreater(quality[1].item(), quality[0].item())

    def test_component_specific_denominators_and_coverage_degradation(self):
        cases = (
            (False, True, 0.6),
            (True, False, (0.5 + 0.5 * 0.6) / 1.5),
            (False, False, 0.5),
        )
        for rgb_is_valid, depth_is_valid, expected in cases:
            with self.subTest(
                rgb_valid=rgb_is_valid,
                depth_valid=depth_is_valid,
            ):
                quality, _, _, _ = self._weighted_quality(
                    alpha=[0.5],
                    rgb_l1=[0.6],
                    depth_saturated=[0.8],
                    rgb_valid=[rgb_is_valid],
                    depth_valid=[depth_is_valid],
                )
                self.assertAlmostEqual(quality.item(), expected, places=6)

    def test_partial_nonfinite_components_are_removed_and_renormalized(self):
        alpha = torch.full((1, self.height, self.width), 0.5)
        depth = 2.0 * alpha
        depth[0, 0, 1] = float("nan")
        rgb = torch.zeros(3, self.height, self.width)
        rgb[:, 0, 0] = float("nan")
        _, result = self._select(
            "evidence",
            getter=SnapshotGetter(torch.full((1, 4), SQRT_2)),
            evidence=self._evidence(alpha=alpha, depth=depth, rgb=rgb),
        )
        self.assertEqual(result.selected_indices.tolist(), [0, 1])
        self.assertEqual(result.token.fields["rgb_l1_valid_count"], 3)
        self.assertEqual(
            result.token.fields["depth_relative_error_valid_count"], 3
        )

    def test_active_fallback_contracts(self):
        unavailable = self._evidence(available=False, reason="empty_old_map")
        cases = (
            (
                "coverage",
                self.getter,
                unavailable,
                [0, 3],
                "deterministic_uniform_fallback",
            ),
            (
                "evidence",
                SnapshotGetter(self.confidence, stale=True),
                self._evidence(),
                [0, 1],
                "coverage_only_fallback",
            ),
            (
                "evidence",
                self.getter,
                unavailable,
                [2, 3],
                "confidence_topk_fallback",
            ),
            (
                "evidence",
                SnapshotGetter(
                    self.confidence,
                    error=RuntimeError("confidence is not valid"),
                ),
                unavailable,
                [0, 3],
                "deterministic_uniform_fallback",
            ),
        )
        for strategy, getter, evidence, indices, reason in cases:
            with self.subTest(reason=reason):
                _, result = self._select(
                    strategy,
                    getter=getter,
                    evidence=evidence,
                )
                self.assertEqual(result.selected_indices.tolist(), indices)
                self.assertEqual(result.token.fields["reason"], reason)
                self.assertIsNotNone(result.token.fields["fallback_reason"])

    def test_confidence_fallback_uses_frozen_raw_v1_ranking(self):
        confidence = torch.tensor(
            [[2.0 * SQRT_2, 4.0 * SQRT_2, 3.0 * SQRT_2, 0.0]],
            dtype=torch.float32,
        )
        _, result = self._select(
            "evidence",
            getter=SnapshotGetter(confidence),
            evidence=self._evidence(available=False),
        )
        # Normalized values 0..2 all clamp to one; raw V1 ranking must retain
        # indices 1 and 2 rather than turning index 0 into an artificial tie.
        self.assertEqual(result.selected_indices.tolist(), [1, 2])
        self.assertEqual(result.token.fields["reason"], "confidence_topk_fallback")

    def test_active_boundaries_ties_and_five_tensor_alignment(self):
        empty = tuple(value[:0] for value in self.candidates)
        _, empty_result = self._select(
            "coverage",
            candidates=empty,
            fixed_k=2,
        )
        self.assertEqual(empty_result.xyz.shape[0], 0)
        self.assertIsNone(empty_result.selected_indices)
        small = tuple(value[:2] for value in self.candidates)
        _, small_result = self._select(
            "coverage",
            candidates=small,
            fixed_k=2,
        )
        self.assertIsNone(small_result.selected_indices)
        for actual, original in zip(
            (
                small_result.xyz,
                small_result.features,
                small_result.scales,
                small_result.rotations,
                small_result.opacities,
            ),
            small,
        ):
            self.assertIs(actual, original)

        tied_alpha = torch.full((1, self.height, self.width), 0.5)
        _, result = self._select(
            "coverage",
            evidence=self._evidence(alpha=tied_alpha, depth=2.0 * tied_alpha),
        )
        self.assertEqual(result.selected_indices.tolist(), [0, 1])
        for actual, original in zip(
            (
                result.xyz,
                result.features,
                result.scales,
                result.rotations,
                result.opacities,
            ),
            self.candidates,
        ):
            torch.testing.assert_close(
                actual,
                torch.index_select(original, 0, result.selected_indices),
                atol=0.0,
                rtol=0.0,
            )

    def test_active_candidate_shape_contract_fails_closed(self):
        malformed = list(self.candidates)
        malformed[4] = malformed[4].reshape(-1)
        with self.assertRaisesRegex(ValueError, "shape contract violation"):
            self._select("coverage", candidates=tuple(malformed))

    def test_active_init_bypass_does_not_read_evidence_or_confidence(self):
        trap = SnapshotGetter(
            self.confidence,
            error=AssertionError("confidence must not be read"),
        )
        references = tuple(self.candidates)
        _, result = self._select(
            "evidence",
            getter=trap,
            evidence=None,
            init=True,
        )
        self.assertEqual(result.token.fields["reason"], "init_bypass")
        self.assertIsNone(result.selected_indices)
        self.assertEqual(trap.calls, 0)
        for actual, reference in zip(
            (
                result.xyz,
                result.features,
                result.scales,
                result.rotations,
                result.opacities,
            ),
            references,
        ):
            self.assertIs(actual, reference)

    def test_active_reuses_evidence_without_renderer_or_quantile_hot_path(self):
        source = inspect.getsource(GaussianCandidateActiveFixedKV2)
        for forbidden in (
            ".cpu(",
            ".tolist(",
            ".numpy(",
            "torch.quantile",
            "cuda.synchronize",
            "gaussian_renderer",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

        class Trap(PreinsertRenderEvidenceObserverV1):
            def capture(self, **kwargs):
                raise AssertionError("Active V2 must never render again.")

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
            preinsert_render_evidence_v1=Trap(device="cpu"),
            confidence_snapshot_getter=self.getter,
            device="cpu",
        )
        result = selector.select_before_extend(
            xyz=self.candidates[0],
            features=self.candidates[1],
            scales=self.candidates[2],
            rotations=self.candidates[3],
            opacities=self.candidates[4],
            camera=self.camera,
            depthmap=None,
            depth_source="estimated_clean_depth",
            mapper_update_id=23,
            init=False,
            gaussian_before=11,
            preinsert_render_evidence=self._evidence(),
        )
        self.assertEqual(result.selected_indices.tolist(), [0, 1])

    def test_active_logging_is_tensor_free_and_aggregate_only(self):
        selector = GaussianCandidateActiveFixedKV2(
            confidence_snapshot_getter=self.getter,
            logging_enabled=True,
            device="cpu",
            strategy="coverage",
            fixed_k=2,
            diversity_strength=0.1,
        )
        result = selector.select_before_extend(
            xyz=self.candidates[0],
            features=self.candidates[1],
            scales=self.candidates[2],
            rotations=self.candidates[3],
            opacities=self.candidates[4],
            camera=self.camera,
            depthmap=None,
            depth_source="estimated_clean_depth",
            mapper_update_id=23,
            init=False,
            gaussian_before=11,
            preinsert_render_evidence=self._evidence(),
        )
        self.assertFalse(_contains_tensor(result.token.fields))
        stream = io.StringIO()
        with redirect_stdout(stream):
            summary = selector.record_active_after_extend(
                result.token,
                admitted_candidate_count=2,
                dropped_candidate_count=0,
                gaussian_after_extend=13,
                m01_second_selection_applied=False,
            )
        event = json.loads(stream.getvalue().strip().split(" ", 1)[1])
        self.assertEqual(event, summary.to_event())
        self.assertEqual(event["event_type"], "candidate_active_selection_v2")
        self.assertEqual(event["strategy"], "coverage")
        self.assertEqual(event["fixed_k"], 2)
        self.assertEqual(event["candidate_count"], 4)
        self.assertEqual(event["selected_count"], 2)
        self.assertNotIn("candidate_values", event)
        self.assertNotIn("quantiles", event)

    def test_gaussian_model_active_hook_filters_once_before_extend(self):
        extend_from_pcd_seq = load_gaussian_model_extend_from_pcd_seq()
        harness = GaussianModelHookHarness(self.candidates)
        selector = self._active_selector("coverage")
        extend_from_pcd_seq(
            harness,
            self.camera,
            self.camera.uid,
            init=False,
            candidate_selector_v2=selector,
            mapper_update_id=23,
            preinsert_render_evidence=self._evidence(),
        )
        self.assertEqual(harness.extend_calls, 1)
        self.assertEqual(harness.last_extended_count, 2)
        self.assertEqual(len(harness), 13)
        self.assertEqual(self.getter.calls, 0)

    def test_gaussian_model_rejects_v1_v2_or_m01_second_selection(self):
        extend_from_pcd_seq = load_gaussian_model_extend_from_pcd_seq()
        selector = self._active_selector("coverage")
        harness = GaussianModelHookHarness(self.candidates)
        with self.assertRaisesRegex(RuntimeError, "mutually exclusive"):
            extend_from_pcd_seq(
                harness,
                self.camera,
                self.camera.uid,
                init=False,
                candidate_selector_v1=object(),
                candidate_selector_v2=selector,
                mapper_update_id=23,
                preinsert_render_evidence=self._evidence(),
            )
        self.assertEqual(harness.extend_calls, 0)

        harness = GaussianModelHookHarness(self.candidates)
        harness.resource_admission = object()
        with self.assertRaisesRegex(RuntimeError, "ResourceAdmission"):
            extend_from_pcd_seq(
                harness,
                self.camera,
                self.camera.uid,
                init=False,
                candidate_selector_v2=selector,
                mapper_update_id=23,
                preinsert_render_evidence=self._evidence(),
            )
        self.assertEqual(harness.extend_calls, 0)


if __name__ == "__main__":
    unittest.main()
