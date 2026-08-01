from __future__ import annotations

import io
import json
import random
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import torch

import src.candidate_selection.gaussian_candidate_selector_v0 as selector_module
from src.candidate_selection import (
    GaussianCandidateSelectorDryRun,
    build_gaussian_candidate_selector,
)
from src.gaussian_candidate_observer import (
    CandidateGenerationMetadata,
    GaussianCandidateObserver,
)
from src.resource_management import ResourceAdmission


class PublicObserver:
    def __init__(self, voxel_size: float = 1.0) -> None:
        self.voxel_size = voxel_size


class ExplodingObserver:
    @property
    def voxel_size(self):
        raise AssertionError("off mode accessed Candidate Observer state")


def selector_config(
    *,
    mode: str = "dry_run",
    algorithm: str = "occupied_voxel_cap_k_v0",
    logging_enabled: bool = True,
) -> dict:
    return {
        "mode": mode,
        "algorithm": algorithm,
        "logging": {"enabled": logging_enabled},
    }


def build_selector(
    *,
    voxel_size: float = 1.0,
    admission_mode: str = "observe",
) -> GaussianCandidateSelectorDryRun:
    selector = build_gaussian_candidate_selector(
        selector_config(),
        candidate_observer=PublicObserver(voxel_size),
        resource_admission_mode=admission_mode,
        device="cpu",
    )
    assert selector is not None
    return selector


def candidates(xyz: torch.Tensor) -> tuple[torch.Tensor, ...]:
    count = int(xyz.shape[0])
    return (
        xyz,
        torch.arange(count * 3, dtype=torch.float32).reshape(count, 3, 1),
        torch.ones((count, 3), dtype=torch.float32),
        torch.nn.functional.pad(
            torch.ones((count, 1), dtype=torch.float32),
            (0, 3),
        ),
        torch.zeros((count, 1), dtype=torch.float32),
    )


def observe(
    xyz: torch.Tensor,
    current: torch.Tensor,
    *,
    selector: GaussianCandidateSelectorDryRun | None = None,
    init: bool = False,
    mapper_update_id: int = 3,
    source_camera_id: int = 7,
) -> tuple[dict, object, tuple[torch.Tensor, ...]]:
    selector = selector or build_selector()
    bundle = candidates(xyz)
    token = selector.observe_before_extend(
        xyz=bundle[0],
        features=bundle[1],
        scales=bundle[2],
        rotations=bundle[3],
        opacities=bundle[4],
        current_gaussian_xyz=current,
        mapper_update_id=mapper_update_id,
        source_camera_id=source_camera_id,
        init=init,
    )
    output = io.StringIO()
    with redirect_stdout(output):
        summary = selector.record_after_extend(
            token,
            admitted_candidate_count=int(xyz.shape[0]),
            dropped_candidate_count=0,
            gaussian_after_extend=(
                int(current.shape[0]) + int(xyz.shape[0])
            ),
        )
    line = output.getvalue().strip()
    event = json.loads(line[line.index("{") :])
    return event, summary, bundle


class CandidateSelectorConfigTests(unittest.TestCase):
    def test_missing_and_off_do_not_construct(self) -> None:
        self.assertIsNone(
            build_gaussian_candidate_selector(
                None,
                candidate_observer=None,
                resource_admission_mode="fixed_budget",
                device="cpu",
            )
        )
        with mock.patch.object(
            selector_module.torch,
            "device",
        ) as device_constructor:
            self.assertIsNone(
                build_gaussian_candidate_selector(
                    selector_config(mode="off"),
                    candidate_observer=ExplodingObserver(),
                    resource_admission_mode="fixed_budget",
                    device="cuda:0",
                )
            )
        device_constructor.assert_not_called()

    def test_public_observer_voxel_size_is_inherited(self) -> None:
        selector = build_selector(voxel_size=0.05)
        self.assertEqual(selector.voxel_size, 0.05)

    def test_dry_run_requires_candidate_observer(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate_observer"):
            build_gaussian_candidate_selector(
                selector_config(),
                candidate_observer=None,
                resource_admission_mode="disabled",
                device="cpu",
            )

    def test_control_m01_modes_fail_closed(self) -> None:
        for mode in ("fixed_budget", "control", "unknown"):
            with self.subTest(mode=mode), self.assertRaisesRegex(
                ValueError, "requires M01"
            ):
                build_gaussian_candidate_selector(
                    selector_config(),
                    candidate_observer=PublicObserver(),
                    resource_admission_mode=mode,
                    device="cpu",
                )

    def test_unknown_fields_mode_algorithm_and_disabled_log_fail(self) -> None:
        cases = []
        unknown = selector_config()
        unknown["typo"] = 1
        cases.append(unknown)
        cases.append(selector_config(mode="select"))
        cases.append(selector_config(algorithm="random"))
        cases.append(selector_config(logging_enabled=False))
        for config in cases:
            with self.subTest(config=config), self.assertRaises(
                (TypeError, ValueError)
            ):
                build_gaussian_candidate_selector(
                    config,
                    candidate_observer=PublicObserver(),
                    resource_admission_mode="observe",
                    device="cpu",
                )

    def test_bool_voxel_size_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            build_gaussian_candidate_selector(
                selector_config(),
                candidate_observer=PublicObserver(True),
                resource_admission_mode="observe",
                device="cpu",
            )


class CandidateSelectorEvidenceTests(unittest.TestCase):
    def test_empty_candidate(self) -> None:
        event, _, _ = observe(
            torch.empty((0, 3)),
            torch.empty((0, 3)),
        )
        self.assertEqual(event["status"], "ok")
        self.assertTrue(event["empty_candidate"])
        self.assertEqual(event["candidate_count"], 0)
        self.assertEqual(event["occupied_multiplicity_histogram"], [])
        for item in event["k_scan"]:
            self.assertEqual(item["estimated_admitted_count"], 0)
            self.assertIsNone(item["admitted_ratio"])

    def test_init_protects_every_candidate_for_every_k(self) -> None:
        event, _, _ = observe(
            torch.tensor([[0.1, 0.0, 0.0]] * 5),
            torch.tensor([[0.2, 0.0, 0.0]]),
            init=True,
        )
        self.assertTrue(event["protected_init"])
        self.assertEqual(event["protected_init"], event["init"])
        for item in event["k_scan"]:
            self.assertEqual(item["estimated_admitted_count"], 5)
            self.assertEqual(item["estimated_dropped_count"], 0)

    def test_legacy_and_standard_empty_maps_are_all_novel(self) -> None:
        xyz = torch.tensor([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]])
        for current in (torch.empty(0), torch.empty((0, 3))):
            with self.subTest(shape=list(current.shape)):
                event, _, _ = observe(xyz, current)
                self.assertEqual(event["status"], "ok")
                self.assertEqual(event["occupied_candidate_count"], 0)
                self.assertEqual(event["novel_candidate_count"], 2)
                self.assertEqual(event["novel_unique_voxel_count"], 2)

    def test_all_occupied_and_multiplicity_histogram(self) -> None:
        xyz = torch.tensor(
            [
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [1.1, 0.0, 0.0],
                [1.2, 0.0, 0.0],
            ]
        )
        current = torch.tensor(
            [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]]
        )
        event, _, _ = observe(xyz, current)
        self.assertEqual(event["occupied_candidate_count"], 5)
        self.assertEqual(event["occupied_unique_voxel_count"], 2)
        self.assertEqual(
            event["occupied_multiplicity_histogram"],
            [
                {"multiplicity": 2, "voxel_count": 1},
                {"multiplicity": 3, "voxel_count": 1},
            ],
        )
        self.assertEqual(event["occupied_multiplicity_mean"], 2.5)
        self.assertEqual(event["occupied_multiplicity_max"], 3)
        self.assertEqual(event["k_scan"][0]["estimated_admitted_count"], 2)
        self.assertEqual(event["k_scan"][1]["estimated_admitted_count"], 4)
        self.assertEqual(event["k_scan"][2]["estimated_admitted_count"], 5)
        self.assertEqual(event["k_scan"][3]["estimated_admitted_count"], 5)

    def test_mixed_novel_occupied_and_negative_coordinates(self) -> None:
        event, _, _ = observe(
            torch.tensor(
                [
                    [-0.1, 0.0, 0.0],
                    [-0.2, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ]
            ),
            torch.tensor([[-0.5, 0.0, 0.0]]),
        )
        self.assertEqual(event["occupied_candidate_count"], 2)
        self.assertEqual(event["novel_candidate_count"], 2)
        self.assertEqual(event["candidate_unique_voxel_count"], 3)
        self.assertEqual(event["occupied_unique_voxel_count"], 1)
        self.assertEqual(event["novel_unique_voxel_count"], 2)

    def test_nan_and_inf_candidates_are_forwarded_but_error(self) -> None:
        for nonfinite in (float("nan"), float("inf")):
            with self.subTest(nonfinite=nonfinite):
                event, _, bundle = observe(
                    torch.tensor(
                        [[nonfinite, 0.0, 0.0], [1.0, 0.0, 0.0]]
                    ),
                    torch.empty((0, 3)),
                )
                self.assertEqual(event["status"], "error")
                self.assertEqual(
                    event["reason"],
                    "nonfinite_candidate_coordinates",
                )
                self.assertEqual(event["candidate_nonfinite_count"], 1)
                self.assertEqual(event["candidate_unique_voxel_count"], 1)
                self.assertEqual(event["actual_admitted_count"], 2)
                self.assertEqual(event["actual_dropped_count"], 0)
                self.assertTrue(event["all_candidates_forwarded"])
                self.assertTrue(event["actual_conservation_pass"])
                self.assertFalse(torch.isfinite(bundle[0][0, 0]))

    def test_event_sequence_makes_repeated_camera_keys_unique(self) -> None:
        selector = build_selector()
        first, _, _ = observe(
            torch.tensor([[0.1, 0.0, 0.0]]),
            torch.empty((0, 3)),
            selector=selector,
            mapper_update_id=0,
            source_camera_id=0,
        )
        second, _, _ = observe(
            torch.tensor([[0.1, 0.0, 0.0]]),
            torch.empty((0, 3)),
            selector=selector,
            mapper_update_id=0,
            source_camera_id=0,
        )
        self.assertEqual(first["event_sequence"], 0)
        self.assertEqual(second["event_sequence"], 1)
        self.assertNotEqual(first["event_id"], second["event_id"])

    def test_observe_empty_hook_emits_one_event(self) -> None:
        selector = build_selector()
        token = selector.observe_empty(
            current_gaussian_xyz=torch.empty(0),
            mapper_update_id=0,
            source_camera_id=1,
            init=True,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            summary = selector.record_after_extend(
                token,
                admitted_candidate_count=0,
                dropped_candidate_count=0,
                gaussian_after_extend=0,
            )
        event = json.loads(output.getvalue().split(" ", 1)[1])
        self.assertEqual(event["reason"], "point_cloud_not_emitted_leq5")
        self.assertTrue(event["protected_init"])
        self.assertTrue(event["actual_conservation_pass"])
        self.assertEqual(summary.to_event(), event)


class CandidateSelectorTransparencyTests(unittest.TestCase):
    def test_tensor_identity_content_grad_and_rng_are_unchanged(self) -> None:
        selector = build_selector()
        bundle = candidates(
            torch.tensor([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]])
        )
        bundle[0].requires_grad_(True)
        current = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
        identities = [id(tensor) for tensor in bundle]
        pointers = [tensor.data_ptr() for tensor in bundle]
        clones = [tensor.detach().clone() for tensor in bundle]
        gradients = [tensor.requires_grad for tensor in bundle]
        dtypes = [tensor.dtype for tensor in bundle]
        devices = [tensor.device for tensor in bundle]

        random.seed(321)
        np.random.seed(321)
        torch.manual_seed(321)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()

        token = selector.observe_before_extend(
            xyz=bundle[0],
            features=bundle[1],
            scales=bundle[2],
            rotations=bundle[3],
            opacities=bundle[4],
            current_gaussian_xyz=current,
            mapper_update_id=1,
            source_camera_id=2,
            init=False,
        )

        self.assertEqual([id(tensor) for tensor in bundle], identities)
        self.assertEqual([tensor.data_ptr() for tensor in bundle], pointers)
        self.assertEqual(
            [tensor.requires_grad for tensor in bundle], gradients
        )
        self.assertEqual([tensor.dtype for tensor in bundle], dtypes)
        self.assertEqual([tensor.device for tensor in bundle], devices)
        for actual, expected in zip(bundle, clones):
            self.assertTrue(torch.equal(actual.detach(), expected))
        self.assertEqual(random.getstate(), python_state)
        now = np.random.get_state()
        self.assertEqual(now[0], numpy_state[0])
        self.assertTrue(np.array_equal(now[1], numpy_state[1]))
        self.assertEqual(now[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertFalse(
            any(
                isinstance(value, torch.Tensor)
                for value in token.fields.values()
            )
        )

    def test_summary_never_contains_selection_output(self) -> None:
        event, summary, _ = observe(
            torch.tensor([[0.1, 0.0, 0.0]] * 3),
            torch.tensor([[0.2, 0.0, 0.0]]),
        )
        self.assertFalse(event["selection_applied"])
        self.assertIsNone(event["selected_indices"])
        self.assertEqual(summary.to_event(), event)

    def test_selector_then_m01_observe_preserves_original_tensors(self) -> None:
        selector = build_selector(admission_mode="observe")
        admission = ResourceAdmission(mode="observe")
        bundle = candidates(
            torch.tensor([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]])
        )
        identities = [id(tensor) for tensor in bundle]
        selector_token = selector.observe_before_extend(
            xyz=bundle[0],
            features=bundle[1],
            scales=bundle[2],
            rotations=bundle[3],
            opacities=bundle[4],
            current_gaussian_xyz=torch.empty((0, 3)),
            mapper_update_id=0,
            source_camera_id=0,
            init=False,
        )
        result = admission.admit_before_extend(
            xyz=bundle[0],
            features=bundle[1],
            scales=bundle[2],
            rotations=bundle[3],
            opacities=bundle[4],
            camera_uid=0,
            kf_id=0,
            init=False,
            gaussian_before=0,
        )
        self.assertEqual(
            [
                id(result.xyz),
                id(result.features),
                id(result.scales),
                id(result.rotations),
                id(result.opacities),
            ],
            identities,
        )
        self.assertIsNone(result.selected_indices)
        self.assertEqual(result.admitted_count, 2)
        self.assertEqual(result.dropped_count, 0)
        with redirect_stdout(io.StringIO()):
            admission.record_after_extend(result, gaussian_after_extend=2)
            summary = selector.record_after_extend(
                selector_token,
                admitted_candidate_count=result.admitted_count,
                dropped_candidate_count=result.dropped_count,
                gaussian_after_extend=2,
            )
        self.assertTrue(summary.to_event()["actual_conservation_pass"])


class CandidateSelectorGcoConsistencyTests(unittest.TestCase):
    def test_gco_and_selector_voxel_counts_match(self) -> None:
        voxel_size = 1.0
        gco = GaussianCandidateObserver(
            voxel_size=voxel_size,
            gpu_timing=False,
            memory_enabled=False,
            logging_enabled=True,
            device="cpu",
        )
        selector = build_selector(voxel_size=voxel_size)
        xyz = torch.tensor(
            [
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [1.1, 0.0, 0.0],
                [2.1, 0.0, 0.0],
            ]
        )
        current = torch.tensor([[0.5, 0.0, 0.0]])
        bundle = candidates(xyz)
        gco_token = gco.observe_before_extend(
            xyz=bundle[0],
            features=bundle[1],
            scales=bundle[2],
            rotations=bundle[3],
            opacities=bundle[4],
            current_gaussian_xyz=current,
            metadata=CandidateGenerationMetadata(
                depth_source="estimated_clean_depth",
                depth_pixel_count=4,
                valid_depth_count=4,
                pre_downsample_point_count=6,
                post_downsample_point_count=4,
            ),
            mapper_update_id=0,
            source_camera_id=0,
            init=False,
        )
        selector_token = selector.observe_before_extend(
            xyz=bundle[0],
            features=bundle[1],
            scales=bundle[2],
            rotations=bundle[3],
            opacities=bundle[4],
            current_gaussian_xyz=current,
            mapper_update_id=0,
            source_camera_id=0,
            init=False,
        )
        with redirect_stdout(io.StringIO()):
            gco_summary = gco.record_after_extend(
                gco_token,
                admitted_candidate_count=4,
                dropped_candidate_count=0,
                gaussian_after_extend=5,
            )
            selector_summary = selector.record_after_extend(
                selector_token,
                admitted_candidate_count=4,
                dropped_candidate_count=0,
                gaussian_after_extend=5,
            )
        gco_event = gco_summary.to_event()
        selector_event = selector_summary.to_event()
        for field in (
            "candidate_finite_count",
            "candidate_nonfinite_count",
            "candidate_unique_voxel_count",
            "occupied_candidate_count",
            "novel_candidate_count",
        ):
            gco_field = (
                "candidate_3d_count" if field == "candidate_count" else field
            )
            self.assertEqual(selector_event[field], gco_event[gco_field])


if __name__ == "__main__":
    unittest.main()
