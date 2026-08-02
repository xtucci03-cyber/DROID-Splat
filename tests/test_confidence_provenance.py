from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def _install_import_stubs() -> None:
    ipdb = types.ModuleType("ipdb")
    ipdb.set_trace = lambda: None
    sys.modules.setdefault("ipdb", ipdb)

    termcolor = types.ModuleType("termcolor")
    termcolor.colored = lambda value, *args, **kwargs: value
    sys.modules.setdefault("termcolor", termcolor)

    sys.modules.setdefault("lietorch", types.ModuleType("lietorch"))
    sys.modules.setdefault("droid_backends", types.ModuleType("droid_backends"))

    droid_net = types.ModuleType("src.droid_net")

    def cvx_upsample(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        del mask
        value_nchw = value.permute(0, 3, 1, 2)
        return F.interpolate(value_nchw, scale_factor=8, mode="nearest").permute(0, 2, 3, 1)

    droid_net.cvx_upsample = cvx_upsample
    sys.modules.setdefault("src.droid_net", droid_net)

    geom = types.ModuleType("src.geom")
    geom.projective_ops = types.ModuleType("src.geom.projective_ops")
    geom.matrix_to_lie = lambda value: value
    geom.align_scale_and_shift = lambda *args, **kwargs: (None, None, None)
    geom.check_and_correct_transform = lambda value, reference: value
    sys.modules.setdefault("src.geom", geom)
    sys.modules.setdefault("src.geom.projective_ops", geom.projective_ops)

    geom_ba = types.ModuleType("src.geom.ba")
    geom_ba.bundle_adjustment = lambda *args, **kwargs: None
    sys.modules.setdefault("src.geom.ba", geom_ba)

    camera_utils = types.ModuleType("src.gaussian_splatting.camera_utils")
    camera_utils.Camera = object
    sys.modules.setdefault("src.gaussian_splatting.camera_utils", camera_utils)

    renderer = types.ModuleType("src.gaussian_splatting.gaussian_renderer")
    renderer.render = lambda *args, **kwargs: None
    sys.modules.setdefault("src.gaussian_splatting.gaussian_renderer", renderer)

    modules = types.ModuleType("src.modules")
    modules.CorrBlock = object
    modules.AltCorrBlock = object
    sys.modules.setdefault("src.modules", modules)


_install_import_stubs()

from src.depth_video import ConfidenceSnapshotError, DepthVideo
from src.factor_graph import FactorGraph


def make_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            cam=SimpleNamespace(
                camera_model="pinhole",
                H_out=16,
                W_out=24,
            )
        ),
        opt_intr=False,
        mode="rgbd",
        device="cpu",
        tracking=SimpleNamespace(
            buffer=5,
            upsample=True,
            frontend=SimpleNamespace(optimize_scales=False),
        ),
    )


def frame_item(video: DepthVideo, timestamp: float, image_value: int):
    return (
        torch.tensor(timestamp, dtype=torch.float32),
        torch.full((3, video.ht, video.wd), image_value, dtype=torch.uint8),
        torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float32),
        torch.ones((video.ht // 8, video.wd // 8), dtype=torch.float32),
        torch.ones((video.ht, video.wd), dtype=torch.float32),
        torch.tensor([8.0, 8.0, 4.0, 4.0], dtype=torch.float32),
    )


def write_current_confidence(video: DepthVideo, index: int, value: float) -> None:
    indices = torch.tensor([index], dtype=torch.long)
    low_resolution = torch.full(
        (1, video.ht // 8, video.wd // 8),
        value,
        dtype=torch.float32,
    )
    with video.confidence_lock:
        video._write_confidence_locked(indices, low_resolution)
    video.upsample(indices, torch.empty(0))


def make_factor_graph_for_removal(video: DepthVideo) -> FactorGraph:
    graph = FactorGraph.__new__(FactorGraph)
    graph.video = video
    graph.ii_inac = torch.empty(0, dtype=torch.long)
    graph.jj_inac = torch.empty(0, dtype=torch.long)
    graph.target_inac = torch.empty((1, 0, 2, 3, 2), dtype=torch.float32)
    graph.weight_inac = torch.empty((1, 0, 2, 3, 2), dtype=torch.float32)
    graph.ii = torch.empty(0, dtype=torch.long)
    graph.jj = torch.empty(0, dtype=torch.long)
    graph.age = torch.empty(0, dtype=torch.long)
    graph.target = torch.empty((1, 0, 2, 3, 2), dtype=torch.float32)
    graph.weight = torch.empty((1, 0, 2, 3, 2), dtype=torch.float32)
    graph.corr_impl = "alt"
    graph.corr = None
    graph.net = None
    graph.inp = None
    return graph


class ConfidenceProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.video = DepthVideo(make_cfg())

    def append_frame(self, timestamp: float, image_value: int) -> int:
        self.video.append(*frame_item(self.video, timestamp, image_value))
        return int(self.video.source_frame_ids[self.video.counter.value - 1].item())

    def test_rm_keyframe_moves_frame_and_confidence_provenance(self) -> None:
        first_id = self.append_frame(10.0, 10)
        removed_id = self.append_frame(20.0, 20)
        moved_id = self.append_frame(30.0, 30)
        self.assertNotEqual(first_id, removed_id)
        write_current_confidence(self.video, 1, 0.25)
        write_current_confidence(self.video, 2, 0.75)

        make_factor_graph_for_removal(self.video).rm_keyframe(1)

        self.assertEqual(float(self.video.timestamp[1]), 30.0)
        self.assertTrue(torch.equal(self.video.images[1], torch.full_like(self.video.images[1], 30)))
        self.assertEqual(int(self.video.source_frame_ids[1]), moved_id)
        self.assertEqual(int(self.video.confidence_source_frame_ids[1]), moved_id)
        self.assertEqual(int(self.video.confidence_up_source_frame_ids[1]), moved_id)
        self.assertTrue(torch.all(self.video.confidence[1] == 0.75))
        self.assertTrue(torch.all(self.video.confidence_up[1] == 0.75))
        self.assertEqual(int(self.video.source_frame_ids[2]), -1)
        self.assertFalse(bool(self.video.confidence_valid[2]))
        self.assertFalse(bool(self.video.confidence_up_valid[2]))

    def test_penultimate_removal_then_reuse_invalidates_old_confidence(self) -> None:
        self.append_frame(10.0, 10)
        self.append_frame(20.0, 20)
        moved_id = self.append_frame(30.0, 30)
        write_current_confidence(self.video, 2, 0.75)
        make_factor_graph_for_removal(self.video).rm_keyframe(1)
        self.video.counter.value -= 1

        new_id = self.append_frame(40.0, 40)

        self.assertNotEqual(new_id, moved_id)
        self.assertEqual(int(self.video.source_frame_ids[2]), new_id)
        self.assertFalse(bool(self.video.confidence_valid[2]))
        self.assertFalse(bool(self.video.confidence_up_valid[2]))
        self.assertEqual(int(self.video.confidence_versions[2]), 0)
        self.assertTrue(torch.count_nonzero(self.video.confidence[2]) == 0)
        self.assertTrue(torch.count_nonzero(self.video.confidence_up[2]) == 0)
        with self.assertRaisesRegex(ConfidenceSnapshotError, "not valid"):
            self.video.get_confidence_snapshot(2, new_id)

    def test_current_frame_write_and_snapshot_are_identity_checked_clone(self) -> None:
        source_id = self.append_frame(12.5, 12)
        write_current_confidence(self.video, 0, 0.5)

        mapping_item = self.video.get_mapping_item(
            0,
            device="cpu",
            return_frame_identity=True,
        )
        self.assertEqual(mapping_item[-1]["buffer_index"], 0)
        self.assertEqual(mapping_item[-1]["source_frame_id"], source_id)
        self.assertEqual(mapping_item[-1]["source_timestamp"], 12.5)

        snapshot = self.video.get_confidence_snapshot(
            0,
            source_id,
            expected_timestamp=12.5,
        )

        self.assertTrue(snapshot.is_current)
        self.assertFalse(snapshot.is_stale)
        self.assertEqual(snapshot.buffer_index, 0)
        self.assertEqual(snapshot.source_frame_id, source_id)
        self.assertEqual(snapshot.confidence_source_frame_id, source_id)
        self.assertEqual(snapshot.shape, (16, 24))
        self.assertEqual(snapshot.dtype, "torch.float32")
        self.assertEqual(snapshot.device, "cpu")
        self.assertFalse(snapshot.requires_grad)
        self.assertNotEqual(snapshot.confidence.data_ptr(), self.video.confidence_up[0].data_ptr())
        self.assertTrue(torch.all(snapshot.confidence == 0.5))

    def test_wrong_frame_identity_and_timestamp_fail_closed(self) -> None:
        source_id = self.append_frame(12.5, 12)
        write_current_confidence(self.video, 0, 0.5)
        with self.assertRaisesRegex(ConfidenceSnapshotError, "identity mismatch"):
            self.video.get_confidence_snapshot(0, source_id + 1)
        with self.assertRaisesRegex(ConfidenceSnapshotError, "timestamp mismatch"):
            self.video.get_confidence_snapshot(0, source_id, expected_timestamp=99.0)

    def test_stale_upsampled_version_is_rejected_or_explicitly_marked(self) -> None:
        source_id = self.append_frame(1.0, 1)
        write_current_confidence(self.video, 0, 0.25)
        with self.video.confidence_lock:
            self.video._write_confidence_locked(
                torch.tensor([0]),
                torch.full((1, 2, 3), 0.75, dtype=torch.float32),
            )

        with self.assertRaisesRegex(ConfidenceSnapshotError, "stale"):
            self.video.get_confidence_snapshot(0, source_id, require_current=True)

        snapshot = self.video.get_confidence_snapshot(0, source_id, require_current=False)
        self.assertFalse(snapshot.is_current)
        self.assertTrue(snapshot.is_stale)
        self.assertLess(snapshot.confidence_up_version, snapshot.confidence_version)

    def test_all_zero_is_valid_but_nan_inf_and_shape_mismatch_fail(self) -> None:
        source_id = self.append_frame(1.0, 1)
        write_current_confidence(self.video, 0, 0.0)
        snapshot = self.video.get_confidence_snapshot(0, source_id)
        self.assertEqual(int(torch.count_nonzero(snapshot.confidence)), 0)

        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid):
                self.video.confidence_up[0, 0, 0] = invalid
                with self.assertRaisesRegex(ConfidenceSnapshotError, "NaN or Inf"):
                    self.video.get_confidence_snapshot(0, source_id)
                self.video.confidence_up[0, 0, 0] = 0.0

        original = self.video.confidence_up
        self.video.confidence_up = torch.zeros((self.video.buffer_size, 0), dtype=torch.float32)
        try:
            with self.assertRaisesRegex(ConfidenceSnapshotError, "shape mismatch"):
                self.video.get_confidence_snapshot(0, source_id)
        finally:
            self.video.confidence_up = original

    def test_remove_invalidates_identity_and_confidence(self) -> None:
        source_id = self.append_frame(1.0, 1)
        write_current_confidence(self.video, 0, 0.5)
        self.video.remove(0)
        self.assertEqual(int(self.video.source_frame_ids[0]), -1)
        self.assertFalse(bool(self.video.confidence_valid[0]))
        self.assertFalse(bool(self.video.confidence_up_valid[0]))
        with self.assertRaises(ConfidenceSnapshotError):
            self.video.get_confidence_snapshot(0, source_id)

    def test_confidence_writer_rejects_inactive_reused_slot(self) -> None:
        with self.video.confidence_lock:
            with self.assertRaisesRegex(ConfidenceSnapshotError, "inactive buffer"):
                self.video._write_confidence_locked(
                    torch.tensor([0]),
                    torch.ones((1, 2, 3), dtype=torch.float32),
                )

    def test_confidence_reader_is_explicit_and_candidate_path_is_unchanged(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        mapping_source = (repository / "src" / "gaussian_mapping.py").read_text(encoding="utf-8")
        self.assertEqual(mapping_source.count("get_camera_confidence_snapshot("), 1)
        self.assertNotIn("get_confidence_snapshot(", (repository / "src" / "candidate_selection" / "gaussian_candidate_selector_v0.py").read_text(encoding="utf-8"))
        self.assertNotIn("confidence", (repository / "src" / "gaussian_splatting" / "scene" / "gaussian_model.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
