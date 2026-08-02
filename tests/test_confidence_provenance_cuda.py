from __future__ import annotations

import queue
import time
import unittest
from pathlib import Path

import torch
import torch.multiprocessing as mp

from tests.test_confidence_provenance import (
    ConfidenceSnapshotError,
    DepthVideo,
    frame_item,
    make_cfg,
    make_factor_graph_for_removal,
    write_current_confidence,
)


def _cuda_writer(video: DepthVideo, index: int, value: float, result_queue) -> None:
    try:
        torch.cuda.set_device(torch.device(video.device))
        write_current_confidence(video, index, value)
        result_queue.put(
            {
                "status": "ok",
                "version": int(video.confidence_versions[index].item()),
                "up_version": int(video.confidence_up_versions[index].item()),
            }
        )
    except Exception as exc:  # pragma: no cover - diagnostic path for the formal CUDA gate
        result_queue.put({"status": "error", "type": type(exc).__name__, "message": str(exc)})


def _locked_cuda_writer(
    video: DepthVideo,
    ba_lock,
    entered,
    release,
    result_queue,
) -> None:
    try:
        torch.cuda.set_device(torch.device(video.device))
        with ba_lock:
            entered.set()
            if not release.wait(timeout=20):
                raise TimeoutError("writer release event timed out")
            write_current_confidence(video, 2, 0.875)
        result_queue.put({"status": "ok"})
    except Exception as exc:  # pragma: no cover - diagnostic path for the formal CUDA gate
        result_queue.put({"status": "error", "type": type(exc).__name__, "message": str(exc)})


def _locked_remove(video: DepthVideo, ba_lock, entered, result_queue) -> None:
    try:
        torch.cuda.set_device(torch.device(video.device))
        if not entered.wait(timeout=20):
            raise TimeoutError("writer did not enter BA lock")
        with ba_lock:
            make_factor_graph_for_removal(video).rm_keyframe(1)
            with video.get_lock():
                video.counter.value -= 1
        result_queue.put({"status": "ok"})
    except Exception as exc:  # pragma: no cover - diagnostic path for the formal CUDA gate
        result_queue.put({"status": "error", "type": type(exc).__name__, "message": str(exc)})


def _reuse_slot_writer(video: DepthVideo, entered, result_queue) -> None:
    try:
        torch.cuda.set_device(torch.device(video.device))
        with video.get_lock():
            entered.set()
            video._DepthVideo__item_setter(0, frame_item(video, 99.0, 99))
        result_queue.put({"status": "ok"})
    except Exception as exc:  # pragma: no cover - diagnostic path for the formal CUDA gate
        result_queue.put({"status": "error", "type": type(exc).__name__, "message": str(exc)})


@unittest.skipUnless(torch.cuda.is_available(), "formal CUDA provenance gate requires CUDA")
class ConfidenceProvenanceCudaTests(unittest.TestCase):
    timeout_seconds = 30

    def setUp(self) -> None:
        if mp.get_start_method(allow_none=True) != "spawn":
            self.skipTest("run this file directly so the CUDA multiprocessing start method is spawn")
        cfg = make_cfg()
        cfg.device = "cuda:0"
        self.video = DepthVideo(cfg)

    def append_frame(self, timestamp: float, image_value: int) -> int:
        self.video.append(*frame_item(self.video, timestamp, image_value))
        return int(self.video.source_frame_ids[self.video.counter.value - 1].item())

    def join_or_fail(self, process) -> None:
        process.join(timeout=self.timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            self.fail(f"{process.name} exceeded {self.timeout_seconds}s; possible lock deadlock")
        self.assertEqual(process.exitcode, 0, f"{process.name} exited with {process.exitcode}")

    def get_worker_result(self, result_queue):
        try:
            result = result_queue.get(timeout=5)
        except queue.Empty:
            self.fail("worker produced no result")
        self.assertEqual(result.get("status"), "ok", result)
        return result

    def test_cuda_snapshot_metadata_device_and_clone_pointer(self) -> None:
        source_id = self.append_frame(1.0, 1)
        self.assertEqual(self.video.confidence.device, torch.device("cuda:0"))
        self.assertEqual(self.video.confidence.dtype, torch.float32)
        write_current_confidence(self.video, 0, 0.5)
        snapshot = self.video.get_confidence_snapshot(0, source_id)
        self.assertEqual(snapshot.source_frame_id, source_id)
        self.assertEqual(snapshot.confidence_source_frame_id, source_id)
        self.assertEqual(snapshot.shape, (16, 24))
        self.assertEqual(snapshot.dtype, "torch.float32")
        self.assertEqual(snapshot.device, "cuda:0")
        self.assertEqual(snapshot.confidence.device, self.video.confidence_up.device)
        self.assertNotEqual(snapshot.confidence.data_ptr(), self.video.confidence_up[0].data_ptr())

    def test_cuda_shared_write_value_and_versions_are_visible_cross_process(self) -> None:
        source_id = self.append_frame(1.0, 1)
        context = mp.get_context("spawn")
        result_queue = context.Queue()
        process = context.Process(
            target=_cuda_writer,
            args=(self.video, 0, 0.625, result_queue),
            name="confidence-cuda-writer",
        )
        process.start()
        self.join_or_fail(process)
        result = self.get_worker_result(result_queue)
        snapshot = self.video.get_confidence_snapshot(0, source_id)
        self.assertEqual(result["version"], snapshot.confidence_version)
        self.assertEqual(result["up_version"], snapshot.confidence_up_version)
        self.assertTrue(torch.all(snapshot.confidence == 0.625))

    def test_frontend_removal_serializes_with_backend_confidence_writer(self) -> None:
        self.append_frame(1.0, 1)
        self.append_frame(2.0, 2)
        moved_source_id = self.append_frame(3.0, 3)
        context = mp.get_context("spawn")
        ba_lock = context.RLock()
        entered = context.Event()
        release = context.Event()
        result_queue = context.Queue()
        writer = context.Process(
            target=_locked_cuda_writer,
            args=(self.video, ba_lock, entered, release, result_queue),
            name="backend-confidence-writer",
        )
        remover = context.Process(
            target=_locked_remove,
            args=(self.video, ba_lock, entered, result_queue),
            name="frontend-keyframe-remover",
        )
        writer.start()
        remover.start()
        self.assertTrue(entered.wait(timeout=10), "writer never entered BA lock")
        time.sleep(0.1)
        release.set()
        self.join_or_fail(writer)
        self.join_or_fail(remover)
        self.get_worker_result(result_queue)
        self.get_worker_result(result_queue)
        snapshot = self.video.get_confidence_snapshot(1, moved_source_id)
        self.assertTrue(snapshot.is_current)
        self.assertTrue(torch.all(snapshot.confidence == 0.875))
        self.assertEqual(int(self.video.source_frame_ids[2]), -1)

    def test_slot_reuse_camera_payload_and_identity_do_not_tear(self) -> None:
        old_source_id = self.append_frame(1.0, 1)
        context = mp.get_context("spawn")
        entered = context.Event()
        result_queue = context.Queue()
        writer = context.Process(
            target=_reuse_slot_writer,
            args=(self.video, entered, result_queue),
            name="slot-reuse-writer",
        )
        writer.start()
        self.assertTrue(entered.wait(timeout=10), "slot writer never entered video lock")
        item = self.video.get_mapping_item(0, device="cuda:0", return_frame_identity=True)
        self.join_or_fail(writer)
        self.get_worker_result(result_queue)
        identity = item[-1]
        self.assertGreater(identity["source_frame_id"], old_source_id)
        self.assertEqual(identity["source_timestamp"], 99.0)
        self.assertTrue(torch.all(item[0] == 99))

    def test_update_lowmem_order_leaves_upsampled_confidence_stale(self) -> None:
        source = Path(__file__).resolve().parents[1] / "src" / "factor_graph.py"
        update_lowmem = source.read_text(encoding="utf-8").split("def update_lowmem(", 1)[1]
        self.assertLess(update_lowmem.index("self.video.upsample("), update_lowmem.index("self.video.ba("))
        source_id = self.append_frame(1.0, 1)
        write_current_confidence(self.video, 0, 0.25)
        with self.video.confidence_lock:
            self.video._write_confidence_locked(
                torch.tensor([0], device="cuda:0"),
                torch.full((1, 2, 3), 0.75, device="cuda:0", dtype=torch.float32),
            )
        with self.assertRaisesRegex(ConfidenceSnapshotError, "stale"):
            self.video.get_confidence_snapshot(0, source_id)

    def test_normal_ba_then_upsample_returns_current_snapshot(self) -> None:
        import src.depth_video as depth_video_module

        source_id = self.append_frame(1.0, 1)
        self.video.reduce_confidence = lambda weight, ii: (
            torch.full((1, 2, 2, 3), 0.5, device="cuda:0", dtype=torch.float32),
            torch.tensor([0], device="cuda:0", dtype=torch.long),
        )
        depth_video_module.droid_backends.ba = lambda *args, **kwargs: None
        self.video.ba(
            target=torch.empty(0, device="cuda:0"),
            weight=torch.empty(0, device="cuda:0"),
            eta=torch.ones(1, device="cuda:0"),
            ii=torch.tensor([0], device="cuda:0"),
            jj=torch.tensor([0], device="cuda:0"),
            t0=0,
            t1=1,
            iters=0,
            motion_only=True,
        )
        self.video.upsample(torch.tensor([0], device="cuda:0"), torch.empty(0, device="cuda:0"))
        snapshot = self.video.get_confidence_snapshot(0, source_id)
        self.assertTrue(snapshot.is_current)
        self.assertEqual(snapshot.confidence_version, snapshot.confidence_up_version)

    def test_snapshot_clone_survives_later_source_slot_change(self) -> None:
        source_id = self.append_frame(1.0, 1)
        write_current_confidence(self.video, 0, 0.25)
        snapshot = self.video.get_confidence_snapshot(0, source_id)
        write_current_confidence(self.video, 0, 0.75)
        self.assertTrue(torch.all(snapshot.confidence == 0.25))
        self.assertTrue(torch.all(self.video.confidence_up[0] == 0.75))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    unittest.main(verbosity=2)
