"""CPU-only regression tests for CUDA-profiler event attribution.

The tests use synthetic profiler event objects.  They do not initialize CUDA
or execute the Active selector; they validate the benchmark harness that
classifies profiler evidence produced by the server-side CUDA run.
"""

from dataclasses import dataclass, field
from typing import Any
import unittest

from tests import benchmark_gaussian_candidate_active_topk_v1_cuda as benchmark


@dataclass
class _FakeTimeRange:
    start: float
    end: float


@dataclass
class _FakeEvent:
    name: str
    start: float
    end: float
    cpu_children: list[Any] = field(default_factory=list)

    @property
    def time_range(self) -> _FakeTimeRange:
        return _FakeTimeRange(self.start, self.end)


class _FakeProfiler:
    def __init__(self, events: list[_FakeEvent]) -> None:
        self._events = events

    def events(self) -> list[_FakeEvent]:
        return list(self._events)


def _profile_with_orphan(event: _FakeEvent) -> _FakeProfiler:
    production_child = _FakeEvent("aten::sort", 11.0, 19.0)
    harness_child = _FakeEvent("aten::empty", 31.0, 39.0)
    production_scope = _FakeEvent(
        benchmark.PRODUCTION_SCOPE,
        10.0,
        20.0,
        [production_child],
    )
    harness_scope = _FakeEvent(
        benchmark.HARNESS_BOUNDARY_SCOPE,
        30.0,
        40.0,
        [harness_child],
    )
    return _FakeProfiler(
        [
            production_scope,
            production_child,
            harness_scope,
            harness_child,
            event,
        ]
    )


class ProfilerAttributionTests(unittest.TestCase):
    def test_hierarchy_production_sync_is_observed(self) -> None:
        production_sync = _FakeEvent("aten::item", 12.0, 13.0)
        harness_child = _FakeEvent("aten::empty", 31.0, 39.0)
        production_scope = _FakeEvent(
            benchmark.PRODUCTION_SCOPE,
            10.0,
            20.0,
            [production_sync],
        )
        harness_scope = _FakeEvent(
            benchmark.HARNESS_BOUNDARY_SCOPE,
            30.0,
            40.0,
            [harness_child],
        )
        result = benchmark._analyze_profile_events(
            _FakeProfiler(
                [
                    production_scope,
                    production_sync,
                    harness_scope,
                    harness_child,
                ]
            )
        )

        self.assertEqual(result["SYNC_RISK"], "OBSERVED")
        self.assertEqual(
            result["PRODUCTION_SCOPE_SYNC"]["explicit_sync_counts"][
                "aten::item"
            ],
            1,
        )

    def test_interval_orphan_production_sync_is_observed(self) -> None:
        result = benchmark._analyze_profile_events(
            _profile_with_orphan(
                _FakeEvent("cudaDeviceSynchronize", 15.0, 16.0)
            )
        )

        self.assertEqual(result["SYNC_RISK"], "OBSERVED")
        self.assertEqual(
            result["PRODUCTION_SCOPE_SYNC"]["explicit_sync_counts"][
                "cudaDeviceSynchronize"
            ],
            1,
        )
        self.assertEqual(
            result["HARNESS_BOUNDARY_SYNC"]["profiled_events"][
                "explicit_sync_total"
            ],
            0,
        )

    def test_harness_and_profiler_sync_do_not_mark_production(self) -> None:
        harness_result = benchmark._analyze_profile_events(
            _profile_with_orphan(
                _FakeEvent("cudaDeviceSynchronize", 35.0, 36.0)
            )
        )
        infrastructure_result = benchmark._analyze_profile_events(
            _profile_with_orphan(
                _FakeEvent("cudaDeviceSynchronize", 2.0, 3.0)
            )
        )

        self.assertEqual(harness_result["SYNC_RISK"], "NOT_OBSERVED")
        self.assertEqual(
            harness_result["HARNESS_BOUNDARY_SYNC"]["profiled_events"][
                "explicit_sync_counts"
            ]["cudaDeviceSynchronize"],
            1,
        )
        self.assertEqual(
            harness_result["PRODUCTION_SCOPE_SYNC"]["explicit_sync_total"], 0
        )
        self.assertEqual(infrastructure_result["SYNC_RISK"], "NOT_OBSERVED")
        self.assertEqual(
            infrastructure_result["PROFILER_INFRASTRUCTURE_SYNC"][
                "explicit_sync_counts"
            ]["cudaDeviceSynchronize"],
            1,
        )

    def test_unclassified_sync_remains_fail_closed_with_diagnostics(self) -> None:
        with self.assertRaises(benchmark.SyncProfileBlocked) as caught:
            benchmark._analyze_profile_events(
                _profile_with_orphan(
                    _FakeEvent("cudaDeviceSynchronize", 25.0, 26.0)
                )
            )

        error = caught.exception
        self.assertEqual(error.blocked_stage, "profiler_event_extraction")
        self.assertIsInstance(error.original_error, RuntimeError)
        message = str(error.original_error)
        self.assertIn("cudaDeviceSynchronize", message)
        self.assertIn("(25.0, 26.0)", message)
        self.assertIn("not_fully_contained_in_one_scope", message)
        record = error.to_record()
        self.assertEqual(record["status"], "BLOCKED")
        self.assertEqual(record["SYNC_PROFILE"], "BLOCKED")
        self.assertEqual(record["CUDA_GATE"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
