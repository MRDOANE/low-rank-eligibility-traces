import unittest

import numpy as np

from adaptive_memory.n08_tasks import (
    boundary_suite,
    full_reference_traces,
    premise_summary,
    reference_factors,
    sample_batch,
    task_audit,
    task_suite,
    trace_state_floats,
)


class N08TaskTests(unittest.TestCase):
    def test_sampling_is_reproducible_and_target_is_terminal_aggregate(self):
        for task in task_suite().values():
            first = sample_batch(task, 32, 19, target_noise=0.0)
            second = sample_batch(task, 32, 19, target_noise=0.0)
            self.assertTrue(np.array_equal(first.x, second.x))
            self.assertTrue(np.array_equal(first.target, second.target))
            self.assertTrue(np.allclose(first.target, first.event_scores.mean(axis=1)))

    def test_reference_trace_is_average_of_per_event_rank_one_factors(self):
        for task in task_suite().values():
            batch = sample_batch(task, 12, 31, target_noise=0.0)
            _, left, right = reference_factors(task, batch.x)
            traces = full_reference_traces(task, batch.x)
            for trace, left_layer, right_layer in zip(traces, left, right):
                direct = np.einsum("bti,btj->bij", left_layer, right_layer)
                direct = direct / batch.x.shape[1]
                self.assertTrue(np.allclose(trace, direct, atol=1e-10))

    def test_rank_two_state_is_matched_and_much_smaller_than_full_trace(self):
        task = next(iter(task_suite().values()))
        candidate = trace_state_floats(task, 2, "lowrank_svd")
        full = trace_state_floats(task, 2, "full_trace")
        self.assertEqual(candidate, trace_state_floats(task, 2, "recent_factors"))
        self.assertEqual(candidate, trace_state_floats(task, 2, "reservoir_factors"))
        self.assertLessEqual(trace_state_floats(task, 2, "multi_timescale"), candidate)
        self.assertLessEqual(trace_state_floats(task, 2, "replay_reservoir"), candidate)
        self.assertLess(candidate / full, 0.20)

    def test_designed_rank_boundary_reduces_top_two_spectral_coverage(self):
        low = task_audit(task_suite()["latent_rank_02"], n=96)
        high = task_audit(boundary_suite()["latent_rank_12"], n=96)
        self.assertGreater(low["mean_top_rank_two_energy"], 0.95)
        self.assertLess(high["mean_top_rank_two_energy"], 0.65)
        self.assertGreater(
            low["mean_top_rank_two_energy"] - high["mean_top_rank_two_energy"],
            0.10,
        )

    def test_reference_premise_passes(self):
        summary = premise_summary()
        self.assertTrue(summary["gate_passed"])
        self.assertTrue(all(summary["checks"].values()))
        self.assertEqual(len(summary["primary_task_audits"]), 2)
        self.assertEqual(len(summary["boundary_task_audits"]), 2)


if __name__ == "__main__":
    unittest.main()
