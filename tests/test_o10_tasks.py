import unittest

import numpy as np

from adaptive_memory.o10_tasks import (
    full_reference_traces,
    full_state_floats,
    premise_summary,
    reference_factors,
    replay_slots,
    sample_batch,
    sketch_state_floats,
    task_audit,
    task_suite,
    trace_cosine,
    two_sided_sketch_numpy,
)


class O10TaskTests(unittest.TestCase):
    def test_sampling_is_reproducible_and_only_terminal_target_is_returned(self):
        for task in task_suite().values():
            first = sample_batch(task, 8, 11)
            second = sample_batch(task, 8, 11)
            np.testing.assert_array_equal(first.x, second.x)
            np.testing.assert_array_equal(first.target, second.target)
            self.assertEqual(first.target.shape, (8,))
            self.assertEqual(first.x.shape[1], task.train_length)

    def test_manual_traces_match_finite_differences(self):
        for task in task_suite().values():
            audit = task_audit(task, n=8)
            self.assertLessEqual(audit["maximum_finite_difference_error"], 2e-5)

    def test_sketch_and_replay_state_accounting_is_frozen(self):
        for task in task_suite().values():
            self.assertEqual(sketch_state_floats(task, 8), 4352)
            self.assertEqual(full_state_floats(task), 16384)
            self.assertEqual(replay_slots(task, 8), 66)
            self.assertLessEqual(66 * (task.dim + 1), 4352)
            self.assertLessEqual(4352 / 16384, 0.30)

    def test_layerwise_sketch_has_valid_gradient_direction(self):
        for task in task_suite().values():
            batch = sample_batch(task, 12, 17, target_noise=0.0)
            _, left, right = reference_factors(task, batch.x)
            exact = full_reference_traces(task, batch.x)
            estimate = two_sided_sketch_numpy(task, left, right, 8)
            cosine = trace_cosine(estimate, exact)
            self.assertTrue(np.isfinite(cosine).all())
            self.assertGreater(float(np.mean(cosine)), 0.80)

    def test_frozen_premise_passes(self):
        summary = premise_summary()
        self.assertTrue(summary["premise_passed"])
        self.assertTrue(all(summary["checks"].values()))


if __name__ == "__main__":
    unittest.main()
