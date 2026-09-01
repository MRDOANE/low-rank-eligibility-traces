import unittest

try:
    import torch  # noqa: F401

    from adaptive_memory.n08_experiment import (
        Config,
        _bootstrap,
        _spearman,
        evaluation_batches,
        screen_decision,
    )
    from adaptive_memory.n08_traces import METHODS
    from adaptive_memory.n08_tasks import task_suite

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class N08ExperimentTests(unittest.TestCase):
    def test_evaluation_grid_contains_all_prespecified_shifts(self):
        task = next(iter(task_suite().values()))
        batches = evaluation_batches(task, 16, 7)
        self.assertEqual(
            set(batches),
            {
                "iid",
                "length_2x",
                "length_4x",
                "noise_shift",
                "scale_shift",
                "joint_shift",
            },
        )
        self.assertEqual(batches["length_4x"].x.shape[1], 4 * task.train_length)
        self.assertEqual(batches["iid"].x.shape[1], task.train_length)

    def test_statistics_are_deterministic_and_have_correct_direction(self):
        first = _bootstrap([0.01, 0.02, 0.03, 0.04], 1000, 13)
        second = _bootstrap([0.01, 0.02, 0.03, 0.04], 1000, 13)
        self.assertEqual(first, second)
        self.assertGreater(first["lower_95"], 0.0)
        self.assertAlmostEqual(_spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)

    def test_frozen_cascade_requires_replication_and_small_state(self):
        config = Config()
        self.assertEqual(config.required_successful_seeds, 4)
        self.assertEqual(config.trace_rank, 2)
        self.assertLessEqual(config.maximum_state_fraction, 0.20)
        self.assertGreaterEqual(config.confirmation_steps, config.screen_steps)
        self.assertGreaterEqual(config.minimum_gradient_cosine, 0.90)

    def test_positive_screen_fixture_passes_all_frozen_checks(self):
        rows = []
        method_mse = {
            "full_trace": 0.10,
            "lowrank_svd": 0.15,
            "recent_factors": 0.35,
            "reservoir_factors": 0.34,
            "replay_reservoir": 0.30,
            "multi_timescale": 0.40,
        }
        method_cosine = {
            "full_trace": 1.00,
            "lowrank_svd": 0.96,
            "recent_factors": 0.78,
            "reservoir_factors": 0.80,
            "replay_reservoir": 0.90,
            "multi_timescale": 0.70,
        }
        method_state = {
            "full_trace": 1728,
            "lowrank_svd": 294,
            "recent_factors": 294,
            "reservoir_factors": 294,
            "replay_reservoir": 275,
            "multi_timescale": 288,
        }
        initial = {}
        for task in task_suite():
            initial[f"{task}:7"] = {
                "aggregate": {
                    "mean_mse": 1.0,
                    "ood_mean_mse": 1.0,
                    "ood_worst_mse": 1.0,
                }
            }
            for method in METHODS:
                evaluation = {
                    condition: {"terminal_mse": method_mse[method]}
                    for condition in (
                        "iid",
                        "length_2x",
                        "length_4x",
                        "noise_shift",
                        "scale_shift",
                        "joint_shift",
                    )
                }
                evaluation["aggregate"] = {
                    "mean_mse": method_mse[method],
                    "ood_mean_mse": method_mse[method],
                    "ood_worst_mse": method_mse[method],
                }
                rows.append(
                    {
                        "task": task,
                        "method": method,
                        "seed": 7,
                        "training": {
                            "parameters": 1728,
                            "optimizer_steps": 400,
                            "state_floats_per_episode": method_state[method],
                            "mean_gradient_cosine": method_cosine[method],
                        },
                        "evaluation": evaluation,
                    }
                )
        decision = screen_decision(rows, initial, Config())
        self.assertTrue(decision["gate_passed"])
        self.assertTrue(all(decision["checks"].values()))


if __name__ == "__main__":
    unittest.main()
