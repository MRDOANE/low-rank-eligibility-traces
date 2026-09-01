import unittest

try:
    import torch

    from adaptive_memory.o10_experiment import (
        CONFIRMATION_SEEDS,
        HORIZON_CONDITIONS,
        OOD_CONDITIONS,
        Config,
        _relative_gain,
        confirmation_decision,
        evaluation_batches,
        screen_decision,
    )
    from adaptive_memory.o10_layerwise import METHODS
    from adaptive_memory.o10_tasks import task_suite

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class O10ExperimentTests(unittest.TestCase):
    @staticmethod
    def _row(task, seed, method, mse):
        if method == "full_trace":
            state = 16384
            cosine = 1.0
        else:
            state = 4352
            cosine = 0.90
        if method == "layerwise_sketch":
            cosine = 0.95
        conditions = {
            name: {"terminal_mse": mse}
            for name in ("iid_4x", *OOD_CONDITIONS)
        }
        return {
            "task": task,
            "seed": seed,
            "method": method,
            "trace_rank": 8,
            "training": {
                "parameters": 16384,
                "optimizer_steps": 500,
                "allocated_state_floats_per_episode": state,
                "mean_gradient_cosine": cosine,
            },
            "evaluation": {
                **conditions,
                "aggregate": {
                    "horizon_mean_mse": mse,
                    "ood_mean_mse": mse,
                    "mean_mse": mse,
                },
            },
        }

    @classmethod
    def _positive(cls, seeds=(7,)):
        rows = []
        initial = {}
        for seed in seeds:
            for task in task_suite():
                initial[f"{task}:{seed}"] = {
                    "aggregate": {"horizon_mean_mse": 0.10}
                }
                for method in METHODS:
                    if method == "full_trace":
                        mse = 0.005
                    elif method == "layerwise_sketch":
                        mse = 0.010
                    elif method in {"pooled_layer_sketch", "permuted_layer_sketch"}:
                        mse = 0.030
                    else:
                        mse = 0.020
                    rows.append(cls._row(task, seed, method, mse))
        return rows, initial

    def test_evaluation_suite_contains_frozen_horizons_and_shifts(self):
        for task in task_suite().values():
            batches = evaluation_batches(task, 8, 111)
            self.assertEqual(set(batches), {"iid_4x", *OOD_CONDITIONS})
            self.assertEqual(
                batches["length_8x"].x.shape[1], 8 * task.train_length
            )

    def test_relative_gain_direction(self):
        self.assertAlmostEqual(_relative_gain(0.09, 0.10), 0.10)
        self.assertLess(_relative_gain(0.11, 0.10), 0.0)

    def test_positive_screen_fixture_passes(self):
        rows, initial = self._positive()
        decision = screen_decision(rows, initial, Config())
        self.assertTrue(decision["gate_passed"])
        self.assertEqual(decision["task_condition_wins"], 6)

    def test_positive_confirmation_fixture_passes(self):
        rows, initial = self._positive(CONFIRMATION_SEEDS)
        decision = confirmation_decision(rows, initial, Config())
        self.assertTrue(decision["gate_passed"])
        self.assertEqual(decision["seed_wins"], 5)

    def test_horizon_conditions_are_three_prespecified_cells(self):
        self.assertEqual(len(HORIZON_CONDITIONS), 3)


if __name__ == "__main__":
    unittest.main()
