import unittest

import numpy as np

from adaptive_memory.n08_pareto_experiment import (
    Config,
    PRIMARY_SEEDS,
    _bootstrap_mean,
    make_v2_task,
    premise_summary_v2,
    rank_analysis,
)
from adaptive_memory.n08_tasks import trace_state_floats

try:
    import torch

    from adaptive_memory.n08_pareto_experiment import evaluation_batches, run_one

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


def _row(task, horizon, seed, method, rank, mse, cosine=0.95):
    state_rank = 2 if method == "full_trace" else rank
    task_object = make_v2_task(2, horizon, task_seed=91_001, split="fixture")
    return {
        "task": task,
        "latent_rank": 2,
        "horizon_multiplier": horizon,
        "sequence_length": 24 * horizon,
        "method": method,
        "trace_rank": 0 if method == "full_trace" else rank,
        "method_key": method if method == "full_trace" else f"{method}:r{rank}",
        "seed": seed,
        "training": {
            "parameters": 1728,
            "optimizer_steps": 550,
            "state_floats_per_episode": trace_state_floats(
                task_object, state_rank, method
            ),
            "mean_gradient_cosine": cosine,
        },
        "evaluation": {"aggregate": {"ood_mean_mse": mse}},
    }


def _fixture(candidate_mse, replay_mse, cosine):
    rows = []
    initial = {}
    for horizon in (1, 4, 8):
        task = f"fixture_h{horizon}"
        for seed in PRIMARY_SEEDS:
            initial[f"{task}:{seed}"] = {"aggregate": {"ood_mean_mse": 1.0}}
            rows.extend(
                (
                    _row(task, horizon, seed, "full_trace", 0, 0.20, 1.0),
                    _row(task, horizon, seed, "lowrank_svd", 2, candidate_mse, cosine),
                    _row(task, horizon, seed, "replay_reservoir", 2, replay_mse, 0.90),
                    _row(task, horizon, seed, "multi_timescale", 2, 0.25, 0.75),
                )
            )
    return rows, initial


class N08ParetoExperimentTests(unittest.TestCase):
    def test_horizons_share_teacher_but_have_unique_names(self):
        short = make_v2_task(4, 1, task_seed=208_083, split="primary")
        long = make_v2_task(4, 8, task_seed=208_083, split="primary")
        self.assertNotEqual(short.name, long.name)
        self.assertEqual(short.train_length, 24)
        self.assertEqual(long.train_length, 192)
        for left, right in zip(short.teacher_updates, long.teacher_updates):
            self.assertTrue(np.array_equal(left, right))
        for left, right in zip(short.base_matrices, long.base_matrices):
            self.assertTrue(np.array_equal(left, right))

    def test_rank_state_ladder_is_exact(self):
        task = make_v2_task(2, 1, task_seed=208_083, split="primary")
        full = trace_state_floats(task, 2, "full_trace")
        fractions = [
            trace_state_floats(task, rank, "lowrank_svd") / full
            for rank in (1, 2, 4)
        ]
        self.assertEqual(full, 1728)
        self.assertEqual(fractions, sorted(fractions))
        self.assertLessEqual(fractions[0], 0.10)
        self.assertLessEqual(fractions[1], 0.18)
        self.assertLessEqual(fractions[2], 0.35)

    def test_bootstrap_is_deterministic(self):
        first = _bootstrap_mean([0.1, 0.2, 0.3, 0.4, 0.5], 1000, 17)
        second = _bootstrap_mean([0.1, 0.2, 0.3, 0.4, 0.5], 1000, 17)
        self.assertEqual(first, second)

    def test_positive_fixture_passes(self):
        rows, initial = _fixture(candidate_mse=0.205, replay_mse=0.220, cosine=0.95)
        result = rank_analysis(
            rows, initial, rank=2, seeds=PRIMARY_SEEDS, config=Config()
        )
        self.assertTrue(result["gate_passed"])
        self.assertTrue(result["strong_pareto_signal"])
        self.assertFalse(result["decisive_failure"])

    def test_ambiguous_rank_two_fixture_triggers_rank_four_rescue(self):
        rows, initial = _fixture(candidate_mse=0.30, replay_mse=0.22, cosine=0.85)
        result = rank_analysis(
            rows, initial, rank=2, seeds=PRIMARY_SEEDS, config=Config()
        )
        self.assertFalse(result["gate_passed"])
        self.assertTrue(result["rank_four_rescue_warranted"])
        self.assertFalse(result["decisive_failure"])

    def test_clear_failure_fixture_is_decisive(self):
        rows, initial = _fixture(candidate_mse=0.50, replay_mse=0.22, cosine=0.50)
        result = rank_analysis(
            rows, initial, rank=2, seeds=PRIMARY_SEEDS, config=Config()
        )
        self.assertFalse(result["gate_passed"])
        self.assertFalse(result["rank_four_rescue_warranted"])
        self.assertTrue(result["decisive_failure"])

    def test_reference_premise_passes(self):
        summary = premise_summary_v2()
        self.assertTrue(summary["gate_passed"], summary["checks"])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class N08ParetoTorchTests(unittest.TestCase):
    def test_every_primary_method_runs_one_step(self):
        task = make_v2_task(2, 1, task_seed=108_083, split="smoke")
        config = Config(batch_size=4)
        batches = evaluation_batches(task, 8, 123, config.target_noise)
        for method in (
            "full_trace",
            "lowrank_svd",
            "replay_reservoir",
            "multi_timescale",
        ):
            row = run_one(
                task,
                batches,
                method=method,
                rank=2,
                seed=19,
                steps=1,
                learning_rate=0.01,
                config=config,
                device=torch.device("cpu"),
            )
            self.assertTrue(np.isfinite(row["evaluation"]["aggregate"]["ood_mean_mse"]))
            self.assertEqual(row["training"]["optimizer_steps"], 1)


if __name__ == "__main__":
    unittest.main()
