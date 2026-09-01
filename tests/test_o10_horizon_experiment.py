import unittest

import numpy as np

from adaptive_memory.o10_horizon_experiment import (
    CONFIRMATION_REPLICATES,
    LATENT_RANKS,
    PRIMARY_HORIZONS,
    Config,
    analyze,
    make_v2_task,
    premise_summary_v2,
)
from adaptive_memory.o10_layerwise import (
    global_rank_for_budget,
    method_state,
)
from adaptive_memory.o10_tasks import sketch_state_floats

try:
    import torch

    from adaptive_memory.o10_horizon_experiment import evaluation_batches, run_one

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


def _row(replicate, latent_rank, horizon, method, mse, cosine=0.92):
    state = 16384 if method == "full_trace" else 4352
    return {
        "replicate": replicate,
        "latent_rank": latent_rank,
        "horizon_multiplier": horizon,
        "method": method,
        "training": {
            "allocated_state_floats_per_episode": state,
            "parameters": 16384,
            "optimizer_steps": 550,
            "mean_gradient_cosine": 1.0 if method == "full_trace" else cosine,
        },
        "evaluation": {"aggregate": {"mean_mse": mse}},
    }


def _fixture(candidate_by_horizon, replay_by_horizon, cosine=0.92):
    rows = []
    initial = {}
    global_by_horizon = {1: 0.255, 4: 0.235, 8: 0.210}
    for replicate in CONFIRMATION_REPLICATES:
        for latent_rank in LATENT_RANKS:
            for horizon in PRIMARY_HORIZONS:
                key = f"rep{replicate}:latent{latent_rank}:h{horizon}"
                initial[key] = {"aggregate": {"mean_mse": 1.0}}
                rows.extend(
                    (
                        _row(replicate, latent_rank, horizon, "full_trace", 0.20, 1.0),
                        _row(
                            replicate,
                            latent_rank,
                            horizon,
                            "layerwise_sketch",
                            candidate_by_horizon[horizon],
                            cosine,
                        ),
                        _row(
                            replicate,
                            latent_rank,
                            horizon,
                            "global_sketch",
                            global_by_horizon[horizon],
                        ),
                        _row(
                            replicate,
                            latent_rank,
                            horizon,
                            "reservoir_replay",
                            replay_by_horizon[horizon],
                        ),
                    )
                )
    return rows, initial


class O10HorizonExperimentTests(unittest.TestCase):
    def test_paired_horizons_share_teacher_not_samples(self):
        short = make_v2_task(101, 4, 1, split="confirmation")
        long = make_v2_task(101, 4, 8, split="confirmation")
        self.assertNotEqual(short.name, long.name)
        self.assertEqual(short.train_length, 128)
        self.assertEqual(long.train_length, 1024)
        for left, right in zip(short.base_matrices, long.base_matrices):
            np.testing.assert_array_equal(left, right)
        for left, right in zip(short.teacher_updates, long.teacher_updates):
            np.testing.assert_array_equal(left, right)

    def test_global_control_spends_no_more_than_candidate(self):
        task = make_v2_task(101, 2, 1, split="confirmation")
        budget = sketch_state_floats(task, 8)
        global_rank = global_rank_for_budget(task, budget)
        allocated, used = method_state(task, 8, "global_sketch")
        self.assertEqual(allocated, budget)
        self.assertLessEqual(used, allocated)
        self.assertGreater(global_rank, 8)

    def test_positive_fixture_passes_and_establishes_specificity(self):
        rows, initial = _fixture(
            {1: 0.25, 4: 0.22, 8: 0.20},
            {1: 0.24, 4: 0.26, 8: 0.30},
        )
        result = analyze(
            rows,
            initial,
            replicates=CONFIRMATION_REPLICATES,
            horizons=PRIMARY_HORIZONS,
            rank=8,
            replay_by_horizon={1: "reservoir_replay", 4: "reservoir_replay", 8: "reservoir_replay"},
            config=Config(bootstrap_samples=1000),
            maximum_state_fraction=0.30,
        )
        self.assertTrue(result["gate_passed"], result["checks"])
        self.assertTrue(result["layerwise_specificity_established"])

    def test_clear_failure_is_decisive(self):
        rows, initial = _fixture(
            {1: 0.45, 4: 0.45, 8: 0.45},
            {1: 0.25, 4: 0.25, 8: 0.25},
            cosine=0.50,
        )
        result = analyze(
            rows,
            initial,
            replicates=CONFIRMATION_REPLICATES,
            horizons=PRIMARY_HORIZONS,
            rank=8,
            replay_by_horizon={1: "reservoir_replay", 4: "reservoir_replay", 8: "reservoir_replay"},
            config=Config(bootstrap_samples=1000),
            maximum_state_fraction=0.30,
        )
        self.assertFalse(result["gate_passed"])
        self.assertFalse(result["horizon_extension_warranted"])
        self.assertFalse(result["capacity_rescue_warranted"])
        self.assertTrue(result["decisive_failure"])

    def test_reference_premise_passes(self):
        summary = premise_summary_v2()
        self.assertTrue(summary["gate_passed"], summary["checks"])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class O10HorizonTorchTests(unittest.TestCase):
    def test_every_primary_method_runs_one_step(self):
        task = make_v2_task(9, 2, 1, split="smoke")
        batches = evaluation_batches(task, 4, 91, 0.002)
        config = Config(calibration_eval_examples=4, primary_eval_examples=4)
        for method in (
            "full_trace",
            "layerwise_sketch",
            "global_sketch",
            "reservoir_replay",
            "stratified_replay",
            "gradient_replay",
            "multi_timescale",
        ):
            row = run_one(
                task,
                batches,
                replicate=9,
                horizon=1,
                method=method,
                rank=8,
                steps=1,
                batch_size=2,
                learning_rate=0.008,
                stage="smoke",
                config=config,
                device=torch.device("cpu"),
            )
            self.assertTrue(np.isfinite(row["evaluation"]["aggregate"]["mean_mse"]))
            self.assertEqual(row["training"]["optimizer_steps"], 1)


if __name__ == "__main__":
    unittest.main()
