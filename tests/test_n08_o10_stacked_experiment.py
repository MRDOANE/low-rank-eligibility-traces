import unittest

import numpy as np

from adaptive_memory.n08_o10_stacked_experiment import (
    CANDIDATE,
    GLOBAL_ORACLE,
    LAYERWISE_CONTROL,
    Config,
    analyze,
    make_bridge_task,
)
from adaptive_memory.o10_tasks import sketch_state_floats

try:
    import torch

    from adaptive_memory.o10_layerwise import (
        LayerwiseAdapterModel,
        approximate_trace,
        full_trace,
        method_state,
        stacked_right_subspace_rank_for_budget,
    )
    from adaptive_memory.o10_tasks import sample_batch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


def _row(teacher, train_replicate, latent_rank, horizon, method, mse, cosine=0.95):
    full = method == "full_trace"
    used = 16384 if full else (4160 if method == CANDIDATE else 4352)
    return {
        "teacher": teacher,
        "train_replicate": train_replicate,
        "latent_rank": latent_rank,
        "horizon_multiplier": horizon,
        "method": method,
        "training": {
            "parameters": 16384,
            "optimizer_steps": 100,
            "allocated_state_floats_per_episode": 16384 if full else 4352,
            "used_state_floats_per_episode": used,
            "averaged_steps": 50,
            "mean_gradient_cosine": 1.0 if full else cosine,
        },
        "evaluation": {"aggregate": {"mean_mse": mse}},
        "endpoint_evaluation": {"aggregate": {"mean_mse": mse + 0.01}},
    }


def _fixture(candidate_mse, cosine=0.95):
    teachers = (11, 13, 17)
    horizons = (8, 32)
    rows = []
    initial = {}
    for teacher in teachers:
        for latent_rank in (2, 4):
            for horizon in horizons:
                initial[f"teacher{teacher}:latent{latent_rank}:h{horizon}"] = {
                    "aggregate": {"mean_mse": 1.0}
                }
                replay = "gradient_replay" if horizon == 8 else "reservoir_replay"
                rows.extend(
                    (
                        _row(teacher, 0, latent_rank, horizon, "full_trace", 0.15, 1.0),
                        _row(
                            teacher,
                            0,
                            latent_rank,
                            horizon,
                            CANDIDATE,
                            candidate_mse,
                            cosine,
                        ),
                        _row(
                            teacher, 0, latent_rank, horizon, LAYERWISE_CONTROL, 0.24
                        ),
                        _row(teacher, 0, latent_rank, horizon, GLOBAL_ORACLE, 0.19),
                        _row(teacher, 0, latent_rank, horizon, replay, 0.25),
                    )
                )
    return teachers, horizons, rows, initial


class StackedDecisionTests(unittest.TestCase):
    def test_positive_fixture_is_strong(self):
        teachers, horizons, rows, initial = _fixture(0.18)
        result = analyze(
            rows,
            initial,
            teachers=teachers,
            train_replicates=(0,),
            horizons=horizons,
            config=Config(bootstrap_samples=1000),
            stage="fixture_positive",
        )
        self.assertTrue(result["supportive"], result["core_checks"])
        self.assertTrue(result["strong_positive"], result["strong_checks"])
        self.assertTrue(result["continue_after_screen"], result["screen_checks"])

    def test_clear_failure_stops_at_screen(self):
        teachers, horizons, rows, initial = _fixture(0.60, cosine=0.50)
        result = analyze(
            rows,
            initial,
            teachers=teachers,
            train_replicates=(0,),
            horizons=horizons,
            config=Config(bootstrap_samples=1000),
            stage="fixture_failure",
        )
        self.assertFalse(result["supportive"])
        self.assertFalse(result["strong_positive"])
        self.assertFalse(result["continue_after_screen"])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class StackedTensorTests(unittest.TestCase):
    def test_stacked_right_subspace_uses_the_matched_budget(self):
        task = make_bridge_task(17, 2, 1, split="test")
        budget = sketch_state_floats(task, 8)
        allocated, used = method_state(task, 8, CANDIDATE)
        self.assertEqual(budget, 4352)
        self.assertEqual(allocated, budget)
        self.assertEqual(stacked_right_subspace_rank_for_budget(task, budget), 13)
        self.assertEqual(used, 4160)
        self.assertLessEqual(used, allocated)

    def test_stacked_right_subspace_is_finite_and_layer_shaped(self):
        task = make_bridge_task(19, 2, 1, split="test")
        batch = sample_batch(task, 2, 23)
        model = LayerwiseAdapterModel(task).eval()
        _, left, right = model.forward_factors(torch.as_tensor(batch.x))
        exact = full_trace(left, right)
        generator = torch.Generator(device="cpu").manual_seed(29)
        estimate = approximate_trace(CANDIDATE, model, left, right, 8, generator)
        self.assertEqual(estimate.shape, exact.shape)
        self.assertTrue(torch.isfinite(estimate).all())
        flattened_estimate = estimate.flatten(start_dim=1)
        flattened_exact = exact.flatten(start_dim=1)
        cosine = torch.nn.functional.cosine_similarity(
            flattened_estimate, flattened_exact, dim=1
        )
        self.assertGreater(float(torch.mean(cosine)), 0.50)

if __name__ == "__main__":
    unittest.main()
