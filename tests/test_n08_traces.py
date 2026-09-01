import unittest

import numpy as np

try:
    import torch

    from adaptive_memory.n08_tasks import sample_batch, task_suite
    from adaptive_memory.n08_traces import (
        METHODS,
        DelayedAdapterModel,
        approximate_trace,
        full_trace,
        gradient_cosine,
        streaming_lowrank_trace,
        train_method,
    )

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class N08TraceTests(unittest.TestCase):
    def test_full_rank_streaming_update_matches_exact_trace(self):
        generator = torch.Generator().manual_seed(7)
        left = torch.randn(3, 2, 7, 5, generator=generator)
        right = torch.randn(3, 2, 7, 5, generator=generator)
        exact = full_trace(left, right)
        streamed = streaming_lowrank_trace(left, right, rank=5)
        self.assertTrue(torch.allclose(streamed, exact, atol=2e-5, rtol=2e-5))

    def test_exact_trace_has_unit_per_episode_gradient_cosine(self):
        generator = torch.Generator().manual_seed(11)
        exact = torch.randn(8, 3, 6, 6, generator=generator)
        error = torch.randn(8, generator=generator)
        self.assertAlmostEqual(gradient_cosine(error, exact, exact), 1.0, places=6)

    def test_every_method_has_the_same_trace_shape_and_finite_values(self):
        task = next(iter(task_suite().values()))
        batch = sample_batch(task, 4, 17)
        model = DelayedAdapterModel(task).eval()
        _, left, right = model.forward_factors(torch.as_tensor(batch.x))
        expected = (4, task.layers, task.dim, task.dim)
        for method in METHODS:
            generator = torch.Generator().manual_seed(23)
            trace = approximate_trace(method, task, left, right, 2, generator)
            self.assertEqual(tuple(trace.shape), expected)
            self.assertTrue(bool(torch.isfinite(trace).all()))

    def test_two_step_cpu_training_smoke_is_deterministic(self):
        task = next(iter(task_suite().values()))
        kwargs = dict(
            task=task,
            method="lowrank_svd",
            seed=29,
            rank=2,
            steps=2,
            batch_size=4,
            learning_rate=0.01,
            weight_decay=0.0001,
            device=torch.device("cpu"),
        )
        first = train_method(**kwargs)
        second = train_method(**kwargs)
        for left, right in zip(first.weights, second.weights):
            self.assertTrue(np.array_equal(left, right))
        self.assertEqual(first.trajectory, second.trajectory)
        self.assertTrue(np.isfinite(first.final_train_mse))


if __name__ == "__main__":
    unittest.main()
