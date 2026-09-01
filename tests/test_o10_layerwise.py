import unittest

try:
    import torch

    from adaptive_memory.o10_layerwise import (
        MATCHED_CONTROLS,
        LayerwiseAdapterModel,
        approximate_trace,
        full_trace,
        method_state,
    )
    from adaptive_memory.o10_tasks import sample_batch, task_suite

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class O10LayerwiseTests(unittest.TestCase):
    def test_all_matched_methods_receive_the_same_allocated_state(self):
        for task in task_suite().values():
            candidate, _ = method_state(task, 8, "layerwise_sketch")
            for method in MATCHED_CONTROLS:
                allocated, used = method_state(task, 8, method)
                self.assertEqual(allocated, candidate)
                self.assertLessEqual(used, allocated)

    def test_trace_outputs_have_exact_layerwise_shape(self):
        task = next(iter(task_suite().values()))
        batch = sample_batch(task, 3, 21)
        model = LayerwiseAdapterModel(task).eval()
        x = torch.as_tensor(batch.x)
        _, left, right = model.forward_factors(x)
        generator = torch.Generator(device="cpu").manual_seed(10)
        exact = full_trace(left, right)
        for method in ("layerwise_sketch", *MATCHED_CONTROLS):
            estimate = approximate_trace(method, model, left, right, 8, generator)
            self.assertEqual(estimate.shape, exact.shape)
            self.assertTrue(torch.isfinite(estimate).all())

    def test_permuting_layers_changes_the_candidate_update(self):
        task = next(iter(task_suite().values()))
        batch = sample_batch(task, 4, 25)
        model = LayerwiseAdapterModel(task).eval()
        _, left, right = model.forward_factors(torch.as_tensor(batch.x))
        generator = torch.Generator(device="cpu").manual_seed(13)
        candidate = approximate_trace(
            "layerwise_sketch", model, left, right, 8, generator
        )
        permuted = approximate_trace(
            "permuted_layer_sketch", model, left, right, 8, generator
        )
        self.assertFalse(torch.allclose(candidate, permuted))


if __name__ == "__main__":
    unittest.main()
