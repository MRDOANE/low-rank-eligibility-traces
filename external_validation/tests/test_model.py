import unittest

import torch

from n08cascade.engine import apply_trace_step
from n08cascade.model import TraceAdapter, exact_trace


class TraceAdapterTest(unittest.TestCase):
    def test_analytic_trace_matches_autograd(self) -> None:
        torch.manual_seed(3)
        model = TraceAdapter(4, 2, seed=7, adapter_scale=0.2, readout_scale=0.15)
        tokens = torch.randn(1, 5, 4)
        logits, left, right = model.forward_factors(tokens)
        analytic = exact_trace(left, right)[0]
        automatic = torch.autograd.grad(logits[0], model.updates)[0]
        torch.testing.assert_close(analytic, automatic, rtol=2e-5, atol=2e-6)

    def test_batch_shape(self) -> None:
        model = TraceAdapter(8, 3, seed=1, adapter_scale=0.1, readout_scale=0.15)
        logits, left, right = model.forward_factors(torch.randn(6, 7, 8))
        self.assertEqual(tuple(logits.shape), (6,))
        self.assertEqual(tuple(exact_trace(left, right).shape), (6, 3, 8, 8))

    def test_traced_matrices_can_learn_a_constant_class_prior(self) -> None:
        model = TraceAdapter(8, 3, seed=1, adapter_scale=1.0, readout_scale=1.0)
        tokens = torch.zeros(64, 7, 8)
        labels = torch.zeros(64)
        optimizer = torch.optim.AdamW([model.updates], lr=0.01, weight_decay=1e-5)
        for _ in range(100):
            with torch.no_grad():
                logits, left, right = model.forward_factors(tokens)
                trace = exact_trace(left, right)
            apply_trace_step(model, optimizer, trace, logits, labels, 5.0)
        self.assertLess(float(torch.sigmoid(model(tokens)).mean()), 0.10)


if __name__ == "__main__":
    unittest.main()
