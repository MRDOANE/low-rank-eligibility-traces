import unittest

import torch

from n08cascade.credit import CreditCodec


class CreditCodecTest(unittest.TestCase):
    def setUp(self) -> None:
        self.layers = 3
        self.dimension = 8
        torch.manual_seed(5)
        self.trace = torch.randn(4, self.layers, self.dimension, self.dimension)
        self.logits = torch.randn(4)
        self.left = torch.randn(4, self.layers, 6, self.dimension)
        self.right = torch.randn(4, self.layers, 6, self.dimension)

    def test_global_svd_payload_is_exactly_declared(self) -> None:
        codec = CreditCodec(
            "global_svd",
            layers=self.layers,
            dimension=self.dimension,
            candidate_rank=2,
            seed=1,
        )
        encoded, quality = codec.encode(self.trace, self.logits, left=self.left, right=self.right)
        expected = codec.target_floats * 4 * len(self.logits)
        self.assertEqual(encoded.nbytes, expected)
        decoded, logits = codec.decode(encoded, torch.device("cpu"))
        self.assertEqual(tuple(decoded.shape), tuple(self.trace.shape))
        torch.testing.assert_close(logits, self.logits)
        self.assertEqual(len(quality["gradient_cosine"]), len(self.logits))

    def test_matched_codecs_fit_primary_budget(self) -> None:
        for method in ("random_projection", "o10_layerwise", "o10_shared"):
            with self.subTest(method=method):
                codec = CreditCodec(
                    method,
                    layers=self.layers,
                    dimension=self.dimension,
                    candidate_rank=2,
                    seed=1,
                )
                encoded, _ = codec.encode(self.trace, self.logits, left=self.left, right=self.right)
                self.assertLessEqual(encoded.nbytes, codec.target_floats * 4 * len(self.logits))
                decoded, _ = codec.decode(encoded, torch.device("cpu"))
                self.assertTrue(torch.isfinite(decoded).all())

    def test_exact_round_trip(self) -> None:
        codec = CreditCodec(
            "exact_credit",
            layers=self.layers,
            dimension=self.dimension,
            candidate_rank=1,
            seed=1,
        )
        encoded, _ = codec.encode(self.trace, self.logits, left=self.left, right=self.right)
        decoded, _ = codec.decode(encoded, torch.device("cpu"))
        torch.testing.assert_close(decoded, self.trace)


if __name__ == "__main__":
    unittest.main()

