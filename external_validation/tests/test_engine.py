import tempfile
import unittest
from pathlib import Path

import torch

from n08cascade.data import SyntheticStream
from n08cascade.engine import run_trial


class EngineTest(unittest.TestCase):
    def test_delayed_global_trial_completes_and_drains(self) -> None:
        model_config = {
            "dimension": 8,
            "layers": 3,
            "adapter_scale": 0.1,
            "readout_scale": 0.15,
            "learning_rate": 0.01,
            "momentum": 0.9,
            "weight_decay": 1e-5,
            "gradient_clip": 5.0,
        }
        stream = SyntheticStream(seed=8, examples=32, positions=10, dimension=8, variable_delay=True)
        trial = {
            "stage": "test",
            "benchmark": "criteo",
            "delay": "natural",
            "seed": 9,
            "method": "global_svd",
            "rank": 2,
            "config_digest": "test",
            "update_batch_size": 16,
        }
        with tempfile.TemporaryDirectory() as directory:
            result = run_trial(
                trial=trial,
                model_config=model_config,
                batches=stream.batches(8, device=torch.device("cpu")),
                device=torch.device("cpu"),
                output_path=Path(directory) / "trial.json",
            )
        self.assertEqual(result["observed_examples"], 32)
        self.assertEqual(result["updated_examples"], 32)
        self.assertGreater(result["state"]["peak_bytes"], 0)
        self.assertEqual(result["credit_quality"]["gradient_cosine_count"], 32)


if __name__ == "__main__":
    unittest.main()

