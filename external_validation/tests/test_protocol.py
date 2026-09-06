import json
import unittest
from pathlib import Path


class ProtocolTest(unittest.TestCase):
    def test_criteo_feature_allowlist_has_no_outcome_fields(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "config" / "frozen_protocol.json").read_text(encoding="utf-8"))
        safe = set(config["criteo"]["safe_feature_columns"])
        forbidden = set(config["criteo"]["forbidden_model_columns"])
        self.assertFalse(safe.intersection(forbidden))
        self.assertIn("conversion_timestamp", forbidden)
        self.assertNotIn("conversion_timestamp", safe)

    def test_five_seeds_three_delays_and_several_ranks_are_frozen(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "config" / "frozen_protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(len(config["seeds"]), 5)
        self.assertEqual(config["yearbook"]["delays_in_batches"], [10, 50, 100])
        self.assertGreaterEqual(len(config["ranks"]), 3)

    def test_yearbook_audit_extends_past_every_delayed_feedback_horizon(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "config" / "frozen_protocol.json").read_text(encoding="utf-8"))
        latest_first_feedback = (
            max(config["yearbook"]["delays_in_batches"])
            * config["yearbook"]["batch_size"]
        )
        self.assertGreaterEqual(
            config["audit"]["yearbook_examples"] - latest_first_feedback,
            config["gate"]["minimum_predictions_after_feedback"],
        )


if __name__ == "__main__":
    unittest.main()
