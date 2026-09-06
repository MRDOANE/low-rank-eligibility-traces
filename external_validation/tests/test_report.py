import unittest

from n08cascade.report import build_gate, paired_intervals, select_rank


def full_result(
    benchmark: str,
    method: str,
    delay: int | str,
    seed: int,
    *,
    accuracy: float = 0.8,
    log_loss: float = 0.45,
    balanced_accuracy: float = 0.8,
    positive_rate: float = 0.2,
) -> dict:
    group = {
        "count": 100,
        "accuracy": accuracy,
        "log_loss": log_loss,
        "balanced_accuracy": balanced_accuracy,
        "positive_rate": positive_rate,
    }
    predictive = {"overall": dict(group)}
    if benchmark == "criteo":
        predictive["split:test"] = dict(group)
        # These outcome-linked bins must never become separate gate votes.
        predictive["split:test|delay:delay_seconds=[0,3600)"] = dict(group)
        predictive["split:test|delay:delay_seconds=[2592000,inf)"] = dict(group)
    return {
        "trial": {
            "stage": "full",
            "benchmark": benchmark,
            "delay": delay,
            "seed": seed,
            "method": method,
        },
        "predictive": predictive,
    }


def audit_result(
    benchmark: str,
    method: str,
    delay: int | str,
    seed: int,
    *,
    rank: int,
    cosine: float = 0.98,
    loss: float = 0.40,
    accuracy: float = 0.80,
    predictions_after_feedback: int = 5000,
) -> dict:
    return {
        "trial": {
            "stage": "audit",
            "benchmark": benchmark,
            "delay": delay,
            "seed": seed,
            "method": method,
            "rank": rank,
        },
        "predictive": {
            "overall": {
                "accuracy": accuracy,
                "log_loss": loss,
            }
        },
        "credit_quality": {"gradient_cosine_median": cosine},
        "feedback_diagnostics": {
            "predictions_after_first_feedback": predictions_after_feedback
        },
        "memory_reduction_vs_exact_per_event": 4.0 if method == "global_svd" else 1.0,
    }


GATE = {
    "max_accuracy_gap_percentage_points": 1.0,
    "max_relative_excess_loss": 0.005,
    "min_median_gradient_cosine": 0.95,
    "min_memory_reduction": 3.0,
    "min_delays_with_positive_ci": 2,
    "minimum_predictions_after_feedback": 1024,
    "min_yearbook_oracle_balanced_accuracy": 0.55,
    "min_criteo_oracle_relative_logloss_gain_over_constant": 0.005,
}


def competent_oracles(seeds: tuple[int, ...]) -> list[dict]:
    rows = []
    for delay in (10, 50):
        for seed in seeds:
            rows.append(
                full_result(
                    "yearbook",
                    "immediate_oracle",
                    delay,
                    seed,
                    balanced_accuracy=0.75,
                )
            )
    for seed in seeds:
        rows.append(
            full_result(
                "criteo",
                "immediate_oracle",
                "natural",
                seed,
                log_loss=0.35,
                positive_rate=0.2,
            )
        )
    return rows


class ReportTest(unittest.TestCase):
    def test_two_conditions_for_one_comparator_open_gate(self) -> None:
        seeds = (11, 23, 37, 53, 71)
        rows = []
        for delay in (10, 50):
            for seed in seeds:
                rows.extend(
                    [
                        full_result("yearbook", "global_svd", delay, seed, accuracy=0.82),
                        full_result("yearbook", "equal_byte_replay", delay, seed, accuracy=0.78),
                        full_result("yearbook", "random_projection", delay, seed, accuracy=0.84),
                    ]
                )
        intervals = paired_intervals(rows, 0.95)
        rank_audit = {
            "selected_rank": 4,
            "candidates": [
                {
                    "rank": 4,
                    "performance_pass": True,
                    "cosine_pass": True,
                    "feedback_exposure_pass": True,
                    "memory_pass": True,
                }
            ],
        }
        gate = build_gate(
            rank_audit=rank_audit,
            intervals=intervals,
            full_results=rows + competent_oracles(seeds),
            gate_config=GATE,
        )
        self.assertTrue(gate["all_pass"])
        self.assertEqual(gate["successful_comparators"], ["equal_byte_replay"])

    def test_mixed_comparator_wins_do_not_open_gate(self) -> None:
        seeds = (11, 23, 37, 53, 71)
        rows = []
        for delay in (10, 50):
            for seed in seeds:
                rows.extend(
                    [
                        full_result("yearbook", "global_svd", delay, seed, accuracy=0.82),
                        full_result(
                            "yearbook",
                            "equal_byte_replay",
                            delay,
                            seed,
                            accuracy=0.78 if delay == 10 else 0.84,
                        ),
                        full_result(
                            "yearbook",
                            "random_projection",
                            delay,
                            seed,
                            accuracy=0.84 if delay == 10 else 0.78,
                        ),
                    ]
                )
        intervals = paired_intervals(rows, 0.95)
        rank_audit = {
            "selected_rank": 4,
            "candidates": [
                {
                    "rank": 4,
                    "performance_pass": True,
                    "cosine_pass": True,
                    "feedback_exposure_pass": True,
                    "memory_pass": True,
                }
            ],
        }
        gate = build_gate(
            rank_audit=rank_audit,
            intervals=intervals,
            full_results=rows + competent_oracles(seeds),
            gate_config=GATE,
        )
        self.assertFalse(
            gate["criteria"][
                "paired_advantage_for_one_comparator_at_two_or_more_conditions"
            ]
        )

    def test_criteo_is_one_confirmatory_condition(self) -> None:
        seeds = (11, 23, 37, 53, 71)
        rows = []
        for seed in seeds:
            rows.extend(
                [
                    full_result("criteo", "global_svd", "natural", seed, log_loss=0.40),
                    full_result(
                        "criteo", "equal_byte_replay", "natural", seed, log_loss=0.50
                    ),
                    full_result(
                        "criteo", "random_projection", "natural", seed, log_loss=0.55
                    ),
                ]
            )
        intervals = paired_intervals(rows, 0.95)
        self.assertEqual(len(intervals), 2)
        self.assertEqual(
            {row["condition_id"] for row in intervals},
            {"criteo:test_natural_delay"},
        )

    def test_rank_audit_cannot_pool_weak_criteo_cosine_under_yearbook(self) -> None:
        rows = []
        for benchmark, delays, cosine in (
            ("yearbook", (10, 50, 100), 0.99),
            ("criteo", ("natural",), 0.90),
        ):
            for delay in delays:
                for seed in (11, 23, 37, 53, 71):
                    rows.append(
                        audit_result(
                            benchmark,
                            "exact_credit",
                            delay,
                            seed,
                            rank=1,
                            cosine=1.0,
                        )
                    )
                    rows.append(
                        audit_result(
                            benchmark,
                            "global_svd",
                            delay,
                            seed,
                            rank=4,
                            cosine=cosine,
                        )
                    )
        selected_rank, audit = select_rank(rows, [4], GATE)
        self.assertEqual(selected_rank, 4)
        candidate = audit["candidates"][0]
        self.assertFalse(candidate["cosine_pass"])
        self.assertFalse(candidate["rank_pass"])
        criteo_cell = next(
            cell for cell in candidate["cells"] if cell["benchmark"] == "criteo"
        )
        self.assertAlmostEqual(criteo_cell["median_gradient_cosine"], 0.90)

    def test_incompetent_oracle_makes_scientific_status_red(self) -> None:
        seeds = (11, 23, 37, 53, 71)
        rank_audit = {
            "selected_rank": 4,
            "candidates": [
                {
                    "rank": 4,
                    "performance_pass": True,
                    "cosine_pass": True,
                    "feedback_exposure_pass": True,
                    "memory_pass": True,
                }
            ],
        }
        weak_oracles = competent_oracles(seeds)
        for item in weak_oracles:
            if item["trial"]["benchmark"] == "yearbook":
                item["predictive"]["overall"]["balanced_accuracy"] = 0.50
        gate = build_gate(
            rank_audit=rank_audit,
            intervals=[],
            full_results=weak_oracles,
            gate_config=GATE,
        )
        self.assertFalse(gate["all_valid"])
        self.assertEqual(gate["color"], "RED")


if __name__ == "__main__":
    unittest.main()
