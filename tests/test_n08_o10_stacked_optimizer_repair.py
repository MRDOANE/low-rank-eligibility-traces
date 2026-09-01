import unittest

from adaptive_memory.n08_o10_stacked_experiment import (
    CANDIDATE,
    GLOBAL_ORACLE,
    LAYERWISE_CONTROL,
    REPLAY_BY_HORIZON,
)
from adaptive_memory.n08_o10_stacked_optimizer_repair import (
    BOUNDARY_TEACHERS,
    CALIBRATION_HOLDOUT_TEACHERS,
    CALIBRATION_TEACHERS,
    CONFIRMATION_TEACHERS,
    SCREEN_TEACHERS,
    Config,
    analyze,
    select_learning_rates,
)


def _selected(horizons):
    result = {
        "full_trace": {},
        CANDIDATE: {},
        LAYERWISE_CONTROL: {},
        GLOBAL_ORACLE: {},
        "gradient_replay": {},
        "reservoir_replay": {},
    }
    for horizon in horizons:
        for method in (
            "full_trace",
            CANDIDATE,
            LAYERWISE_CONTROL,
            GLOBAL_ORACLE,
            REPLAY_BY_HORIZON[horizon],
        ):
            result[method][horizon] = 0.001
    return {method: values for method, values in result.items() if values}


def _row(
    teacher,
    train_replicate,
    latent_rank,
    horizon,
    method,
    mse,
    *,
    learning_rate=0.001,
    cosine=0.95,
):
    full = method == "full_trace"
    used = 16384 if full else (4160 if method == CANDIDATE else 4352)
    return {
        "teacher": teacher,
        "train_replicate": train_replicate,
        "latent_rank": latent_rank,
        "horizon_multiplier": horizon,
        "method": method,
        "learning_rate": learning_rate,
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


def _fixture(candidate_mse, *, full_mse=0.15, cosine=0.95):
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
                replay = REPLAY_BY_HORIZON[horizon]
                rows.extend(
                    (
                        _row(
                            teacher,
                            0,
                            latent_rank,
                            horizon,
                            "full_trace",
                            full_mse,
                            cosine=1.0,
                        ),
                        _row(
                            teacher,
                            0,
                            latent_rank,
                            horizon,
                            CANDIDATE,
                            candidate_mse,
                            cosine=cosine,
                        ),
                        _row(
                            teacher,
                            0,
                            latent_rank,
                            horizon,
                            LAYERWISE_CONTROL,
                            0.24,
                        ),
                        _row(
                            teacher,
                            0,
                            latent_rank,
                            horizon,
                            GLOBAL_ORACLE,
                            0.19,
                        ),
                        _row(teacher, 0, latent_rank, horizon, replay, 0.25),
                    )
                )
    return teachers, horizons, rows, initial, _selected(horizons)


class OptimizerRepairDecisionTests(unittest.TestCase):
    def test_valid_positive_fixture_is_strong(self):
        teachers, horizons, rows, initial, selected = _fixture(0.18)
        result = analyze(
            rows,
            initial,
            teachers=teachers,
            train_replicates=(0,),
            horizons=horizons,
            selected_learning_rates=selected,
            config=Config(bootstrap_samples=1000),
            stage="fixture_positive",
        )
        self.assertEqual(result["analysis_status"], "valid_inference")
        self.assertTrue(result["supportive"], result["core_checks"])
        self.assertTrue(result["strong_positive"], result["strong_checks"])
        self.assertTrue(result["continue_after_screen"], result["screen_checks"])

    def test_valid_clear_failure_stops_at_screen(self):
        teachers, horizons, rows, initial, selected = _fixture(0.60, cosine=0.50)
        result = analyze(
            rows,
            initial,
            teachers=teachers,
            train_replicates=(0,),
            horizons=horizons,
            selected_learning_rates=selected,
            config=Config(bootstrap_samples=1000),
            stage="fixture_failure",
        )
        self.assertTrue(result["reference_valid"])
        self.assertFalse(result["supportive"])
        self.assertFalse(result["strong_positive"])
        self.assertFalse(result["continue_after_screen"])

    def test_invalid_reference_suppresses_all_performance_inference(self):
        teachers, horizons, rows, initial, selected = _fixture(0.18, full_mse=1.10)
        result = analyze(
            rows,
            initial,
            teachers=teachers,
            train_replicates=(0,),
            horizons=horizons,
            selected_learning_rates=selected,
            config=Config(bootstrap_samples=1000),
            stage="fixture_invalid",
        )
        self.assertEqual(
            result["analysis_status"], "invalid_reference_no_performance_inference"
        )
        self.assertFalse(result["reference_valid"])
        self.assertIsNone(result["bootstrap_over_independent_teachers"])
        self.assertEqual(result["core_checks"], {})
        self.assertEqual(result["strong_checks"], {})
        self.assertFalse(result["continue_after_screen"])

    def test_fresh_teacher_sets_do_not_overlap(self):
        groups = (
            set(CALIBRATION_TEACHERS),
            set(CALIBRATION_HOLDOUT_TEACHERS),
            set(SCREEN_TEACHERS),
            set(CONFIRMATION_TEACHERS),
            set(BOUNDARY_TEACHERS),
        )
        for index, group in enumerate(groups):
            for earlier in groups[:index]:
                self.assertTrue(group.isdisjoint(earlier))


class OptimizerRepairCalibrationTests(unittest.TestCase):
    def test_selection_is_method_specific_and_uses_worst_cell_first(self):
        horizons = (8,)
        rates = (0.0005, 0.001, 0.002)
        rows = []
        initial = {}
        methods = (
            "full_trace",
            CANDIDATE,
            LAYERWISE_CONTROL,
            GLOBAL_ORACLE,
            REPLAY_BY_HORIZON[8],
        )
        for teacher in (31, 37):
            initial[f"teacher{teacher}:latent2:h8"] = {
                "aggregate": {"mean_mse": 1.0}
            }
            for method in methods:
                for rate in rates:
                    # 0.001 has the best worst cell. 0.002 has a slightly better
                    # first cell but an unstable second cell.
                    if rate == 0.001:
                        mse = 0.20
                    elif rate == 0.002:
                        mse = 0.18 if teacher == 31 else 0.80
                    else:
                        mse = 0.35
                    rows.append(
                        _row(
                            teacher,
                            0,
                            2,
                            8,
                            method,
                            mse,
                            learning_rate=rate,
                        )
                    )
        selected, scores = select_learning_rates(
            rows,
            initial,
            horizons=horizons,
            learning_rates=rates,
        )
        for method in methods:
            self.assertEqual(selected[method][8], 0.001)
            self.assertEqual(scores[method]["8"][0]["learning_rate"], 0.001)


if __name__ == "__main__":
    unittest.main()
