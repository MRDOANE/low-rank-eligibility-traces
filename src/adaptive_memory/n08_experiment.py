from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .n08_tasks import (
    DelayedOutcomeTask,
    boundary_suite,
    premise_summary,
    sample_batch,
    task_audit,
    task_suite,
    zero_student_scores,
)
from .n08_traces import (
    MEMORY_MATCHED_CONTROLS,
    METHODS,
    predict,
    train_method,
)


SEEDS = (7, 19, 31, 43, 59)
BOUNDARY_SEEDS = (7, 19, 31)
OOD_CONDITIONS = (
    "length_2x",
    "length_4x",
    "noise_shift",
    "scale_shift",
    "joint_shift",
)


@dataclass(frozen=True)
class Config:
    trace_rank: int = 2
    screen_steps: int = 400
    confirmation_steps: int = 550
    boundary_steps: int = 550
    batch_size: int = 64
    screen_eval_examples: int = 512
    confirmation_eval_examples: int = 768
    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    minimum_full_learning_fraction: float = 0.50
    minimum_candidate_full_gain_retention: float = 0.90
    minimum_control_gain: float = 0.10
    minimum_gradient_cosine: float = 0.90
    minimum_cosine_margin: float = 0.01
    maximum_task_degradation: float = 0.05
    maximum_condition_degradation: float = 0.10
    maximum_state_fraction: float = 0.20
    required_successful_seeds: int = 4
    required_task_seed_win_fraction: float = 0.80
    bootstrap_samples: int = 4000
    minimum_boundary_retention: float = 0.75
    minimum_coverage_retention_spearman: float = 0.60


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def evaluation_batches(
    task: DelayedOutcomeTask,
    n: int,
    seed: int,
) -> dict[str, object]:
    return {
        "iid": sample_batch(task, n, seed),
        "length_2x": sample_batch(task, n, seed + 1, length=2 * task.train_length),
        "length_4x": sample_batch(task, n, seed + 2, length=4 * task.train_length),
        "noise_shift": sample_batch(task, n, seed + 3, input_noise=0.12),
        "scale_shift": sample_batch(task, n, seed + 4, latent_scale=1.40),
        "joint_shift": sample_batch(
            task,
            n,
            seed + 5,
            length=2 * task.train_length,
            input_noise=0.12,
            latent_scale=1.40,
        ),
    }


def initial_evaluation(
    task: DelayedOutcomeTask,
    batches: dict[str, object],
) -> dict[str, object]:
    result = {}
    for name, batch in batches.items():
        prediction = zero_student_scores(task, batch.x).mean(axis=1)
        result[name] = {
            "terminal_mse": float(np.mean((prediction - batch.target) ** 2)),
            "terminal_mae": float(np.mean(np.abs(prediction - batch.target))),
        }
    result["aggregate"] = {
        "mean_mse": float(np.mean([result[name]["terminal_mse"] for name in batches])),
        "ood_mean_mse": float(
            np.mean([result[name]["terminal_mse"] for name in OOD_CONDITIONS])
        ),
        "ood_worst_mse": float(
            np.max([result[name]["terminal_mse"] for name in OOD_CONDITIONS])
        ),
    }
    return result


def evaluate_weights(
    task: DelayedOutcomeTask,
    weights: tuple[np.ndarray, ...],
    batches: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    result = {}
    for name, batch in batches.items():
        prediction = predict(task, weights, batch.x, device)
        result[name] = {
            "terminal_mse": float(np.mean((prediction - batch.target) ** 2)),
            "terminal_mae": float(np.mean(np.abs(prediction - batch.target))),
            "prediction_mean": float(np.mean(prediction)),
            "target_mean": float(np.mean(batch.target)),
        }
    result["aggregate"] = {
        "mean_mse": float(np.mean([result[name]["terminal_mse"] for name in batches])),
        "ood_mean_mse": float(
            np.mean([result[name]["terminal_mse"] for name in OOD_CONDITIONS])
        ),
        "ood_worst_mse": float(
            np.max([result[name]["terminal_mse"] for name in OOD_CONDITIONS])
        ),
    }
    return result


def run_method(
    task: DelayedOutcomeTask,
    batches: dict[str, object],
    *,
    method: str,
    seed: int,
    steps: int,
    config: Config,
    device: torch.device,
) -> dict[str, object]:
    trained = train_method(
        task,
        method=method,
        seed=seed,
        rank=config.trace_rank,
        steps=steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=device,
    )
    return {
        "task": task.name,
        "latent_rank": task.latent_rank,
        "method": method,
        "seed": seed,
        "training": {
            "runtime_seconds": trained.runtime_seconds,
            "parameters": trained.parameters,
            "optimizer_steps": trained.optimizer_steps,
            "state_floats_per_episode": trained.state_floats_per_episode,
            "state_bytes_fp32_per_episode": 4 * trained.state_floats_per_episode,
            "mean_gradient_cosine": trained.mean_gradient_cosine,
            "final_gradient_cosine": trained.final_gradient_cosine,
            "final_train_mse": trained.final_train_mse,
            "trajectory": trained.trajectory,
        },
        "evaluation": evaluate_weights(task, trained.weights, batches, device),
    }


def stage_runs(
    tasks: dict[str, DelayedOutcomeTask],
    *,
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
    steps: int,
    eval_examples: int,
    stage: str,
    config: Config,
    device: torch.device,
    progress_path: Path,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    rows = []
    initial = {}
    for task_index, task in enumerate(tasks.values()):
        for seed in seeds:
            eval_seed = 900_000 + task_index * 20_000 + seed
            batches = evaluation_batches(task, eval_examples, eval_seed)
            initial[f"{task.name}:{seed}"] = initial_evaluation(task, batches)
            for method in methods:
                row = run_method(
                    task,
                    batches,
                    method=method,
                    seed=seed,
                    steps=steps,
                    config=config,
                    device=device,
                )
                rows.append(row)
                _append_jsonl(
                    progress_path,
                    {
                        "stage": stage,
                        "task": task.name,
                        "latent_rank": task.latent_rank,
                        "seed": seed,
                        "method": method,
                        "ood_mean_mse": row["evaluation"]["aggregate"]["ood_mean_mse"],
                        "mean_gradient_cosine": row["training"]["mean_gradient_cosine"],
                    },
                )
    return rows, initial


def _mean_metric(
    rows: list[dict[str, object]],
    method: str,
    field: str,
    *,
    task: str | None = None,
    seed: int | None = None,
    condition: str | None = None,
) -> float:
    values = []
    for row in rows:
        if row["method"] != method:
            continue
        if task is not None and row["task"] != task:
            continue
        if seed is not None and row["seed"] != seed:
            continue
        if condition is None:
            values.append(float(row["evaluation"]["aggregate"][field]))
        else:
            values.append(float(row["evaluation"][condition][field]))
    if not values:
        raise ValueError(f"empty metric group: {method}, {field}, {task}, {seed}, {condition}")
    return float(np.mean(values))


def _mean_training(
    rows: list[dict[str, object]],
    method: str,
    field: str,
    *,
    task: str | None = None,
    seed: int | None = None,
) -> float:
    values = [
        float(row["training"][field])
        for row in rows
        if row["method"] == method
        and (task is None or row["task"] == task)
        and (seed is None or row["seed"] == seed)
    ]
    if not values:
        raise ValueError("empty training metric group")
    return float(np.mean(values))


def _initial_metric(
    initial: dict[str, dict[str, object]],
    field: str,
    *,
    task: str | None = None,
    seed: int | None = None,
) -> float:
    values = []
    for key, evaluation in initial.items():
        key_task, key_seed = key.rsplit(":", 1)
        if task is not None and key_task != task:
            continue
        if seed is not None and int(key_seed) != seed:
            continue
        values.append(float(evaluation["aggregate"][field]))
    if not values:
        raise ValueError("empty initial metric group")
    return float(np.mean(values))


def _relative_gain(candidate: float, baseline: float) -> float:
    return (baseline - candidate) / max(abs(baseline), 1e-12)


def _gain_retention(initial: float, candidate: float, full: float) -> float:
    full_gain = initial - full
    return (initial - candidate) / max(full_gain, 1e-12)


def screen_decision(
    rows: list[dict[str, object]],
    initial: dict[str, dict[str, object]],
    config: Config,
) -> dict[str, object]:
    control_mse = {
        method: _mean_metric(rows, method, "ood_mean_mse")
        for method in MEMORY_MATCHED_CONTROLS
    }
    strongest = min(control_mse, key=control_mse.get)
    candidate = _mean_metric(rows, "lowrank_svd", "ood_mean_mse")
    full = _mean_metric(rows, "full_trace", "ood_mean_mse")
    initial_mse = _initial_metric(initial, "ood_mean_mse")
    full_learning_fraction = _relative_gain(full, initial_mse)
    retention = _gain_retention(initial_mse, candidate, full)
    control_gain = _relative_gain(candidate, control_mse[strongest])
    candidate_cosine = _mean_training(rows, "lowrank_svd", "mean_gradient_cosine")
    control_cosines = {
        method: _mean_training(rows, method, "mean_gradient_cosine")
        for method in MEMORY_MATCHED_CONTROLS
    }
    strongest_control_cosine = max(control_cosines.values())
    task_results = {}
    for task in task_suite():
        candidate_task = _mean_metric(
            rows, "lowrank_svd", "ood_mean_mse", task=task
        )
        control_task = _mean_metric(rows, strongest, "ood_mean_mse", task=task)
        full_task = _mean_metric(rows, "full_trace", "ood_mean_mse", task=task)
        initial_task = _initial_metric(initial, "ood_mean_mse", task=task)
        task_results[task] = {
            "initial_ood_mse": initial_task,
            "full_trace_ood_mse": full_task,
            "candidate_ood_mse": candidate_task,
            "strongest_control_ood_mse": control_task,
            "relative_gain_over_control": _relative_gain(candidate_task, control_task),
            "full_gain_retention": _gain_retention(
                initial_task, candidate_task, full_task
            ),
        }
    condition_results = {}
    for condition in OOD_CONDITIONS:
        candidate_condition = _mean_metric(
            rows,
            "lowrank_svd",
            "terminal_mse",
            condition=condition,
        )
        control_condition = _mean_metric(
            rows, strongest, "terminal_mse", condition=condition
        )
        condition_results[condition] = {
            "candidate_mse": candidate_condition,
            "control_mse": control_condition,
            "relative_gain": _relative_gain(candidate_condition, control_condition),
        }
    candidate_state = int(
        next(row for row in rows if row["method"] == "lowrank_svd")["training"][
            "state_floats_per_episode"
        ]
    )
    full_state = int(
        next(row for row in rows if row["method"] == "full_trace")["training"][
            "state_floats_per_episode"
        ]
    )
    matched_parameters = all(
        len({row["training"]["parameters"] for row in rows if row["task"] == task})
        == 1
        for task in task_suite()
    )
    matched_steps = len({row["training"]["optimizer_steps"] for row in rows}) == 1
    checks = {
        "full_trace_learns_at_least_half_of_available_ood_error": full_learning_fraction
        >= config.minimum_full_learning_fraction,
        "candidate_retains_at_least_0_90_of_full_trace_gain": retention
        >= config.minimum_candidate_full_gain_retention,
        "candidate_beats_strongest_equal_budget_control_by_0_10": control_gain
        >= config.minimum_control_gain,
        "candidate_mean_gradient_cosine_is_at_least_0_90": candidate_cosine
        >= config.minimum_gradient_cosine,
        "candidate_cosine_exceeds_best_control_by_0_01": candidate_cosine
        >= strongest_control_cosine + config.minimum_cosine_margin,
        "neither_primary_task_degrades_by_more_than_0_05": min(
            item["relative_gain_over_control"] for item in task_results.values()
        )
        >= -config.maximum_task_degradation,
        "no_ood_condition_degrades_by_more_than_0_10": min(
            item["relative_gain"] for item in condition_results.values()
        )
        >= -config.maximum_condition_degradation,
        "candidate_uses_at_most_0_20_of_full_trace_state": candidate_state / full_state
        <= config.maximum_state_fraction,
        "all_methods_have_matched_parameters": matched_parameters,
        "all_methods_have_matched_optimizer_steps": matched_steps,
    }
    return {
        "strongest_memory_matched_control": strongest,
        "control_ood_mse": control_mse,
        "initial_ood_mse": initial_mse,
        "full_trace_ood_mse": full,
        "candidate_ood_mse": candidate,
        "full_trace_learning_fraction": full_learning_fraction,
        "candidate_full_gain_retention": retention,
        "candidate_relative_gain_over_control": control_gain,
        "candidate_mean_gradient_cosine": candidate_cosine,
        "control_mean_gradient_cosines": control_cosines,
        "candidate_state_floats_per_episode": candidate_state,
        "full_state_floats_per_episode": full_state,
        "candidate_state_fraction": candidate_state / full_state,
        "task_results": task_results,
        "condition_results": condition_results,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }


def _bootstrap(values: list[float], samples: int, seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(len(array), size=(samples, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
    }


def confirmation_decision(
    rows: list[dict[str, object]],
    initial: dict[str, dict[str, object]],
    config: Config,
) -> dict[str, object]:
    aggregate = screen_decision(rows, initial, config)
    strongest = aggregate["strongest_memory_matched_control"]
    seed_decisions = []
    cells = []
    differences = []
    for seed in SEEDS:
        selected_rows = [row for row in rows if row["seed"] == seed]
        selected_initial = {
            key: value for key, value in initial.items() if int(key.rsplit(":", 1)[1]) == seed
        }
        decision = screen_decision(selected_rows, selected_initial, config)
        seed_decisions.append(
            {"seed": seed, "gate_passed": decision["gate_passed"], "decision": decision}
        )
        for task in task_suite():
            candidate = _mean_metric(
                rows, "lowrank_svd", "ood_mean_mse", task=task, seed=seed
            )
            control = _mean_metric(
                rows, strongest, "ood_mean_mse", task=task, seed=seed
            )
            differences.append(control - candidate)
            cells.append(
                {
                    "seed": seed,
                    "task": task,
                    "candidate_ood_mse": candidate,
                    "control_ood_mse": control,
                    "candidate_wins": candidate < control,
                }
            )
    successful_seeds = sum(item["gate_passed"] for item in seed_decisions)
    win_fraction = float(np.mean([item["candidate_wins"] for item in cells]))
    bootstrap = _bootstrap(differences, config.bootstrap_samples, 80_803)
    checks = {
        "aggregate_screen_gate_passes": bool(aggregate["gate_passed"]),
        "at_least_four_of_five_seeds_pass": successful_seeds
        >= config.required_successful_seeds,
        "candidate_wins_at_least_0_80_of_task_seed_cells": win_fraction
        >= config.required_task_seed_win_fraction,
        "paired_mse_difference_bootstrap_lower_bound_is_positive": bootstrap["lower_95"]
        > 0.0,
    }
    return {
        "strongest_memory_matched_control": strongest,
        "aggregate_screen_decision": aggregate,
        "successful_seeds": successful_seeds,
        "required_successful_seeds": config.required_successful_seeds,
        "task_seed_win_fraction": win_fraction,
        "task_seed_cells": cells,
        "paired_control_minus_candidate_mse_bootstrap": bootstrap,
        "seed_decisions": seed_decisions,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }


def _spearman(x: list[float], y: list[float]) -> float:
    def ranks(values: list[float]) -> np.ndarray:
        order = np.argsort(values)
        result = np.empty(len(values), dtype=np.float64)
        result[order] = np.arange(len(values), dtype=np.float64)
        return result

    left = ranks(x)
    right = ranks(y)
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def boundary_decision(
    primary_rows: list[dict[str, object]],
    primary_initial: dict[str, dict[str, object]],
    boundary_rows: list[dict[str, object]],
    boundary_initial: dict[str, dict[str, object]],
    strongest_control: str,
    config: Config,
) -> dict[str, object]:
    coverage = {}
    for task in list(task_suite().values()) + list(boundary_suite().values()):
        coverage[task.name] = float(task_audit(task)["mean_top_rank_two_energy"])
    task_rows = []
    task_level_coverage = []
    task_level_retention = []
    all_rows = primary_rows + boundary_rows
    all_initial = {**primary_initial, **boundary_initial}
    for task in (*task_suite(), *boundary_suite()):
        available_seeds = sorted(
            {row["seed"] for row in all_rows if row["task"] == task}
        )
        for seed in available_seeds:
            initial = _initial_metric(
                all_initial, "ood_mean_mse", task=task, seed=seed
            )
            full = _mean_metric(
                all_rows, "full_trace", "ood_mean_mse", task=task, seed=seed
            )
            candidate = _mean_metric(
                all_rows, "lowrank_svd", "ood_mean_mse", task=task, seed=seed
            )
            control = _mean_metric(
                all_rows, strongest_control, "ood_mean_mse", task=task, seed=seed
            )
            retention = _gain_retention(initial, candidate, full)
            task_rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "top_rank_two_energy": coverage[task],
                    "candidate_full_gain_retention": retention,
                    "candidate_relative_gain_over_control": _relative_gain(
                        candidate, control
                    ),
                }
            )
        task_level_coverage.append(coverage[task])
        task_level_retention.append(
            float(
                np.mean(
                    [
                        item["candidate_full_gain_retention"]
                        for item in task_rows
                        if item["task"] == task
                    ]
                )
            )
        )
    boundary_retention = float(
        np.mean(
            [
                item["candidate_full_gain_retention"]
                for item in task_rows
                if item["task"] in boundary_suite()
            ]
        )
    )
    # Correlate at the task-family level. Repeating the same spectral coverage once
    # per seed would create tied x-values and a misleading rank statistic.
    spearman = _spearman(task_level_coverage, task_level_retention)
    broad_success = boundary_retention >= config.minimum_boundary_retention
    predictable_boundary = spearman >= config.minimum_coverage_retention_spearman
    checks = {
        "boundary_is_broad_success_or_coverage_predicts_retention": broad_success
        or predictable_boundary,
        "candidate_avoids_catastrophic_boundary_loss": min(
            item["candidate_relative_gain_over_control"]
            for item in task_rows
            if item["task"] in boundary_suite()
        )
        >= -0.20,
    }
    return {
        "top_rank_two_energy_by_task": coverage,
        "task_seed_rows": task_rows,
        "boundary_mean_full_gain_retention": boundary_retention,
        "coverage_retention_spearman": spearman,
        "broad_boundary_success": broad_success,
        "predictable_rank_boundary": predictable_boundary,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }


def run(output_dir: Path, device: torch.device, config: Config) -> dict[str, object]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=False)
    progress_path = output_dir / "progress.jsonl"
    premise = premise_summary()
    _write_json(output_dir / "premise_summary.json", premise)
    _append_jsonl(progress_path, {"stage": "premise", "gate_passed": premise["gate_passed"]})
    if not premise["gate_passed"]:
        summary = {
            "project": "N08 Low-Rank Eligibility Traces for Delayed Outcomes",
            "version": "1.0",
            "premise_passed": False,
            "screen_ran": False,
            "screen_passed": False,
            "confirmation_ran": False,
            "boundary_ran": False,
            "gate_passed": False,
            "decision": "stop_n08_after_reference_premise",
            "runtime_seconds": time.perf_counter() - started,
            "config": asdict(config),
        }
        _write_json(output_dir / "cascade_summary.json", summary)
        return summary

    screen_rows, screen_initial = stage_runs(
        task_suite(),
        methods=METHODS,
        seeds=(SEEDS[0],),
        steps=config.screen_steps,
        eval_examples=config.screen_eval_examples,
        stage="screen",
        config=config,
        device=device,
        progress_path=progress_path,
    )
    screen_gate = screen_decision(screen_rows, screen_initial, config)
    screen_summary = {
        "project": "N08 Low-Rank Eligibility Traces for Delayed Outcomes",
        "version": "1.0",
        "stage": "one-seed streaming low-rank trace screen",
        "main_question": premise["main_question"],
        "specific_interrogation": (
            "Test whether a streaming rank-two SVD trace preserves delayed terminal-outcome "
            "learning better than equally sized recent-event, reservoir-event, replay, and "
            "multi-timescale factor controls while using about one sixth of full-trace state."
        ),
        "initial_evaluations": screen_initial,
        "rows": screen_rows,
        "gate": screen_gate,
        "gate_passed": screen_gate["gate_passed"],
    }
    _write_json(output_dir / "screen_summary.json", screen_summary)
    if not screen_gate["gate_passed"]:
        summary = {
            "project": "N08 Low-Rank Eligibility Traces for Delayed Outcomes",
            "version": "1.0",
            "premise_passed": True,
            "screen_ran": True,
            "screen_passed": False,
            "confirmation_ran": False,
            "boundary_ran": False,
            "gate_passed": False,
            "decision": "stop_n08_after_learned_screen",
            "runtime_seconds": time.perf_counter() - started,
            "config": asdict(config),
        }
        _write_json(output_dir / "cascade_summary.json", summary)
        return summary

    confirmation_rows, confirmation_initial = stage_runs(
        task_suite(),
        methods=METHODS,
        seeds=SEEDS,
        steps=config.confirmation_steps,
        eval_examples=config.confirmation_eval_examples,
        stage="confirmation",
        config=config,
        device=device,
        progress_path=progress_path,
    )
    confirmation_gate = confirmation_decision(
        confirmation_rows, confirmation_initial, config
    )
    confirmation_summary = {
        "project": "N08 Low-Rank Eligibility Traces for Delayed Outcomes",
        "version": "1.0",
        "stage": "five-seed delayed-outcome confirmation",
        "initial_evaluations": confirmation_initial,
        "rows": confirmation_rows,
        "gate": confirmation_gate,
        "gate_passed": confirmation_gate["gate_passed"],
    }
    _write_json(output_dir / "confirmation_summary.json", confirmation_summary)
    if not confirmation_gate["gate_passed"]:
        summary = {
            "project": "N08 Low-Rank Eligibility Traces for Delayed Outcomes",
            "version": "1.0",
            "premise_passed": True,
            "screen_ran": True,
            "screen_passed": True,
            "confirmation_ran": True,
            "confirmation_passed": False,
            "boundary_ran": False,
            "gate_passed": False,
            "decision": "stop_n08_after_five_seed_confirmation",
            "runtime_seconds": time.perf_counter() - started,
            "config": asdict(config),
        }
        _write_json(output_dir / "cascade_summary.json", summary)
        return summary

    strongest = confirmation_gate["strongest_memory_matched_control"]
    boundary_methods = ("full_trace", "lowrank_svd", strongest)
    boundary_rows, boundary_initial = stage_runs(
        boundary_suite(),
        methods=boundary_methods,
        seeds=BOUNDARY_SEEDS,
        steps=config.boundary_steps,
        eval_examples=config.confirmation_eval_examples,
        stage="rank_boundary",
        config=config,
        device=device,
        progress_path=progress_path,
    )
    boundary_gate = boundary_decision(
        confirmation_rows,
        confirmation_initial,
        boundary_rows,
        boundary_initial,
        strongest,
        config,
    )
    boundary_summary = {
        "project": "N08 Low-Rank Eligibility Traces for Delayed Outcomes",
        "version": "1.0",
        "stage": "three-seed high-rank boundary characterization",
        "strongest_memory_matched_control": strongest,
        "initial_evaluations": boundary_initial,
        "rows": boundary_rows,
        "gate": boundary_gate,
        "gate_passed": boundary_gate["gate_passed"],
    }
    _write_json(output_dir / "boundary_summary.json", boundary_summary)
    summary = {
        "project": "N08 Low-Rank Eligibility Traces for Delayed Outcomes",
        "version": "1.0",
        "premise_passed": True,
        "screen_ran": True,
        "screen_passed": True,
        "confirmation_ran": True,
        "confirmation_passed": True,
        "boundary_ran": True,
        "boundary_passed": boundary_gate["gate_passed"],
        "gate_passed": boundary_gate["gate_passed"],
        "decision": (
            "advance_n08_to_transformer_adapter_benchmarks"
            if boundary_gate["gate_passed"]
            else "stop_n08_after_rank_boundary"
        ),
        "runtime_seconds": time.perf_counter() - started,
        "config": asdict(config),
    }
    _write_json(output_dir / "cascade_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--screen-steps", type=int, default=Config.screen_steps)
    parser.add_argument(
        "--confirmation-steps", type=int, default=Config.confirmation_steps
    )
    parser.add_argument("--boundary-steps", type=int, default=Config.boundary_steps)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    config = Config(
        screen_steps=args.screen_steps,
        confirmation_steps=args.confirmation_steps,
        boundary_steps=args.boundary_steps,
        batch_size=args.batch_size,
    )
    summary = run(args.output_dir, device, config)
    print(
        json.dumps(
            {
                key: summary.get(key)
                for key in (
                    "decision",
                    "premise_passed",
                    "screen_passed",
                    "confirmation_ran",
                    "boundary_ran",
                    "gate_passed",
                    "runtime_seconds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
