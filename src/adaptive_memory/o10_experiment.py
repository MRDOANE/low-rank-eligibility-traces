from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .o10_layerwise import (
    MATCHED_CONTROLS,
    METHODS,
    REPLAY_CONTROLS,
    predict,
    train_method,
)
from .o10_tasks import (
    LayerwiseTraceTask,
    full_state_floats,
    premise_summary,
    replay_slots,
    sample_batch,
    sketch_state_floats,
    task_suite,
    zero_student_scores,
)


SCREEN_SEED = 7
CONFIRMATION_SEEDS = (7, 19, 31, 43, 59)
LADDER_SEEDS = (7, 19, 31)
OOD_CONDITIONS = (
    "length_1x",
    "length_2x",
    "length_8x",
    "noise_shift_4x",
    "scale_shift_4x",
    "joint_8x",
)
HORIZON_CONDITIONS = (
    "iid_4x",
    "length_8x",
    "joint_8x",
)


@dataclass(frozen=True)
class Config:
    trace_rank: int = 8
    training_horizon_multiplier: int = 4
    screen_steps: int = 300
    confirmation_steps: int = 425
    ladder_steps: int = 375
    batch_size: int = 8
    screen_eval_examples: int = 256
    confirmation_eval_examples: int = 384
    learning_rate: float = 0.008
    weight_decay: float = 0.0001
    minimum_full_learning_fraction: float = 0.60
    minimum_full_gain_retention: float = 0.90
    minimum_control_gain: float = 0.10
    minimum_replay_gain_at_8x: float = 0.10
    minimum_gradient_cosine: float = 0.85
    maximum_cosine_deficit: float = 0.01
    minimum_layer_ablation_gain: float = 0.10
    maximum_task_deficit: float = 0.05
    required_screen_condition_wins: int = 4
    required_confirmation_seed_wins: int = 4
    required_confirmation_task_seed_wins: int = 8
    bootstrap_samples: int = 5000
    maximum_state_fraction: float = 0.30
    minimum_ladder_rank_wins: int = 2
    minimum_horizon_gain_spearman: float = 0.60


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def evaluation_batches(
    task: LayerwiseTraceTask,
    n: int,
    seed: int,
) -> dict[str, object]:
    return {
        "length_1x": sample_batch(task, n, seed),
        "length_2x": sample_batch(
            task, n, seed + 1, length=2 * task.train_length
        ),
        "iid_4x": sample_batch(
            task, n, seed + 2, length=4 * task.train_length
        ),
        "length_8x": sample_batch(
            task, n, seed + 3, length=8 * task.train_length
        ),
        "noise_shift_4x": sample_batch(
            task,
            n,
            seed + 4,
            length=4 * task.train_length,
            input_noise=0.10,
        ),
        "scale_shift_4x": sample_batch(
            task,
            n,
            seed + 5,
            length=4 * task.train_length,
            latent_scale=1.40,
        ),
        "joint_8x": sample_batch(
            task,
            n,
            seed + 6,
            length=8 * task.train_length,
            input_noise=0.10,
            latent_scale=1.40,
        ),
    }


def initial_evaluation(
    task: LayerwiseTraceTask,
    batches: dict[str, object],
) -> dict[str, object]:
    result = {}
    for name, batch in batches.items():
        prediction = zero_student_scores(task, batch.x).sum(axis=1) / np.sqrt(
            batch.x.shape[1]
        )
        result[name] = {
            "terminal_mse": float(np.mean((prediction - batch.target) ** 2)),
            "terminal_mae": float(np.mean(np.abs(prediction - batch.target))),
        }
    result["aggregate"] = {
        "mean_mse": float(np.mean([result[name]["terminal_mse"] for name in batches])),
        "ood_mean_mse": float(
            np.mean([result[name]["terminal_mse"] for name in OOD_CONDITIONS])
        ),
        "horizon_mean_mse": float(
            np.mean([result[name]["terminal_mse"] for name in HORIZON_CONDITIONS])
        ),
    }
    return result


def evaluate_weights(
    task: LayerwiseTraceTask,
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
        "horizon_mean_mse": float(
            np.mean([result[name]["terminal_mse"] for name in HORIZON_CONDITIONS])
        ),
    }
    return result


def run_method(
    task: LayerwiseTraceTask,
    batches: dict[str, object],
    *,
    method: str,
    seed: int,
    rank: int,
    steps: int,
    stage: str,
    config: Config,
    device: torch.device,
    checkpoint_root: Path,
) -> dict[str, object]:
    trained = train_method(
        task,
        method=method,
        seed=seed,
        rank=rank,
        training_length=config.training_horizon_multiplier * task.train_length,
        steps=steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=device,
    )
    checkpoint = checkpoint_root / stage / task.name / f"seed_{seed}" / (
        f"{method}_rank_{rank}.pt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "project": "O10 Layerwise Eligibility Traces",
            "stage": stage,
            "task": task.name,
            "method": method,
            "seed": seed,
            "rank": rank,
            "weights": trained.weights,
        },
        checkpoint,
    )
    return {
        "stage": stage,
        "task": task.name,
        "latent_rank": task.latent_rank,
        "method": method,
        "seed": seed,
        "trace_rank": rank,
        "training_sequence_length": config.training_horizon_multiplier
        * task.train_length,
        "checkpoint": str(checkpoint.relative_to(checkpoint_root.parent)),
        "training": {
            "runtime_seconds": trained.runtime_seconds,
            "parameters": trained.parameters,
            "optimizer_steps": trained.optimizer_steps,
            "allocated_state_floats_per_episode": trained.allocated_state_floats,
            "used_state_floats_per_episode": trained.used_state_floats,
            "allocated_state_bytes_fp32_per_episode": 4
            * trained.allocated_state_floats,
            "mean_gradient_cosine": trained.mean_gradient_cosine,
            "final_gradient_cosine": trained.final_gradient_cosine,
            "final_train_mse": trained.final_train_mse,
            "trajectory": trained.trajectory,
        },
        "evaluation": evaluate_weights(task, trained.weights, batches, device),
    }


def stage_runs(
    tasks: dict[str, LayerwiseTraceTask],
    *,
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
    rank: int,
    steps: int,
    eval_examples: int,
    stage: str,
    config: Config,
    device: torch.device,
    output_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    rows = []
    initial = {}
    progress_path = output_dir / "progress.jsonl"
    for task_index, task in enumerate(tasks.values()):
        for seed in seeds:
            eval_seed = 1_000_000 + task_index * 50_000 + seed
            batches = evaluation_batches(task, eval_examples, eval_seed)
            initial[f"{task.name}:{seed}"] = initial_evaluation(task, batches)
            for method in methods:
                row = run_method(
                    task,
                    batches,
                    method=method,
                    seed=seed,
                    rank=rank,
                    steps=steps,
                    stage=stage,
                    config=config,
                    device=device,
                    checkpoint_root=output_dir / "checkpoints",
                )
                rows.append(row)
                _append_jsonl(
                    progress_path,
                    {
                        "event": "run_complete",
                        "stage": stage,
                        "task": task.name,
                        "seed": seed,
                        "method": method,
                        "rank": rank,
                        "horizon_mean_mse": row["evaluation"]["aggregate"][
                            "horizon_mean_mse"
                        ],
                        "length_8x_mse": row["evaluation"]["length_8x"][
                            "terminal_mse"
                        ],
                        "mean_gradient_cosine": row["training"][
                            "mean_gradient_cosine"
                        ],
                    },
                )
    return rows, initial


def _rows_for(
    rows: list[dict[str, object]],
    method: str,
    *,
    task: str | None = None,
    seed: int | None = None,
    rank: int | None = None,
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row["method"] == method
        and (task is None or row["task"] == task)
        and (seed is None or row["seed"] == seed)
        and (rank is None or row["trace_rank"] == rank)
    ]


def _metric(
    rows: list[dict[str, object]],
    method: str,
    *,
    aggregate: str | None = None,
    condition: str | None = None,
    task: str | None = None,
    seed: int | None = None,
    rank: int | None = None,
) -> float:
    selected = _rows_for(rows, method, task=task, seed=seed, rank=rank)
    if not selected:
        raise ValueError(f"empty metric group for {method}")
    if aggregate is not None:
        values = [float(row["evaluation"]["aggregate"][aggregate]) for row in selected]
    elif condition is not None:
        values = [float(row["evaluation"][condition]["terminal_mse"]) for row in selected]
    else:
        raise ValueError("aggregate or condition is required")
    return float(np.mean(values))


def _training_metric(
    rows: list[dict[str, object]],
    method: str,
    field: str,
    *,
    task: str | None = None,
    seed: int | None = None,
    rank: int | None = None,
) -> float:
    selected = _rows_for(rows, method, task=task, seed=seed, rank=rank)
    if not selected:
        raise ValueError(f"empty training group for {method}")
    return float(np.mean([float(row["training"][field]) for row in selected]))


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
    return (initial - candidate) / max(initial - full, 1e-12)


def screen_decision(
    rows: list[dict[str, object]],
    initial: dict[str, dict[str, object]],
    config: Config,
    *,
    rank: int | None = None,
) -> dict[str, object]:
    control_mse = {
        method: _metric(rows, method, aggregate="horizon_mean_mse", rank=rank)
        for method in MATCHED_CONTROLS
        if _rows_for(rows, method, rank=rank)
    }
    strongest = min(control_mse, key=control_mse.get)
    replay_mse = {
        method: _metric(rows, method, aggregate="horizon_mean_mse", rank=rank)
        for method in REPLAY_CONTROLS
        if _rows_for(rows, method, rank=rank)
    }
    strongest_replay = min(replay_mse, key=replay_mse.get)
    candidate = _metric(
        rows, "layerwise_sketch", aggregate="horizon_mean_mse", rank=rank
    )
    full = _metric(rows, "full_trace", aggregate="horizon_mean_mse", rank=rank)
    initial_mse = _initial_metric(initial, "horizon_mean_mse")
    control_gain = _relative_gain(candidate, control_mse[strongest])
    candidate_8x = _metric(
        rows, "layerwise_sketch", condition="length_8x", rank=rank
    )
    replay_8x = _metric(rows, strongest_replay, condition="length_8x", rank=rank)
    candidate_cosine = _training_metric(
        rows, "layerwise_sketch", "mean_gradient_cosine", rank=rank
    )
    control_cosines = {
        method: _training_metric(rows, method, "mean_gradient_cosine", rank=rank)
        for method in MATCHED_CONTROLS
        if _rows_for(rows, method, rank=rank)
    }
    task_results = {}
    task_condition_wins = 0
    for task in task_suite():
        candidate_task = _metric(
            rows,
            "layerwise_sketch",
            aggregate="horizon_mean_mse",
            task=task,
            rank=rank,
        )
        control_task = _metric(
            rows,
            strongest,
            aggregate="horizon_mean_mse",
            task=task,
            rank=rank,
        )
        task_results[task] = {
            "candidate_horizon_mse": candidate_task,
            "control_horizon_mse": control_task,
            "relative_gain": _relative_gain(candidate_task, control_task),
        }
        for condition in HORIZON_CONDITIONS:
            candidate_condition = _metric(
                rows,
                "layerwise_sketch",
                condition=condition,
                task=task,
                rank=rank,
            )
            control_condition = _metric(
                rows,
                strongest,
                condition=condition,
                task=task,
                rank=rank,
            )
            task_condition_wins += int(candidate_condition < control_condition)
    candidate_state = int(
        _rows_for(rows, "layerwise_sketch", rank=rank)[0]["training"][
            "allocated_state_floats_per_episode"
        ]
    )
    full_state = int(
        _rows_for(rows, "full_trace", rank=rank)[0]["training"][
            "allocated_state_floats_per_episode"
        ]
    )
    matched_state = all(
        int(row["training"]["allocated_state_floats_per_episode"])
        == candidate_state
        for row in rows
        if row["method"] != "full_trace" and (rank is None or row["trace_rank"] == rank)
    )
    matched_parameters = all(
        len(
            {
                row["training"]["parameters"]
                for row in rows
                if row["task"] == task and (rank is None or row["trace_rank"] == rank)
            }
        )
        == 1
        for task in task_suite()
    )
    matched_steps = len(
        {
            row["training"]["optimizer_steps"]
            for row in rows
            if rank is None or row["trace_rank"] == rank
        }
    ) == 1
    pooled = _metric(
        rows, "pooled_layer_sketch", aggregate="horizon_mean_mse", rank=rank
    )
    permuted = _metric(
        rows, "permuted_layer_sketch", aggregate="horizon_mean_mse", rank=rank
    )
    checks = {
        "full_trace_learns_at_least_0_60_of_available_error": _relative_gain(
            full, initial_mse
        )
        >= config.minimum_full_learning_fraction,
        "candidate_retains_at_least_0_90_of_full_trace_gain": _gain_retention(
            initial_mse, candidate, full
        )
        >= config.minimum_full_gain_retention,
        "candidate_beats_strongest_matched_control_by_0_10": control_gain
        >= config.minimum_control_gain,
        "candidate_beats_strongest_replay_at_8x_by_0_10": _relative_gain(
            candidate_8x, replay_8x
        )
        >= config.minimum_replay_gain_at_8x,
        "candidate_wins_four_of_six_task_conditions": task_condition_wins
        >= config.required_screen_condition_wins,
        "neither_task_is_worse_by_more_than_0_05": min(
            item["relative_gain"] for item in task_results.values()
        )
        >= -config.maximum_task_deficit,
        "candidate_gradient_cosine_is_at_least_0_85": candidate_cosine
        >= config.minimum_gradient_cosine,
        "candidate_cosine_is_within_0_01_of_best_control": candidate_cosine
        >= max(control_cosines.values()) - config.maximum_cosine_deficit,
        "candidate_beats_pooled_layer_ablation_by_0_10": _relative_gain(
            candidate, pooled
        )
        >= config.minimum_layer_ablation_gain,
        "candidate_beats_permuted_layer_ablation_by_0_10": _relative_gain(
            candidate, permuted
        )
        >= config.minimum_layer_ablation_gain,
        "candidate_uses_at_most_0_30_of_full_trace_state": candidate_state / full_state
        <= config.maximum_state_fraction,
        "all_matched_methods_have_exact_allocated_state_budget": matched_state,
        "all_methods_have_matched_parameters": matched_parameters,
        "all_methods_have_matched_optimizer_steps": matched_steps,
    }
    return {
        "strongest_matched_control": strongest,
        "strongest_replay_control": strongest_replay,
        "control_horizon_mse": control_mse,
        "replay_horizon_mse": replay_mse,
        "initial_horizon_mse": initial_mse,
        "full_trace_horizon_mse": full,
        "candidate_horizon_mse": candidate,
        "candidate_relative_gain": control_gain,
        "candidate_full_gain_retention": _gain_retention(
            initial_mse, candidate, full
        ),
        "candidate_length_8x_mse": candidate_8x,
        "replay_length_8x_mse": replay_8x,
        "candidate_length_8x_relative_gain": _relative_gain(
            candidate_8x, replay_8x
        ),
        "candidate_mean_gradient_cosine": candidate_cosine,
        "control_mean_gradient_cosines": control_cosines,
        "task_condition_wins": task_condition_wins,
        "task_condition_count": len(task_suite()) * len(HORIZON_CONDITIONS),
        "task_results": task_results,
        "candidate_state_floats": candidate_state,
        "full_trace_state_floats": full_state,
        "candidate_state_fraction": candidate_state / full_state,
        "matched_replay_slots": replay_slots(next(iter(task_suite().values())), rank or config.trace_rank),
        "checks": checks,
        "gate_passed": all(checks.values()),
    }


def _paired_bootstrap(
    differences: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[indices].mean(axis=1)
    return {
        "mean_control_minus_candidate": float(np.mean(differences)),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
    }


def confirmation_decision(
    rows: list[dict[str, object]],
    initial: dict[str, dict[str, object]],
    config: Config,
) -> dict[str, object]:
    aggregate = screen_decision(rows, initial, config)
    strongest = aggregate["strongest_matched_control"]
    seed_results = {}
    differences = []
    seed_wins = 0
    task_seed_wins = 0
    eight_x_seed_wins = 0
    for seed in CONFIRMATION_SEEDS:
        candidate_seed = _metric(
            rows,
            "layerwise_sketch",
            aggregate="horizon_mean_mse",
            seed=seed,
        )
        control_seed = _metric(
            rows, strongest, aggregate="horizon_mean_mse", seed=seed
        )
        gain = _relative_gain(candidate_seed, control_seed)
        seed_wins += int(gain >= config.minimum_control_gain)
        candidate_8x = _metric(
            rows, "layerwise_sketch", condition="length_8x", seed=seed
        )
        replay_8x = min(
            _metric(rows, method, condition="length_8x", seed=seed)
            for method in REPLAY_CONTROLS
        )
        eight_x_seed_wins += int(
            _relative_gain(candidate_8x, replay_8x)
            >= config.minimum_replay_gain_at_8x
        )
        seed_results[str(seed)] = {
            "candidate_horizon_mse": candidate_seed,
            "control_horizon_mse": control_seed,
            "relative_gain": gain,
            "candidate_length_8x_mse": candidate_8x,
            "replay_length_8x_mse": replay_8x,
        }
        for task in task_suite():
            candidate_task = _metric(
                rows,
                "layerwise_sketch",
                aggregate="horizon_mean_mse",
                seed=seed,
                task=task,
            )
            control_task = _metric(
                rows,
                strongest,
                aggregate="horizon_mean_mse",
                seed=seed,
                task=task,
            )
            task_seed_wins += int(candidate_task < control_task)
            differences.append(control_task - candidate_task)
    bootstrap = _paired_bootstrap(
        np.asarray(differences), config.bootstrap_samples, 10_010_501
    )
    checks = {
        "aggregate_screen_gate_replicates": aggregate["gate_passed"],
        "candidate_wins_four_of_five_seeds_by_0_10": seed_wins
        >= config.required_confirmation_seed_wins,
        "candidate_wins_eight_of_ten_task_seed_cells": task_seed_wins
        >= config.required_confirmation_task_seed_wins,
        "candidate_beats_replay_at_8x_in_four_of_five_seeds": eight_x_seed_wins
        >= config.required_confirmation_seed_wins,
        "paired_bootstrap_lower_bound_excludes_zero": bootstrap["lower_95"] > 0.0,
    }
    return {
        "aggregate": aggregate,
        "seed_results": seed_results,
        "seed_wins": seed_wins,
        "task_seed_wins": task_seed_wins,
        "eight_x_seed_wins": eight_x_seed_wins,
        "paired_bootstrap": bootstrap,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }


def _rankdata(values: list[float]) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])


def ladder_decision(
    rows: list[dict[str, object]],
    initial_by_rank: dict[int, dict[str, dict[str, object]]],
    strongest_control: str,
    config: Config,
) -> dict[str, object]:
    rank_results = {}
    rank_wins = 0
    for rank in (4, 8, 16):
        candidate = _metric(
            rows, "layerwise_sketch", aggregate="horizon_mean_mse", rank=rank
        )
        control = _metric(
            rows, strongest_control, aggregate="horizon_mean_mse", rank=rank
        )
        full = _metric(rows, "full_trace", aggregate="horizon_mean_mse", rank=rank)
        initial = _initial_metric(initial_by_rank[rank], "horizon_mean_mse")
        gain = _relative_gain(candidate, control)
        rank_wins += int(gain >= config.minimum_control_gain)
        condition_gains = []
        for condition in ("length_2x", "iid_4x", "length_8x"):
            condition_gains.append(
                _relative_gain(
                    _metric(
                        rows,
                        "layerwise_sketch",
                        condition=condition,
                        rank=rank,
                    ),
                    _metric(
                        rows,
                        strongest_control,
                        condition=condition,
                        rank=rank,
                    ),
                )
            )
        rank_results[str(rank)] = {
            "candidate_horizon_mse": candidate,
            "control_horizon_mse": control,
            "full_trace_horizon_mse": full,
            "relative_gain": gain,
            "full_gain_retention": _gain_retention(initial, candidate, full),
            "condition_gains_2x_4x_8x": condition_gains,
            "horizon_gain_spearman": _spearman([2.0, 4.0, 8.0], condition_gains),
            "candidate_state_floats": sketch_state_floats(
                next(iter(task_suite().values())), rank
            ),
            "full_state_floats": full_state_floats(
                next(iter(task_suite().values()))
            ),
        }
    checks = {
        "candidate_wins_at_two_of_three_state_budgets": rank_wins
        >= config.minimum_ladder_rank_wins,
        "rank_eight_confirmation_is_positive": rank_results["8"]["relative_gain"]
        >= config.minimum_control_gain,
        "rank_eight_gain_grows_with_horizon": rank_results["8"][
            "horizon_gain_spearman"
        ]
        >= config.minimum_horizon_gain_spearman,
        "rank_eight_retains_0_90_of_full_trace_gain": rank_results["8"][
            "full_gain_retention"
        ]
        >= config.minimum_full_gain_retention,
    }
    return {
        "strongest_confirmation_control": strongest_control,
        "rank_results": rank_results,
        "rank_wins": rank_wins,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }


def run(output_dir: Path, device: torch.device, config: Config) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    premise = premise_summary(config.trace_rank)
    _write_json(output_dir / "premise_summary.json", premise)
    _append_jsonl(
        output_dir / "progress.jsonl",
        {"event": "premise_complete", "premise_passed": premise["premise_passed"]},
    )
    if not premise["premise_passed"]:
        cascade = {
            "version": "1.0",
            "project": "O10 Layerwise Eligibility Traces",
            "config": asdict(config),
            "premise_passed": False,
            "screen_ran": False,
            "confirmation_ran": False,
            "ladder_ran": False,
            "gate_passed": False,
            "decision": "stop_o10_after_premise",
            "next_project_if_no_auspicious_signal": "O09",
            "runtime_seconds": time.perf_counter() - started,
        }
        _write_json(output_dir / "cascade_summary.json", cascade)
        return cascade

    screen_rows, screen_initial = stage_runs(
        task_suite(),
        methods=METHODS,
        seeds=(SCREEN_SEED,),
        rank=config.trace_rank,
        steps=config.screen_steps,
        eval_examples=config.screen_eval_examples,
        stage="screen",
        config=config,
        device=device,
        output_dir=output_dir,
    )
    screen_gate = screen_decision(screen_rows, screen_initial, config)
    screen = {
        "version": "1.0",
        "project": "O10 Layerwise Eligibility Traces",
        "stage": "one-seed horizon-scaling screen",
        "main_question": premise["main_question"],
        "specific_interrogation": premise["specific_interrogation"],
        "initial_evaluations": screen_initial,
        "rows": screen_rows,
        "gate": screen_gate,
        "gate_passed": screen_gate["gate_passed"],
    }
    _write_json(output_dir / "screen_summary.json", screen)
    _append_jsonl(
        output_dir / "progress.jsonl",
        {"event": "screen_complete", **screen_gate},
    )
    if not screen_gate["gate_passed"]:
        cascade = {
            "version": "1.0",
            "project": "O10 Layerwise Eligibility Traces",
            "config": asdict(config),
            "premise_passed": True,
            "screen_ran": True,
            "screen_passed": False,
            "confirmation_ran": False,
            "ladder_ran": False,
            "gate_passed": False,
            "decision": "stop_o10_after_screen",
            "next_project_if_no_auspicious_signal": "O09",
            "screen_gate": screen_gate,
            "runtime_seconds": time.perf_counter() - started,
        }
        _write_json(output_dir / "cascade_summary.json", cascade)
        return cascade

    confirmation_rows, confirmation_initial = stage_runs(
        task_suite(),
        methods=METHODS,
        seeds=CONFIRMATION_SEEDS,
        rank=config.trace_rank,
        steps=config.confirmation_steps,
        eval_examples=config.confirmation_eval_examples,
        stage="confirmation",
        config=config,
        device=device,
        output_dir=output_dir,
    )
    confirmation_gate = confirmation_decision(
        confirmation_rows, confirmation_initial, config
    )
    confirmation = {
        "version": "1.0",
        "project": "O10 Layerwise Eligibility Traces",
        "stage": "five-seed confirmation",
        "initial_evaluations": confirmation_initial,
        "rows": confirmation_rows,
        "gate": confirmation_gate,
        "gate_passed": confirmation_gate["gate_passed"],
    }
    _write_json(output_dir / "confirmation_summary.json", confirmation)
    _append_jsonl(
        output_dir / "progress.jsonl",
        {"event": "confirmation_complete", **confirmation_gate},
    )
    if not confirmation_gate["gate_passed"]:
        cascade = {
            "version": "1.0",
            "project": "O10 Layerwise Eligibility Traces",
            "config": asdict(config),
            "premise_passed": True,
            "screen_ran": True,
            "screen_passed": True,
            "confirmation_ran": True,
            "confirmation_passed": False,
            "ladder_ran": False,
            "gate_passed": False,
            "decision": "stop_o10_after_confirmation",
            "next_project_if_no_auspicious_signal": "O09",
            "screen_gate": screen_gate,
            "confirmation_gate": confirmation_gate,
            "runtime_seconds": time.perf_counter() - started,
        }
        _write_json(output_dir / "cascade_summary.json", cascade)
        return cascade

    strongest = confirmation_gate["aggregate"]["strongest_matched_control"]
    ladder_rows = []
    initial_by_rank = {}
    for rank in (4, 8, 16):
        rank_config = replace(config, trace_rank=rank)
        rows, initial = stage_runs(
            task_suite(),
            methods=("full_trace", "layerwise_sketch", strongest),
            seeds=LADDER_SEEDS,
            rank=rank,
            steps=config.ladder_steps,
            eval_examples=config.confirmation_eval_examples,
            stage=f"state_ladder_rank_{rank}",
            config=rank_config,
            device=device,
            output_dir=output_dir,
        )
        ladder_rows.extend(rows)
        initial_by_rank[rank] = initial
    ladder_gate = ladder_decision(
        ladder_rows, initial_by_rank, strongest, config
    )
    ladder = {
        "version": "1.0",
        "project": "O10 Layerwise Eligibility Traces",
        "stage": "three-budget horizon-scaling ladder",
        "rows": ladder_rows,
        "gate": ladder_gate,
        "gate_passed": ladder_gate["gate_passed"],
    }
    _write_json(output_dir / "state_ladder_summary.json", ladder)
    _append_jsonl(
        output_dir / "progress.jsonl",
        {"event": "state_ladder_complete", **ladder_gate},
    )
    cascade = {
        "version": "1.0",
        "project": "O10 Layerwise Eligibility Traces",
        "config": asdict(config),
        "premise_passed": True,
        "screen_ran": True,
        "screen_passed": True,
        "confirmation_ran": True,
        "confirmation_passed": True,
        "ladder_ran": True,
        "ladder_passed": ladder_gate["gate_passed"],
        "gate_passed": ladder_gate["gate_passed"],
        "decision": (
            "continue_o10_to_transformer_delayed_feedback"
            if ladder_gate["gate_passed"]
            else "stop_o10_after_state_ladder"
        ),
        "next_project_if_no_auspicious_signal": "O09",
        "screen_gate": screen_gate,
        "confirmation_gate": confirmation_gate,
        "ladder_gate": ladder_gate,
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(output_dir / "cascade_summary.json", cascade)
    return cascade


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the O10 layerwise-trace cascade")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/o10_layerwise_traces_v10"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--screen-steps", type=int, default=Config.screen_steps)
    args = parser.parse_args()
    config = replace(Config(), screen_steps=args.screen_steps)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    result = run(args.output_dir, device, config)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "gate_passed": result["gate_passed"],
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
