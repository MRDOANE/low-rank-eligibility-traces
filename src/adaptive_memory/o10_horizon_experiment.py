from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .o10_layerwise import (
    LayerwiseAdapterModel,
    approximate_trace,
    full_trace,
    global_rank_for_budget,
    method_state,
    predict,
    train_method,
)
from .o10_tasks import (
    LayerwiseTraceTask,
    full_state_floats,
    make_task,
    replay_slots,
    sample_batch,
    sketch_state_floats,
    stable_seed,
    task_audit,
    trace_cosine,
    zero_student_scores,
)


PROJECT = "O10 Layerwise Eligibility Traces"
VERSION = "2.0"
BASE_LENGTH = 128
PRIMARY_HORIZONS = (1, 4, 8)
EXTENSION_HORIZONS = (16, 32)
LATENT_RANKS = (2, 4)
PRIMARY_RANK = 8
RESCUE_RANK = 12
CALIBRATION_REPLICATES = (61, 73)
CONFIRMATION_REPLICATES = (101, 211, 307, 401, 503)
PRIMARY_METHODS = (
    "full_trace",
    "layerwise_sketch",
    "global_sketch",
    "reservoir_replay",
    "stratified_replay",
    "gradient_replay",
    "multi_timescale",
)
EXTENSION_METHODS = PRIMARY_METHODS[:-1]
REPLAY_METHODS = (
    "reservoir_replay",
    "stratified_replay",
    "gradient_replay",
)
EVAL_CONDITIONS = ("iid", "noise_shift", "scale_shift", "joint_shift")


@dataclass(frozen=True)
class Config:
    learning_rates: tuple[float, ...] = (0.002, 0.004, 0.008, 0.016, 0.032)
    calibration_steps: int = 200
    primary_steps: int = 550
    rescue_steps: int = 550
    extension_steps: int = 550
    calibration_batch_size: int = 8
    primary_batch_size: int = 8
    calibration_eval_examples: int = 128
    primary_eval_examples: int = 256
    target_noise: float = 0.002
    weight_decay: float = 0.0001
    bootstrap_samples: int = 10_000
    minimum_full_learning_overall: float = 0.60
    minimum_full_learning_each_horizon: float = 0.45
    minimum_full_learning_each_cell: float = 0.20
    minimum_retention_mean: float = 0.95
    minimum_retention_lower_95: float = 0.90
    minimum_retention_each_horizon: float = 0.85
    minimum_long_replay_gain: float = 0.05
    minimum_horizon_interaction: float = 0.05
    global_noninferiority_margin: float = 0.03
    minimum_mean_cosine: float = 0.88
    minimum_horizon_cosine: float = 0.80
    maximum_primary_state_fraction: float = 0.30
    maximum_rescue_state_fraction: float = 0.45
    minimum_long_cell_win_fraction: float = 0.70
    minimum_layer_specific_gain: float = 0.03
    extension_retention_floor: float = 0.88
    extension_cosine_floor: float = 0.80
    extension_replay_gain_floor: float = -0.05
    rescue_retention_floor: float = 0.70
    rescue_cosine_floor: float = 0.65


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _lr_key(method: str, rank: int, horizon: int) -> str:
    return f"{method}:r{rank}:h{horizon}"


def make_v2_task(
    replicate: int,
    latent_rank: int,
    horizon: int,
    *,
    split: str,
) -> LayerwiseTraceTask:
    """Create paired horizons from one independently drawn teacher per replicate."""
    task_seed = stable_seed("o10-v2-teacher", split, replicate, latent_rank)
    base = make_task(
        latent_rank,
        train_length=BASE_LENGTH,
        sketch_rank=32,
        seed=task_seed,
    )
    return replace(
        base,
        name=f"{split}_rep{replicate}_latent{latent_rank:02d}_h{horizon:02d}",
        train_length=BASE_LENGTH * horizon,
    )


def evaluation_batches(
    task: LayerwiseTraceTask,
    n: int,
    seed: int,
    target_noise: float,
) -> dict[str, object]:
    return {
        "iid": sample_batch(task, n, seed, target_noise=target_noise),
        "noise_shift": sample_batch(
            task,
            n,
            seed + 1,
            input_noise=0.08,
            target_noise=target_noise,
        ),
        "scale_shift": sample_batch(
            task,
            n,
            seed + 2,
            latent_scale=1.30,
            target_noise=target_noise,
        ),
        "joint_shift": sample_batch(
            task,
            n,
            seed + 3,
            input_noise=0.08,
            latent_scale=1.30,
            target_noise=target_noise,
        ),
    }


def initial_evaluation(
    task: LayerwiseTraceTask, batches: dict[str, object]
) -> dict[str, object]:
    rows: dict[str, object] = {}
    for name, batch in batches.items():
        prediction = zero_student_scores(task, batch.x).sum(axis=1) / np.sqrt(
            batch.x.shape[1]
        )
        rows[name] = {
            "terminal_mse": float(np.mean((prediction - batch.target) ** 2)),
            "terminal_mae": float(np.mean(np.abs(prediction - batch.target))),
        }
    rows["aggregate"] = {
        "mean_mse": float(np.mean([rows[name]["terminal_mse"] for name in batches])),
        "worst_mse": float(np.max([rows[name]["terminal_mse"] for name in batches])),
    }
    return rows


def evaluate_weights(
    task: LayerwiseTraceTask,
    weights: tuple[np.ndarray, ...],
    batches: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    rows: dict[str, object] = {}
    for name, batch in batches.items():
        prediction = predict(task, weights, batch.x, device)
        rows[name] = {
            "terminal_mse": float(np.mean((prediction - batch.target) ** 2)),
            "terminal_mae": float(np.mean(np.abs(prediction - batch.target))),
            "prediction_mean": float(np.mean(prediction)),
            "target_mean": float(np.mean(batch.target)),
        }
    rows["aggregate"] = {
        "mean_mse": float(np.mean([rows[name]["terminal_mse"] for name in batches])),
        "worst_mse": float(np.max([rows[name]["terminal_mse"] for name in batches])),
    }
    return rows


def premise_summary_v2() -> dict[str, object]:
    audits = []
    matched_state = []
    global_cosines = []
    horizon_teacher_checks = []
    for latent_rank in LATENT_RANKS:
        short = make_v2_task(17, latent_rank, 1, split="premise")
        long = make_v2_task(17, latent_rank, 8, split="premise")
        horizon_teacher_checks.append(
            all(np.array_equal(a, b) for a, b in zip(short.base_matrices, long.base_matrices))
            and all(
                np.array_equal(a, b)
                for a, b in zip(short.teacher_updates, long.teacher_updates)
            )
        )
        audit = task_audit(short, rank=PRIMARY_RANK, n=16, seed=17 + latent_rank)
        audits.append(audit)
        batch = sample_batch(long, 8, 1_700 + latent_rank, target_noise=0.0)
        model = LayerwiseAdapterModel(long)
        x = torch.as_tensor(batch.x)
        _, left, right = model.forward_factors(x)
        exact = full_trace(left, right)
        generator = torch.Generator(device="cpu").manual_seed(9_010 + latent_rank)
        global_estimate = approximate_trace(
            "global_sketch", model, left, right, PRIMARY_RANK, generator
        )
        global_cosines.append(
            float(
                np.mean(
                    trace_cosine(
                        global_estimate.numpy(),
                        exact.numpy(),
                    )
                )
            )
        )
        candidate_state = sketch_state_floats(long, PRIMARY_RANK)
        method_states = {
            method: method_state(long, PRIMARY_RANK, method)
            for method in PRIMARY_METHODS
        }
        matched_state.append(
            all(
                allocated == candidate_state
                for method, (allocated, _) in method_states.items()
                if method != "full_trace"
            )
            and all(used <= allocated for allocated, used in method_states.values())
        )
    example = make_v2_task(17, 2, 1, split="premise")
    budget = sketch_state_floats(example, PRIMARY_RANK)
    global_rank = global_rank_for_budget(example, budget)
    checks = {
        "paired_horizons_share_the_same_teacher": all(horizon_teacher_checks),
        "manual_gradients_match_finite_differences": max(
            item["maximum_finite_difference_error"] for item in audits
        )
        <= 2e-5,
        "rank_eight_layerwise_sketch_is_finite_and_informative": min(
            item["sketch_mean_cosine"] for item in audits
        )
        >= 0.70,
        "global_same_budget_sketch_is_finite_and_informative": min(global_cosines)
        >= 0.50,
        "layer_identity_changes_the_gradient_direction": min(
            item["layer_identity_cosine_drop"] for item in audits
        )
        >= 0.10,
        "all_matched_methods_respect_the_candidate_budget": all(matched_state),
        "candidate_uses_at_most_thirty_percent_of_full_state": budget
        / full_state_floats(example)
        <= 0.30,
        "targets_are_finite_with_learning_headroom": all(
            item["all_targets_finite"] and item["initial_student_mse"] >= 1e-4
            for item in audits
        ),
    }
    return {
        "project": PROJECT,
        "version": VERSION,
        "main_question": (
            "When terminal feedback is delayed, do separate fixed-state layer traces become "
            "more useful than same-state event replay as the trained horizon grows?"
        ),
        "specific_interrogation": (
            "Train independently at 1x, 4x, and 8x delays after method-specific optimizer "
            "calibration; compare separate rank-eight traces with a stronger global sketch, "
            "three replay policies, a multi-timescale trace, and the exact trace."
        ),
        "old_v1_result_remains_frozen": True,
        "old_v1_rank_eight_gain_over_replay_at_evaluation_only_8x": 0.24104975696895029,
        "candidate_state_floats": budget,
        "candidate_state_fraction": budget / full_state_floats(example),
        "global_sketch_rank_at_same_budget": global_rank,
        "task_audits": audits,
        "global_sketch_cosines": global_cosines,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }


def run_one(
    task: LayerwiseTraceTask,
    batches: dict[str, object],
    *,
    replicate: int,
    horizon: int,
    method: str,
    rank: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    stage: str,
    config: Config,
    device: torch.device,
    checkpoint_root: Path | None = None,
) -> dict[str, object]:
    train_seed = stable_seed("o10-v2-train", stage, replicate, task.latent_rank, horizon)
    trained = train_method(
        task,
        method=method,
        seed=train_seed,
        rank=rank,
        training_length=task.train_length,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=config.weight_decay,
        device=device,
    )
    checkpoint = None
    if checkpoint_root is not None:
        checkpoint_path = (
            checkpoint_root
            / stage
            / f"rep_{replicate}"
            / f"latent_{task.latent_rank:02d}"
            / f"h_{horizon:02d}"
            / f"{method}_rank_{rank}.pt"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "project": PROJECT,
                "version": VERSION,
                "stage": stage,
                "replicate": replicate,
                "latent_rank": task.latent_rank,
                "horizon": horizon,
                "method": method,
                "trace_rank": rank,
                "learning_rate": learning_rate,
                "weights": trained.weights,
            },
            checkpoint_path,
        )
        checkpoint = str(checkpoint_path.relative_to(checkpoint_root.parent))
    return {
        "stage": stage,
        "task": task.name,
        "replicate": replicate,
        "latent_rank": task.latent_rank,
        "horizon_multiplier": horizon,
        "training_sequence_length": task.train_length,
        "method": method,
        "trace_rank": rank,
        "learning_rate": learning_rate,
        "checkpoint": checkpoint,
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


def _bootstrap_mean(
    values: list[float], samples: int, seed: int
) -> dict[str, float]:
    if not values:
        raise ValueError("bootstrap requires at least one replicate")
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data), size=(samples, len(data)))
    means = data[indices].mean(axis=1)
    return {
        "mean": float(data.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
    }


def calibrate(
    *,
    methods: tuple[str, ...],
    horizons: tuple[int, ...],
    rank: int,
    stage: str,
    output_dir: Path,
    progress_path: Path,
    config: Config,
    device: torch.device,
) -> tuple[dict[str, float], dict[int, str], dict[str, object]]:
    """Select each method's learning rate on independent teachers and examples."""
    rows: list[dict[str, object]] = []
    total = (
        len(methods)
        * len(horizons)
        * len(config.learning_rates)
        * len(CALIBRATION_REPLICATES)
        * len(LATENT_RANKS)
    )
    completed = 0
    durations: list[float] = []
    for horizon in horizons:
        for method in methods:
            for learning_rate in config.learning_rates:
                for replicate in CALIBRATION_REPLICATES:
                    for latent_rank in LATENT_RANKS:
                        task = make_v2_task(
                            replicate, latent_rank, horizon, split="development"
                        )
                        eval_seed = stable_seed(
                            "o10-v2-calibration-eval",
                            replicate,
                            latent_rank,
                            horizon,
                        )
                        batches = evaluation_batches(
                            task,
                            config.calibration_eval_examples,
                            eval_seed,
                            config.target_noise,
                        )
                        initial = initial_evaluation(task, batches)["aggregate"][
                            "mean_mse"
                        ]
                        row = run_one(
                            task,
                            batches,
                            replicate=replicate,
                            horizon=horizon,
                            method=method,
                            rank=rank,
                            steps=config.calibration_steps,
                            batch_size=config.calibration_batch_size,
                            learning_rate=learning_rate,
                            stage=stage,
                            config=config,
                            device=device,
                        )
                        objective = row["evaluation"]["aggregate"]["mean_mse"] / max(
                            float(initial), 1e-12
                        )
                        rows.append(
                            {
                                "method": method,
                                "horizon_multiplier": horizon,
                                "replicate": replicate,
                                "latent_rank": latent_rank,
                                "learning_rate": learning_rate,
                                "normalized_mse": float(objective),
                                "raw_mse": row["evaluation"]["aggregate"]["mean_mse"],
                                "runtime_seconds": row["training"]["runtime_seconds"],
                            }
                        )
                        completed += 1
                        durations.append(float(row["training"]["runtime_seconds"]))
                        eta = (total - completed) * float(np.mean(durations[-20:]))
                        _append_jsonl(
                            progress_path,
                            {
                                "event": "calibration_run_complete",
                                "stage": stage,
                                "completed": completed,
                                "total": total,
                                "eta_seconds": eta,
                                "method": method,
                                "horizon_multiplier": horizon,
                                "learning_rate": learning_rate,
                                "normalized_mse": float(objective),
                            },
                        )
    selected: dict[str, float] = {}
    score_table: dict[str, list[dict[str, float]]] = {}
    for horizon in horizons:
        for method in methods:
            candidates = []
            for learning_rate in config.learning_rates:
                values = [
                    float(row["normalized_mse"])
                    for row in rows
                    if row["method"] == method
                    and row["horizon_multiplier"] == horizon
                    and row["learning_rate"] == learning_rate
                ]
                finite = [value for value in values if np.isfinite(value)]
                score = float(np.median(finite)) if len(finite) == len(values) else math.inf
                candidates.append(
                    {
                        "learning_rate": learning_rate,
                        "median_normalized_mse": score,
                        "mean_normalized_mse": (
                            float(np.mean(finite)) if finite else math.inf
                        ),
                        "maximum_normalized_mse": (
                            float(np.max(finite)) if finite else math.inf
                        ),
                    }
                )
            best = min(
                candidates,
                key=lambda item: (
                    item["median_normalized_mse"],
                    item["mean_normalized_mse"],
                    item["learning_rate"],
                ),
            )
            selected[_lr_key(method, rank, horizon)] = float(best["learning_rate"])
            score_table[_lr_key(method, rank, horizon)] = candidates
    replay_by_horizon = {}
    for horizon in horizons:
        replay_scores = {}
        for method in REPLAY_METHODS:
            if method not in methods:
                continue
            key = _lr_key(method, rank, horizon)
            selected_lr = selected[key]
            replay_scores[method] = float(
                np.median(
                    [
                        row["normalized_mse"]
                        for row in rows
                        if row["method"] == method
                        and row["horizon_multiplier"] == horizon
                        and row["learning_rate"] == selected_lr
                    ]
                )
            )
        replay_by_horizon[horizon] = min(replay_scores, key=replay_scores.get)
    checks = {
        "every_method_horizon_has_a_finite_selected_rate": all(
            np.isfinite(
                min(
                    entry["median_normalized_mse"]
                    for entry in score_table[key]
                )
            )
            for key in score_table
        ),
        "calibration_uses_two_independent_teacher_replicates": len(
            CALIBRATION_REPLICATES
        )
        == 2,
        "calibration_never_uses_confirmation_replicates": set(
            CALIBRATION_REPLICATES
        ).isdisjoint(CONFIRMATION_REPLICATES),
        "every_horizon_has_a_prospectively_selected_replay": set(
            replay_by_horizon
        )
        == set(horizons),
    }
    summary = {
        "project": PROJECT,
        "version": VERSION,
        "stage": stage,
        "rank": rank,
        "horizons": list(horizons),
        "learning_rate_grid": list(config.learning_rates),
        "selection_objective": (
            "median held-out aggregate MSE divided by zero-student MSE across two "
            "independent teacher draws and both latent-rank families"
        ),
        "selected_learning_rates": selected,
        "selected_replay_by_horizon": replay_by_horizon,
        "score_table": score_table,
        "rows": rows,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }
    _write_json(output_dir / f"{stage}_summary.json", summary)
    return selected, replay_by_horizon, summary


def run_stage(
    *,
    stage: str,
    replicates: tuple[int, ...],
    horizons: tuple[int, ...],
    methods: tuple[str, ...],
    rank: int,
    steps: int,
    eval_examples: int,
    selected_learning_rates: dict[str, float],
    config: Config,
    device: torch.device,
    output_dir: Path,
    progress_path: Path,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    initial: dict[str, dict[str, object]] = {}
    total = len(replicates) * len(LATENT_RANKS) * len(horizons) * len(methods)
    completed = 0
    durations: list[float] = []
    for replicate in replicates:
        for latent_rank in LATENT_RANKS:
            for horizon in horizons:
                task = make_v2_task(
                    replicate, latent_rank, horizon, split="confirmation"
                )
                eval_seed = stable_seed(
                    "o10-v2-confirmation-eval", replicate, latent_rank, horizon
                )
                batches = evaluation_batches(
                    task, eval_examples, eval_seed, config.target_noise
                )
                cell_key = f"rep{replicate}:latent{latent_rank}:h{horizon}"
                initial[cell_key] = initial_evaluation(task, batches)
                for method in methods:
                    learning_rate = selected_learning_rates[
                        _lr_key(method, rank, horizon)
                    ]
                    row = run_one(
                        task,
                        batches,
                        replicate=replicate,
                        horizon=horizon,
                        method=method,
                        rank=rank,
                        steps=steps,
                        batch_size=config.primary_batch_size,
                        learning_rate=learning_rate,
                        stage=stage,
                        config=config,
                        device=device,
                        checkpoint_root=output_dir / "checkpoints",
                    )
                    rows.append(row)
                    completed += 1
                    durations.append(float(row["training"]["runtime_seconds"]))
                    eta = (total - completed) * float(np.mean(durations[-20:]))
                    _append_jsonl(
                        progress_path,
                        {
                            "event": "run_complete",
                            "stage": stage,
                            "completed": completed,
                            "total": total,
                            "eta_seconds": eta,
                            "replicate": replicate,
                            "latent_rank": latent_rank,
                            "horizon_multiplier": horizon,
                            "method": method,
                            "trace_rank": rank,
                            "aggregate_mse": row["evaluation"]["aggregate"][
                                "mean_mse"
                            ],
                            "mean_gradient_cosine": row["training"][
                                "mean_gradient_cosine"
                            ],
                        },
                    )
    return rows, initial


def _row_map(rows: list[dict[str, object]]) -> dict[tuple[int, int, int, str], dict]:
    result = {}
    for row in rows:
        key = (
            int(row["replicate"]),
            int(row["latent_rank"]),
            int(row["horizon_multiplier"]),
            str(row["method"]),
        )
        if key in result:
            raise ValueError(f"duplicate experimental row: {key}")
        result[key] = row
    return result


def _initial_key(replicate: int, latent_rank: int, horizon: int) -> str:
    return f"rep{replicate}:latent{latent_rank}:h{horizon}"


def _gain_retention(initial: float, candidate: float, full: float) -> float:
    return (initial - candidate) / max(initial - full, 1e-12)


def _relative_gain(candidate: float, control: float) -> float:
    return (control - candidate) / max(abs(control), 1e-12)


def analyze(
    rows: list[dict[str, object]],
    initial: dict[str, dict[str, object]],
    *,
    replicates: tuple[int, ...],
    horizons: tuple[int, ...],
    rank: int,
    replay_by_horizon: dict[int, str],
    config: Config,
    maximum_state_fraction: float,
) -> dict[str, object]:
    lookup = _row_map(rows)
    cells = []
    for replicate in replicates:
        for latent_rank in LATENT_RANKS:
            for horizon in horizons:
                base_key = (replicate, latent_rank, horizon)
                full_row = lookup[base_key + ("full_trace",)]
                candidate_row = lookup[base_key + ("layerwise_sketch",)]
                global_row = lookup[base_key + ("global_sketch",)]
                replay_method = replay_by_horizon[horizon]
                replay_row = lookup[base_key + (replay_method,)]
                initial_mse = float(
                    initial[_initial_key(replicate, latent_rank, horizon)]["aggregate"][
                        "mean_mse"
                    ]
                )
                full_mse = float(full_row["evaluation"]["aggregate"]["mean_mse"])
                candidate_mse = float(
                    candidate_row["evaluation"]["aggregate"]["mean_mse"]
                )
                global_mse = float(global_row["evaluation"]["aggregate"]["mean_mse"])
                replay_mse = float(replay_row["evaluation"]["aggregate"]["mean_mse"])
                available = max(initial_mse - full_mse, 1e-12)
                cells.append(
                    {
                        "replicate": replicate,
                        "latent_rank": latent_rank,
                        "horizon_multiplier": horizon,
                        "sequence_length": BASE_LENGTH * horizon,
                        "selected_replay": replay_method,
                        "initial_mse": initial_mse,
                        "full_trace_mse": full_mse,
                        "candidate_mse": candidate_mse,
                        "global_sketch_mse": global_mse,
                        "replay_mse": replay_mse,
                        "full_learning_fraction": (initial_mse - full_mse)
                        / max(initial_mse, 1e-12),
                        "candidate_full_gain_retention": _gain_retention(
                            initial_mse, candidate_mse, full_mse
                        ),
                        "candidate_relative_gain_over_replay": _relative_gain(
                            candidate_mse, replay_mse
                        ),
                        "candidate_relative_gain_over_global": _relative_gain(
                            candidate_mse, global_mse
                        ),
                        "candidate_normalized_excess_vs_replay": (
                            candidate_mse - replay_mse
                        )
                        / available,
                        "candidate_normalized_excess_vs_global": (
                            candidate_mse - global_mse
                        )
                        / available,
                        "candidate_gradient_cosine": float(
                            candidate_row["training"]["mean_gradient_cosine"]
                        ),
                    }
                )

    horizon_rows = []
    for horizon in horizons:
        selected = [cell for cell in cells if cell["horizon_multiplier"] == horizon]
        horizon_rows.append(
            {
                "horizon_multiplier": horizon,
                "sequence_length": BASE_LENGTH * horizon,
                "full_learning_fraction": float(
                    np.mean([cell["full_learning_fraction"] for cell in selected])
                ),
                "candidate_full_gain_retention": float(
                    np.mean([cell["candidate_full_gain_retention"] for cell in selected])
                ),
                "candidate_relative_gain_over_replay": float(
                    np.mean([cell["candidate_relative_gain_over_replay"] for cell in selected])
                ),
                "candidate_relative_gain_over_global": float(
                    np.mean([cell["candidate_relative_gain_over_global"] for cell in selected])
                ),
                "candidate_gradient_cosine": float(
                    np.mean([cell["candidate_gradient_cosine"] for cell in selected])
                ),
            }
        )

    minimum_horizon = min(horizons)
    maximum_horizon = max(horizons)
    seed_rows = []
    for replicate in replicates:
        selected = [cell for cell in cells if cell["replicate"] == replicate]
        short = [
            cell
            for cell in selected
            if cell["horizon_multiplier"] == minimum_horizon
        ]
        long = [
            cell
            for cell in selected
            if cell["horizon_multiplier"] == maximum_horizon
        ]
        short_gain = float(
            np.mean([cell["candidate_relative_gain_over_replay"] for cell in short])
        )
        long_gain = float(
            np.mean([cell["candidate_relative_gain_over_replay"] for cell in long])
        )
        seed_rows.append(
            {
                "replicate": replicate,
                "candidate_full_gain_retention": float(
                    np.mean([cell["candidate_full_gain_retention"] for cell in selected])
                ),
                "candidate_gradient_cosine": float(
                    np.mean([cell["candidate_gradient_cosine"] for cell in selected])
                ),
                "long_replay_relative_gain": long_gain,
                "long_global_relative_gain": float(
                    np.mean([cell["candidate_relative_gain_over_global"] for cell in long])
                ),
                "horizon_interaction": long_gain - short_gain,
            }
        )
    bootstrap = {
        field: _bootstrap_mean(
            [float(row[field]) for row in seed_rows],
            config.bootstrap_samples,
            stable_seed("o10-v2-bootstrap", rank, maximum_horizon, field),
        )
        for field in (
            "candidate_full_gain_retention",
            "candidate_gradient_cosine",
            "long_replay_relative_gain",
            "long_global_relative_gain",
            "horizon_interaction",
        )
    }

    candidate_state = int(
        next(row for row in rows if row["method"] == "layerwise_sketch")[
            "training"
        ]["allocated_state_floats_per_episode"]
    )
    full_state = int(
        next(row for row in rows if row["method"] == "full_trace")["training"][
            "allocated_state_floats_per_episode"
        ]
    )
    matched_state = all(
        int(row["training"]["allocated_state_floats_per_episode"])
        == candidate_state
        for row in rows
        if row["method"] != "full_trace"
    )
    matched_parameters = len({row["training"]["parameters"] for row in rows}) == 1
    matched_steps = len({row["training"]["optimizer_steps"] for row in rows}) == 1
    long_cells = [cell for cell in cells if cell["horizon_multiplier"] == maximum_horizon]
    replay_wins = sum(cell["candidate_mse"] < cell["replay_mse"] for cell in long_cells)
    global_wins = sum(
        cell["candidate_mse"] < cell["global_sketch_mse"] for cell in long_cells
    )
    required_wins = math.ceil(config.minimum_long_cell_win_fraction * len(long_cells))
    checks = {
        "full_trace_learns_overall": float(
            np.mean([cell["full_learning_fraction"] for cell in cells])
        )
        >= config.minimum_full_learning_overall,
        "full_trace_learns_at_every_horizon": min(
            row["full_learning_fraction"] for row in horizon_rows
        )
        >= config.minimum_full_learning_each_horizon,
        "full_trace_has_minimum_cell_headroom": min(
            cell["full_learning_fraction"] for cell in cells
        )
        >= config.minimum_full_learning_each_cell,
        "candidate_mean_gain_retention_is_at_least_0_95": bootstrap[
            "candidate_full_gain_retention"
        ]["mean"]
        >= config.minimum_retention_mean,
        "candidate_retention_lower_bound_is_at_least_0_90": bootstrap[
            "candidate_full_gain_retention"
        ]["lower_95"]
        >= config.minimum_retention_lower_95,
        "candidate_retains_at_least_0_85_at_every_horizon": min(
            row["candidate_full_gain_retention"] for row in horizon_rows
        )
        >= config.minimum_retention_each_horizon,
        "candidate_beats_selected_replay_at_longest_horizon_by_0_05": bootstrap[
            "long_replay_relative_gain"
        ]["mean"]
        >= config.minimum_long_replay_gain,
        "longest_horizon_replay_gain_lower_bound_is_positive": bootstrap[
            "long_replay_relative_gain"
        ]["lower_95"]
        > 0.0,
        "candidate_replay_advantage_grows_by_0_05_from_shortest_to_longest": bootstrap[
            "horizon_interaction"
        ]["mean"]
        >= config.minimum_horizon_interaction,
        "horizon_interaction_lower_bound_is_positive": bootstrap[
            "horizon_interaction"
        ]["lower_95"]
        > 0.0,
        "candidate_point_gain_is_positive_at_both_16x_and_32x_when_extended": (
            all(
                row["candidate_relative_gain_over_replay"] > 0.0
                for row in horizon_rows[-2:]
            )
            if len(horizons) >= 5
            else True
        ),
        "candidate_is_noninferior_to_global_sketch_at_longest_horizon": bootstrap[
            "long_global_relative_gain"
        ]["lower_95"]
        >= -config.global_noninferiority_margin,
        "candidate_wins_required_long_replay_cells": replay_wins >= required_wins,
        "candidate_mean_gradient_cosine_is_at_least_0_88": bootstrap[
            "candidate_gradient_cosine"
        ]["mean"]
        >= config.minimum_mean_cosine,
        "candidate_cosine_is_at_least_0_80_at_every_horizon": min(
            row["candidate_gradient_cosine"] for row in horizon_rows
        )
        >= config.minimum_horizon_cosine,
        "candidate_respects_state_ceiling": candidate_state / full_state
        <= maximum_state_fraction,
        "allocated_state_is_matched": matched_state,
        "parameters_are_matched": matched_parameters,
        "optimizer_steps_are_matched": matched_steps,
    }
    reference_valid = all(
        checks[key]
        for key in (
            "full_trace_learns_overall",
            "full_trace_learns_at_every_horizon",
            "full_trace_has_minimum_cell_headroom",
        )
    )
    gate_passed = all(checks.values())
    layer_specificity = (
        bootstrap["long_global_relative_gain"]["mean"]
        >= config.minimum_layer_specific_gain
        and bootstrap["long_global_relative_gain"]["lower_95"] > 0.0
        and global_wins >= required_wins
    )
    extension_warranted = (
        not gate_passed
        and reference_valid
        and bootstrap["candidate_full_gain_retention"]["mean"]
        >= config.extension_retention_floor
        and bootstrap["candidate_gradient_cosine"]["mean"]
        >= config.extension_cosine_floor
        and bootstrap["long_replay_relative_gain"]["mean"]
        >= config.extension_replay_gain_floor
        and (
            bootstrap["horizon_interaction"]["mean"] >= 0.0
            or bootstrap["long_replay_relative_gain"]["mean"] > 0.0
        )
        and matched_state
        and matched_parameters
        and matched_steps
    )
    capacity_rescue_warranted = (
        not gate_passed
        and not extension_warranted
        and reference_valid
        and bootstrap["candidate_full_gain_retention"]["mean"]
        >= config.rescue_retention_floor
        and bootstrap["candidate_gradient_cosine"]["mean"]
        >= config.rescue_cosine_floor
        and bootstrap["long_replay_relative_gain"]["mean"] >= -0.15
        and (
            bootstrap["candidate_full_gain_retention"]["mean"]
            < config.minimum_retention_mean
            or bootstrap["candidate_gradient_cosine"]["mean"]
            < config.minimum_mean_cosine
        )
        and matched_state
        and matched_parameters
        and matched_steps
    )
    return {
        "trace_rank": rank,
        "horizons": list(horizons),
        "state": {
            "candidate_floats": candidate_state,
            "full_trace_floats": full_state,
            "candidate_fraction_of_full": candidate_state / full_state,
            "matched_global_rank": global_rank_for_budget(
                make_v2_task(17, 2, 1, split="analysis"), candidate_state
            ),
        },
        "cells": cells,
        "horizon_aggregates": horizon_rows,
        "replicate_aggregates": seed_rows,
        "bootstrap_over_paired_replicates": bootstrap,
        "longest_horizon_replay_wins": replay_wins,
        "longest_horizon_global_wins": global_wins,
        "required_longest_horizon_wins": required_wins,
        "checks": checks,
        "reference_valid": reference_valid,
        "gate_passed": gate_passed,
        "layerwise_specificity_established": layer_specificity,
        "horizon_extension_warranted": gate_passed or extension_warranted,
        "capacity_rescue_warranted": capacity_rescue_warranted,
        "decisive_failure": not gate_passed
        and reference_valid
        and not extension_warranted
        and not capacity_rescue_warranted,
        "reference_invalid": not reference_valid,
    }


def _stage_summary(
    *,
    stage: str,
    premise: dict[str, object],
    rows: list[dict[str, object]],
    initial: dict[str, dict[str, object]],
    analysis: dict[str, object],
    replay_by_horizon: dict[int, str],
) -> dict[str, object]:
    return {
        "project": PROJECT,
        "version": VERSION,
        "stage": stage,
        "main_question": premise["main_question"],
        "selected_replay_by_horizon": replay_by_horizon,
        "initial_evaluations": initial,
        "rows": rows,
        "analysis": analysis,
        "gate_passed": analysis["gate_passed"],
    }


def run(output_dir: Path, device: torch.device, config: Config) -> dict[str, object]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=False)
    progress_path = output_dir / "progress.jsonl"
    premise = premise_summary_v2()
    _write_json(output_dir / "premise_summary.json", premise)
    _append_jsonl(
        progress_path,
        {"event": "premise_complete", "gate_passed": premise["gate_passed"]},
    )
    if not premise["gate_passed"]:
        summary = {
            "project": PROJECT,
            "version": VERSION,
            "decision": "repair_o10_v2_reference_before_scientific_inference",
            "premise_passed": False,
            "gate_passed": False,
            "runtime_seconds": time.perf_counter() - started,
            "config": asdict(config),
        }
        _write_json(output_dir / "cascade_summary.json", summary)
        return summary

    primary_lrs, primary_replay, primary_calibration = calibrate(
        methods=PRIMARY_METHODS,
        horizons=PRIMARY_HORIZONS,
        rank=PRIMARY_RANK,
        stage="primary_calibration",
        output_dir=output_dir,
        progress_path=progress_path,
        config=config,
        device=device,
    )
    if not primary_calibration["gate_passed"]:
        summary = {
            "project": PROJECT,
            "version": VERSION,
            "decision": "repair_o10_v2_calibration_before_scientific_inference",
            "premise_passed": True,
            "calibration_passed": False,
            "gate_passed": False,
            "runtime_seconds": time.perf_counter() - started,
            "config": asdict(config),
        }
        _write_json(output_dir / "cascade_summary.json", summary)
        return summary

    primary_rows, primary_initial = run_stage(
        stage="five_replicate_actual_horizon_primary",
        replicates=CONFIRMATION_REPLICATES,
        horizons=PRIMARY_HORIZONS,
        methods=PRIMARY_METHODS,
        rank=PRIMARY_RANK,
        steps=config.primary_steps,
        eval_examples=config.primary_eval_examples,
        selected_learning_rates=primary_lrs,
        config=config,
        device=device,
        output_dir=output_dir,
        progress_path=progress_path,
    )
    primary_analysis = analyze(
        primary_rows,
        primary_initial,
        replicates=CONFIRMATION_REPLICATES,
        horizons=PRIMARY_HORIZONS,
        rank=PRIMARY_RANK,
        replay_by_horizon=primary_replay,
        config=config,
        maximum_state_fraction=config.maximum_primary_state_fraction,
    )
    primary_summary = _stage_summary(
        stage="five-replicate rank-eight actual-horizon primary",
        premise=premise,
        rows=primary_rows,
        initial=primary_initial,
        analysis=primary_analysis,
        replay_by_horizon=primary_replay,
    )
    _write_json(output_dir / "primary_summary.json", primary_summary)

    selected_rank: int | None = None
    base_rows = primary_rows
    base_initial = primary_initial
    base_lrs = primary_lrs
    base_replay = primary_replay
    base_analysis = primary_analysis
    rescue_summary = None
    rescue_calibration_failed = False
    if primary_analysis["gate_passed"] or primary_analysis[
        "horizon_extension_warranted"
    ]:
        selected_rank = PRIMARY_RANK
    elif primary_analysis["capacity_rescue_warranted"]:
        rescue_lrs, rescue_replay, rescue_calibration = calibrate(
            methods=PRIMARY_METHODS,
            horizons=PRIMARY_HORIZONS,
            rank=RESCUE_RANK,
            stage="rank_twelve_calibration",
            output_dir=output_dir,
            progress_path=progress_path,
            config=config,
            device=device,
        )
        if rescue_calibration["gate_passed"]:
            rescue_rows, rescue_initial = run_stage(
                stage="five_replicate_rank_twelve_capacity_rescue",
                replicates=CONFIRMATION_REPLICATES,
                horizons=PRIMARY_HORIZONS,
                methods=PRIMARY_METHODS,
                rank=RESCUE_RANK,
                steps=config.rescue_steps,
                eval_examples=config.primary_eval_examples,
                selected_learning_rates=rescue_lrs,
                config=config,
                device=device,
                output_dir=output_dir,
                progress_path=progress_path,
            )
            rescue_analysis = analyze(
                rescue_rows,
                rescue_initial,
                replicates=CONFIRMATION_REPLICATES,
                horizons=PRIMARY_HORIZONS,
                rank=RESCUE_RANK,
                replay_by_horizon=rescue_replay,
                config=config,
                maximum_state_fraction=config.maximum_rescue_state_fraction,
            )
            rescue_summary = _stage_summary(
                stage="automatic rank-twelve capacity rescue",
                premise=premise,
                rows=rescue_rows,
                initial=rescue_initial,
                analysis=rescue_analysis,
                replay_by_horizon=rescue_replay,
            )
            _write_json(output_dir / "rescue_summary.json", rescue_summary)
            if rescue_analysis["gate_passed"] or rescue_analysis[
                "horizon_extension_warranted"
            ]:
                selected_rank = RESCUE_RANK
                base_rows = rescue_rows
                base_initial = rescue_initial
                base_lrs = rescue_lrs
                base_replay = rescue_replay
                base_analysis = rescue_analysis
        else:
            rescue_calibration_failed = True

    extension_summary = None
    extension_calibration_failed = False
    if selected_rank is not None:
        extension_lrs, extension_replay, extension_calibration = calibrate(
            methods=EXTENSION_METHODS,
            horizons=EXTENSION_HORIZONS,
            rank=selected_rank,
            stage=f"rank_{selected_rank}_long_horizon_calibration",
            output_dir=output_dir,
            progress_path=progress_path,
            config=config,
            device=device,
        )
        if extension_calibration["gate_passed"]:
            extension_rows, extension_initial = run_stage(
                stage=f"rank_{selected_rank}_actual_16x_32x_extension",
                replicates=CONFIRMATION_REPLICATES,
                horizons=EXTENSION_HORIZONS,
                methods=EXTENSION_METHODS,
                rank=selected_rank,
                steps=config.extension_steps,
                eval_examples=config.primary_eval_examples,
                selected_learning_rates=extension_lrs,
                config=config,
                device=device,
                output_dir=output_dir,
                progress_path=progress_path,
            )
            combined_rows = base_rows + extension_rows
            combined_initial = {**base_initial, **extension_initial}
            combined_replay = {**base_replay, **extension_replay}
            combined_analysis = analyze(
                combined_rows,
                combined_initial,
                replicates=CONFIRMATION_REPLICATES,
                horizons=PRIMARY_HORIZONS + EXTENSION_HORIZONS,
                rank=selected_rank,
                replay_by_horizon=combined_replay,
                config=config,
                maximum_state_fraction=(
                    config.maximum_primary_state_fraction
                    if selected_rank == PRIMARY_RANK
                    else config.maximum_rescue_state_fraction
                ),
            )
            extension_summary = _stage_summary(
                stage="paired actual-delay extension through 16x and 32x",
                premise=premise,
                rows=extension_rows,
                initial=extension_initial,
                analysis=combined_analysis,
                replay_by_horizon=combined_replay,
            )
            extension_summary["analysis_uses_primary_and_extension_rows"] = True
            _write_json(output_dir / "extension_summary.json", extension_summary)
        else:
            extension_calibration_failed = True

    selected_gate = base_analysis["gate_passed"] if selected_rank is not None else False
    extension_gate = bool(extension_summary and extension_summary["gate_passed"])
    selected_specificity = (
        extension_summary["analysis"]["layerwise_specificity_established"]
        if extension_summary is not None
        else base_analysis["layerwise_specificity_established"]
        if selected_rank is not None
        else False
    )
    if rescue_calibration_failed:
        decision = "repair_o10_v2_rank_twelve_calibration_no_negative_inference"
        gate_passed = False
    elif extension_calibration_failed:
        decision = "repair_o10_v2_long_horizon_calibration_no_negative_inference"
        gate_passed = False
    elif extension_gate and selected_specificity:
        decision = "o10_v2_layerwise_long_delay_crossover_positive"
        gate_passed = True
    elif extension_gate:
        decision = "o10_v2_trace_crossover_positive_layerwise_specificity_not_established"
        gate_passed = True
    elif extension_summary is not None and extension_summary["analysis"][
        "reference_invalid"
    ]:
        decision = (
            "o10_v2_eight_x_positive_long_horizon_reference_invalid"
            if selected_gate
            else "o10_v2_long_horizon_reference_invalid_no_negative_inference"
        )
        gate_passed = selected_gate
    elif selected_gate:
        decision = "o10_v2_eight_x_positive_but_16x_32x_boundary_not_confirmed"
        gate_passed = True
    elif primary_analysis["reference_invalid"]:
        decision = "o10_v2_primary_reference_invalid_no_negative_inference"
        gate_passed = False
    else:
        decision = "stop_o10_v2_after_decisive_actual_horizon_test_move_to_n10"
        gate_passed = False
    summary = {
        "project": PROJECT,
        "version": VERSION,
        "decision": decision,
        "premise_passed": True,
        "primary_calibration_passed": True,
        "primary_passed": primary_analysis["gate_passed"],
        "rank_twelve_rescue_ran": rescue_summary is not None,
        "rank_twelve_calibration_failed": rescue_calibration_failed,
        "rank_twelve_rescue_passed": bool(rescue_summary and rescue_summary["gate_passed"]),
        "long_horizon_extension_ran": extension_summary is not None,
        "long_horizon_calibration_failed": extension_calibration_failed,
        "long_horizon_extension_passed": extension_gate,
        "selected_trace_rank": selected_rank,
        "layerwise_specificity_established": selected_specificity,
        "gate_passed": gate_passed,
        "runtime_seconds": time.perf_counter() - started,
        "config": asdict(config),
        "frozen_priors_before_v2": {
            "p_positive": 0.58,
            "p_big3_given_positive": 0.40,
            "p_tmlr_given_positive": 0.70,
            "p_jmlr_given_positive": 0.38,
        },
        "claim_boundary": (
            "The reference implementation measures persistent eligibility-state information. "
            "It materializes event factors and therefore does not establish lower peak CUDA "
            "memory, lower wall time, or a systems-level online implementation."
        ),
    }
    _write_json(output_dir / "cascade_summary.json", summary)
    _append_jsonl(
        progress_path,
        {
            "event": "cascade_complete",
            "decision": decision,
            "gate_passed": gate_passed,
            "runtime_seconds": summary["runtime_seconds"],
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--calibration-steps", type=int, default=Config.calibration_steps
    )
    parser.add_argument("--primary-steps", type=int, default=Config.primary_steps)
    parser.add_argument("--rescue-steps", type=int, default=Config.rescue_steps)
    parser.add_argument("--extension-steps", type=int, default=Config.extension_steps)
    parser.add_argument(
        "--primary-batch-size", type=int, default=Config.primary_batch_size
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    config = Config(
        calibration_steps=args.calibration_steps,
        primary_steps=args.primary_steps,
        rescue_steps=args.rescue_steps,
        extension_steps=args.extension_steps,
        primary_batch_size=args.primary_batch_size,
    )
    summary = run(args.output_dir, device, config)
    print(
        json.dumps(
            {
                key: summary.get(key)
                for key in (
                    "decision",
                    "primary_passed",
                    "rank_twelve_rescue_ran",
                    "rank_twelve_rescue_passed",
                    "long_horizon_extension_ran",
                    "long_horizon_extension_passed",
                    "selected_trace_rank",
                    "layerwise_specificity_established",
                    "gate_passed",
                    "runtime_seconds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
