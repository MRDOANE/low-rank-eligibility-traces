from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    import torch

    from .o10_horizon_experiment import (
        evaluation_batches,
        evaluate_weights,
        initial_evaluation,
    )
    from .o10_layerwise import method_state, train_method
except ModuleNotFoundError:  # Permit CPU-only decision-rule tests.
    torch = None

from .n08_o10_stacked_experiment import (
    BASE_LENGTH,
    CANDIDATE,
    CLAIM_BOUNDARY,
    GLOBAL_ORACLE,
    LATENT_RANKS,
    LAYERWISE_CONTROL,
    PRIMARY_RANK,
    REPLAY_BY_HORIZON,
    _stacked_right_subspace_rank_for_budget,
    make_bridge_task,
    premise_summary as v10_premise_summary,
)
from .o10_tasks import stable_seed


PROJECT = "N08+O10 Optimizer-Fair Stacked Right Subspace Repair"
VERSION = "1.1"
SOURCE_RESULT_ARCHIVE = "n08_o10_stacked_v10_results.zip"
SOURCE_RESULT_SHA256 = "ff5391547a6411ef98ae7c3b344a05a2d07f16aa4e87b9a8ee0577ddaad380e4"

CALIBRATION_HORIZONS = (8, 16, 32)
SCREEN_HORIZONS = (8, 32)
CONFIRMATION_HORIZONS = (8, 16, 32)
BOUNDARY_HORIZONS = (16, 32)

# The v1.0 boundary did not run, so these teachers were never evaluated. Two are
# used for tuning and the third is a clean holdout for the selected rates.
CALIBRATION_TEACHERS = (1709, 1801)
CALIBRATION_HOLDOUT_TEACHERS = (1901,)

# These are the two v1.0 teachers containing all ten failed exact-reference cells.
# They are used only for an apparatus audit and never enter a performance estimate.
REFERENCE_AUDIT_TEACHERS = (907, 1201)

# Every inferential teacher below is new to v1.0 and to optimizer calibration.
SCREEN_TEACHERS = (2003, 2111, 2203)
CONFIRMATION_TEACHERS = (2309, 2411, 2503, 2609, 2707, 2801, 2903, 3001)
BOUNDARY_TEACHERS = (3109, 3203, 3301)

CALIBRATION_TRAIN_REPLICATES = (0,)
CALIBRATION_HOLDOUT_TRAIN_REPLICATES = (0,)
REFERENCE_AUDIT_TRAIN_REPLICATES = (0, 1)
SCREEN_TRAIN_REPLICATES = (0,)
CONFIRMATION_TRAIN_REPLICATES = (0, 1)
BOUNDARY_TRAIN_REPLICATES = (0,)


@dataclass(frozen=True)
class Config:
    learning_rates: tuple[float, ...] = (0.00025, 0.0005, 0.001, 0.002, 0.004)
    calibration_steps: int = 350
    calibration_holdout_steps: int = 350
    reference_audit_steps: int = 550
    screen_steps: int = 350
    confirmation_steps: int = 550
    boundary_steps: int = 450
    batch_size: int = 8
    calibration_eval_examples: int = 128
    primary_eval_examples: int = 256
    target_noise: float = 0.002
    weight_decay: float = 0.0001
    average_start_fraction: float = 0.50
    bootstrap_samples: int = 10_000
    minimum_full_learning_overall: float = 0.60
    minimum_full_learning_each_horizon: float = 0.45
    minimum_full_learning_each_cell: float = 0.20
    maximum_holdout_normalized_mse: float = 1.25
    minimum_retention_mean: float = 0.95
    minimum_retention_lower_95: float = 0.90
    minimum_retention_each_horizon: float = 0.85
    minimum_mean_cosine: float = 0.90
    minimum_horizon_cosine: float = 0.85
    noninferiority_margin_replay: float = 0.03
    noninferiority_margin_global: float = 0.03
    noninferiority_margin_layerwise: float = 0.02
    minimum_strong_layerwise_gain: float = 0.01
    maximum_candidate_state_fraction: float = 0.27
    screen_retention_floor: float = 0.90
    screen_cosine_floor: float = 0.85
    screen_control_gain_floor: float = -0.10
    screen_maximum_collapse_fraction: float = 0.25


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _bootstrap_mean(values: list[float], samples: int, seed: int) -> dict[str, float]:
    if not values:
        raise ValueError("bootstrap requires at least one independent teacher")
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data), size=(samples, len(data)))
    means = data[indices].mean(axis=1)
    return {
        "mean": float(data.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
    }


def _initial_key(teacher: int, latent_rank: int, horizon: int) -> str:
    return f"teacher{teacher}:latent{latent_rank}:h{horizon}"


def _methods_for_horizon(horizon: int) -> tuple[str, ...]:
    return (
        "full_trace",
        CANDIDATE,
        LAYERWISE_CONTROL,
        GLOBAL_ORACLE,
        REPLAY_BY_HORIZON[horizon],
    )


def _fresh_teacher_sets_are_disjoint() -> bool:
    groups = (
        set(CALIBRATION_TEACHERS),
        set(CALIBRATION_HOLDOUT_TEACHERS),
        set(SCREEN_TEACHERS),
        set(CONFIRMATION_TEACHERS),
        set(BOUNDARY_TEACHERS),
    )
    return all(groups[i].isdisjoint(groups[j]) for i in range(len(groups)) for j in range(i))


def premise_summary(device: torch.device) -> dict[str, object]:
    summary = dict(v10_premise_summary(device))
    checks = dict(summary["checks"])
    checks.update(
        {
            "all_fresh_teacher_sets_are_disjoint": _fresh_teacher_sets_are_disjoint(),
            "known_failure_teachers_are_audit_only": set(REFERENCE_AUDIT_TEACHERS).isdisjoint(
                set(CALIBRATION_TEACHERS)
                | set(CALIBRATION_HOLDOUT_TEACHERS)
                | set(SCREEN_TEACHERS)
                | set(CONFIRMATION_TEACHERS)
                | set(BOUNDARY_TEACHERS)
            ),
            "learning_rate_grid_brackets_v10_candidate_and_control_rates": {
                0.001,
                0.002,
            }.issubset(set(Config.learning_rates)),
        }
    )
    summary.update(
        {
            "project": PROJECT,
            "version": VERSION,
            "repair_scope": (
                "Repair the invalid v1.0 optimizer/reference apparatus only: calibrate every "
                "method by the same rule, validate selected rates on a clean holdout, and require "
                "the exact trace to pass the original failed cells before any fresh inference."
            ),
            "source_result_archive": SOURCE_RESULT_ARCHIVE,
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "v10_classification": (
                "invalid_inconclusive_confirmation_due_to_reference_failure_and_"
                "candidate_only_optimizer_calibration"
            ),
            "metric_repair": (
                "Control differences are divided by initial MSE. Gain retention is computed "
                "only after every exact-reference cell clears the frozen 20% learning gate."
            ),
            "checks": checks,
            "gate_passed": all(checks.values()),
        }
    )
    return summary


def run_one(
    task: object,
    batches: dict[str, object],
    *,
    teacher: int,
    train_replicate: int,
    horizon: int,
    method: str,
    steps: int,
    learning_rate: float,
    stage: str,
    seed_stage: str,
    config: Config,
    device: torch.device,
    checkpoint_root: Path | None = None,
) -> dict[str, object]:
    train_seed = stable_seed(
        "n08-o10-stacked-train",
        seed_stage,
        teacher,
        train_replicate,
        task.latent_rank,
        horizon,
    )
    average_start = max(1, int(math.ceil(steps * config.average_start_fraction)))
    trained = train_method(
        task,
        method=method,
        seed=train_seed,
        rank=PRIMARY_RANK,
        training_length=task.train_length,
        steps=steps,
        batch_size=config.batch_size,
        learning_rate=learning_rate,
        weight_decay=config.weight_decay,
        device=device,
        average_start_step=average_start,
    )
    primary_weights = trained.averaged_weights or trained.weights
    checkpoint = None
    if checkpoint_root is not None:
        checkpoint_path = (
            checkpoint_root
            / stage
            / f"teacher_{teacher}"
            / f"train_{train_replicate}"
            / f"latent_{task.latent_rank:02d}"
            / f"h_{horizon:02d}"
            / f"{method}.pt"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "project": PROJECT,
                "version": VERSION,
                "stage": stage,
                "seed_stage": seed_stage,
                "teacher": teacher,
                "train_replicate": train_replicate,
                "latent_rank": task.latent_rank,
                "horizon": horizon,
                "method": method,
                "learning_rate": learning_rate,
                "average_start_step": average_start,
                "final_weights": trained.weights,
                "averaged_weights": trained.averaged_weights,
            },
            checkpoint_path,
        )
        checkpoint = str(checkpoint_path.relative_to(checkpoint_root.parent))
    return {
        "stage": stage,
        "seed_stage": seed_stage,
        "task": task.name,
        "teacher": teacher,
        "train_replicate": train_replicate,
        "latent_rank": task.latent_rank,
        "horizon_multiplier": horizon,
        "training_sequence_length": task.train_length,
        "method": method,
        "trace_rank": PRIMARY_RANK,
        "learning_rate": learning_rate,
        "checkpoint": checkpoint,
        "training": {
            "runtime_seconds": trained.runtime_seconds,
            "parameters": trained.parameters,
            "optimizer_steps": trained.optimizer_steps,
            "allocated_state_floats_per_episode": trained.allocated_state_floats,
            "used_state_floats_per_episode": trained.used_state_floats,
            "allocated_state_bytes_fp32_per_episode": 4 * trained.allocated_state_floats,
            "averaged_steps": trained.averaged_steps,
            "average_start_step": average_start,
            "mean_gradient_cosine": trained.mean_gradient_cosine,
            "final_gradient_cosine": trained.final_gradient_cosine,
            "final_train_mse": trained.final_train_mse,
            "trajectory": trained.trajectory,
        },
        "evaluation": evaluate_weights(task, primary_weights, batches, device),
        "endpoint_evaluation": evaluate_weights(task, trained.weights, batches, device),
    }


def _record_progress(
    progress_path: Path,
    *,
    stage: str,
    row: dict[str, object],
    completed: int,
    total: int,
    stage_started: float,
) -> None:
    elapsed = time.perf_counter() - stage_started
    _append_jsonl(
        progress_path,
        {
            "event": "completed_run",
            "stage": stage,
            "teacher": row["teacher"],
            "train_replicate": row["train_replicate"],
            "latent_rank": row["latent_rank"],
            "horizon": row["horizon_multiplier"],
            "method": row["method"],
            "learning_rate": row["learning_rate"],
            "runtime_seconds": row["training"]["runtime_seconds"],
            "completed": completed,
            "total": total,
            "eta_seconds": elapsed / completed * (total - completed),
        },
    )


def select_learning_rates(
    rows: list[dict[str, object]],
    initial: dict[str, dict[str, object]],
    *,
    horizons: tuple[int, ...],
    learning_rates: tuple[float, ...],
) -> tuple[dict[str, dict[int, float]], dict[str, dict[str, list[dict[str, object]]]]]:
    selected: dict[str, dict[int, float]] = {}
    score_table: dict[str, dict[str, list[dict[str, object]]]] = {}
    for horizon in horizons:
        for method in _methods_for_horizon(horizon):
            scores = []
            for learning_rate in learning_rates:
                chosen = [
                    row
                    for row in rows
                    if int(row["horizon_multiplier"]) == horizon
                    and str(row["method"]) == method
                    and float(row["learning_rate"]) == learning_rate
                ]
                normalized = []
                for row in chosen:
                    key = _initial_key(
                        int(row["teacher"]), int(row["latent_rank"]), horizon
                    )
                    initial_mse = float(initial[key]["aggregate"]["mean_mse"])
                    mse = float(row["evaluation"]["aggregate"]["mean_mse"])
                    normalized.append(mse / max(initial_mse, 1e-12))
                if not normalized:
                    raise ValueError(
                        f"missing calibration rows for {method}, h={horizon}, lr={learning_rate}"
                    )
                finite = all(np.isfinite(value) for value in normalized)
                scores.append(
                    {
                        "learning_rate": learning_rate,
                        "finite": finite,
                        "median_normalized_mse": float(np.median(normalized))
                        if finite
                        else None,
                        "mean_normalized_mse": float(np.mean(normalized))
                        if finite
                        else None,
                        "maximum_normalized_mse": float(np.max(normalized))
                        if finite
                        else None,
                    }
                )
            scores.sort(
                key=lambda item: (
                    not item["finite"],
                    item["maximum_normalized_mse"]
                    if item["maximum_normalized_mse"] is not None
                    else float("inf"),
                    item["mean_normalized_mse"]
                    if item["mean_normalized_mse"] is not None
                    else float("inf"),
                    item["median_normalized_mse"]
                    if item["median_normalized_mse"] is not None
                    else float("inf"),
                    item["learning_rate"],
                )
            )
            selected.setdefault(method, {})[horizon] = float(scores[0]["learning_rate"])
            score_table.setdefault(method, {})[str(horizon)] = scores
    return selected, score_table


def calibrate_all_methods(
    *,
    output_dir: Path,
    progress_path: Path,
    config: Config,
    device: torch.device,
) -> tuple[dict[str, dict[int, float]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    initial: dict[str, dict[str, object]] = {}
    stage = "optimizer_calibration"
    seed_stage = "optimizer_repair_calibration"
    task_split = "optimizer_repair_calibration"
    total_runs = (
        len(CALIBRATION_TEACHERS)
        * len(LATENT_RANKS)
        * sum(len(_methods_for_horizon(horizon)) for horizon in CALIBRATION_HORIZONS)
        * len(config.learning_rates)
    )
    stage_started = time.perf_counter()
    for teacher in CALIBRATION_TEACHERS:
        for latent_rank in LATENT_RANKS:
            for horizon in CALIBRATION_HORIZONS:
                task = make_bridge_task(teacher, latent_rank, horizon, split=task_split)
                batches = evaluation_batches(
                    task,
                    config.calibration_eval_examples,
                    stable_seed(
                        "n08-o10-repair-calibration-eval", teacher, latent_rank, horizon
                    ),
                    config.target_noise,
                )
                initial[_initial_key(teacher, latent_rank, horizon)] = initial_evaluation(
                    task, batches
                )
                for method in _methods_for_horizon(horizon):
                    for learning_rate in config.learning_rates:
                        row = run_one(
                            task,
                            batches,
                            teacher=teacher,
                            train_replicate=0,
                            horizon=horizon,
                            method=method,
                            steps=config.calibration_steps,
                            learning_rate=learning_rate,
                            stage=stage,
                            seed_stage=seed_stage,
                            config=config,
                            device=device,
                        )
                        rows.append(row)
                        _record_progress(
                            progress_path,
                            stage=stage,
                            row=row,
                            completed=len(rows),
                            total=total_runs,
                            stage_started=stage_started,
                        )
    selected, score_table = select_learning_rates(
        rows,
        initial,
        horizons=CALIBRATION_HORIZONS,
        learning_rates=config.learning_rates,
    )
    expected_pairs = {
        (method, horizon)
        for horizon in CALIBRATION_HORIZONS
        for method in _methods_for_horizon(horizon)
    }
    selected_pairs = {
        (method, horizon) for method, by_horizon in selected.items() for horizon in by_horizon
    }
    checks = {
        "calibration_teachers_are_excluded_from_later_stages": set(
            CALIBRATION_TEACHERS
        ).isdisjoint(
            set(CALIBRATION_HOLDOUT_TEACHERS)
            | set(SCREEN_TEACHERS)
            | set(CONFIRMATION_TEACHERS)
            | set(BOUNDARY_TEACHERS)
        ),
        "every_method_horizon_pair_has_a_selected_rate": selected_pairs == expected_pairs,
        "every_selected_rate_is_on_the_frozen_grid": all(
            rate in config.learning_rates
            for by_horizon in selected.values()
            for rate in by_horizon.values()
        ),
        "every_calibration_run_is_finite": all(
            np.isfinite(float(row["evaluation"]["aggregate"]["mean_mse"]))
            for row in rows
        ),
    }
    summary = {
        "project": PROJECT,
        "version": VERSION,
        "stage": stage,
        "selection_objective": (
            "For each method and horizon, minimize worst normalized averaged-checkpoint MSE, "
            "then mean, then median; ties choose the lower learning rate. All methods receive "
            "the same frozen grid, teacher count, latent ranks, steps, and evaluation size."
        ),
        "learning_rate_grid": list(config.learning_rates),
        "selected_learning_rates": {
            method: {str(horizon): rate for horizon, rate in by_horizon.items()}
            for method, by_horizon in selected.items()
        },
        "score_table": score_table,
        "initial_evaluations": initial,
        "rows": rows,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }
    _write_json(output_dir / "optimizer_calibration_summary.json", summary)
    return selected, summary


def _selected_rate(
    selected_learning_rates: dict[str, dict[int, float]], method: str, horizon: int
) -> float:
    try:
        return float(selected_learning_rates[method][horizon])
    except KeyError as error:
        raise KeyError(f"no selected learning rate for {method} at {horizon}x") from error


def run_stage(
    *,
    stage: str,
    task_split: str,
    seed_stage: str,
    evaluation_seed_stage: str,
    teachers: tuple[int, ...],
    train_replicates: tuple[int, ...],
    latent_ranks: tuple[int, ...],
    horizons: tuple[int, ...],
    steps: int,
    selected_learning_rates: dict[str, dict[int, float]],
    output_dir: Path,
    progress_path: Path,
    config: Config,
    device: torch.device,
    reference_only: bool = False,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    initial: dict[str, dict[str, object]] = {}
    checkpoint_root = output_dir / "checkpoints"
    stage_started = time.perf_counter()
    runs_per_cell = 1 if reference_only else 5
    total_runs = (
        len(teachers)
        * len(latent_ranks)
        * len(horizons)
        * len(train_replicates)
        * runs_per_cell
    )
    for teacher in teachers:
        for latent_rank in latent_ranks:
            for horizon in horizons:
                task = make_bridge_task(teacher, latent_rank, horizon, split=task_split)
                batches = evaluation_batches(
                    task,
                    config.primary_eval_examples,
                    stable_seed(
                        "n08-o10-primary-eval",
                        evaluation_seed_stage,
                        teacher,
                        latent_rank,
                        horizon,
                    ),
                    config.target_noise,
                )
                initial[_initial_key(teacher, latent_rank, horizon)] = initial_evaluation(
                    task, batches
                )
                methods = ("full_trace",) if reference_only else _methods_for_horizon(horizon)
                for train_replicate in train_replicates:
                    for method in methods:
                        row = run_one(
                            task,
                            batches,
                            teacher=teacher,
                            train_replicate=train_replicate,
                            horizon=horizon,
                            method=method,
                            steps=steps,
                            learning_rate=_selected_rate(
                                selected_learning_rates, method, horizon
                            ),
                            stage=stage,
                            seed_stage=seed_stage,
                            config=config,
                            device=device,
                            checkpoint_root=checkpoint_root,
                        )
                        rows.append(row)
                        _record_progress(
                            progress_path,
                            stage=stage,
                            row=row,
                            completed=len(rows),
                            total=total_runs,
                            stage_started=stage_started,
                        )
    return rows, initial


def _row_map(
    rows: list[dict[str, object]],
) -> dict[tuple[int, int, int, int, str], dict[str, object]]:
    result = {}
    for row in rows:
        key = (
            int(row["teacher"]),
            int(row["train_replicate"]),
            int(row["latent_rank"]),
            int(row["horizon_multiplier"]),
            str(row["method"]),
        )
        if key in result:
            raise ValueError(f"duplicate row: {key}")
        result[key] = row
    return result


def analyze_calibration_holdout(
    rows: list[dict[str, object]],
    initial: dict[str, dict[str, object]],
    *,
    config: Config,
) -> dict[str, object]:
    cells = []
    for row in rows:
        teacher = int(row["teacher"])
        latent_rank = int(row["latent_rank"])
        horizon = int(row["horizon_multiplier"])
        initial_mse = float(
            initial[_initial_key(teacher, latent_rank, horizon)]["aggregate"]["mean_mse"]
        )
        mse = float(row["evaluation"]["aggregate"]["mean_mse"])
        cells.append(
            {
                "teacher": teacher,
                "latent_rank": latent_rank,
                "horizon_multiplier": horizon,
                "method": row["method"],
                "learning_rate": row["learning_rate"],
                "initial_mse": initial_mse,
                "mse": mse,
                "normalized_mse": mse / max(initial_mse, 1e-12),
                "learning_fraction": (initial_mse - mse) / max(initial_mse, 1e-12),
            }
        )
    full_cells = [cell for cell in cells if cell["method"] == "full_trace"]
    horizon_learning = {
        str(horizon): float(
            np.mean(
                [
                    cell["learning_fraction"]
                    for cell in full_cells
                    if cell["horizon_multiplier"] == horizon
                ]
            )
        )
        for horizon in CALIBRATION_HORIZONS
    }
    checks = {
        "all_selected_rate_runs_are_finite": all(
            np.isfinite(float(cell["mse"])) for cell in cells
        ),
        "no_selected_optimizer_is_catastrophic_on_holdout": max(
            float(cell["normalized_mse"]) for cell in cells
        )
        <= config.maximum_holdout_normalized_mse,
        "full_trace_learns_overall_on_holdout": float(
            np.mean([cell["learning_fraction"] for cell in full_cells])
        )
        >= config.minimum_full_learning_overall,
        "full_trace_learns_at_every_holdout_horizon": min(horizon_learning.values())
        >= config.minimum_full_learning_each_horizon,
        "full_trace_has_minimum_holdout_cell_headroom": min(
            float(cell["learning_fraction"]) for cell in full_cells
        )
        >= config.minimum_full_learning_each_cell,
    }
    return {
        "stage": "calibration_holdout",
        "role": "optimizer-selection validation only; no candidate inference",
        "cells": cells,
        "full_trace_learning_by_horizon": horizon_learning,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }


def analyze_reference_audit(
    rows: list[dict[str, object]],
    initial: dict[str, dict[str, object]],
    *,
    config: Config,
) -> dict[str, object]:
    cells = []
    for row in rows:
        teacher = int(row["teacher"])
        latent_rank = int(row["latent_rank"])
        horizon = int(row["horizon_multiplier"])
        initial_mse = float(
            initial[_initial_key(teacher, latent_rank, horizon)]["aggregate"]["mean_mse"]
        )
        full_mse = float(row["evaluation"]["aggregate"]["mean_mse"])
        cells.append(
            {
                "teacher": teacher,
                "train_replicate": int(row["train_replicate"]),
                "latent_rank": latent_rank,
                "horizon_multiplier": horizon,
                "learning_rate": float(row["learning_rate"]),
                "initial_mse": initial_mse,
                "full_trace_mse": full_mse,
                "full_learning_fraction": (initial_mse - full_mse)
                / max(initial_mse, 1e-12),
            }
        )
    horizon_learning = {
        str(horizon): float(
            np.mean(
                [
                    cell["full_learning_fraction"]
                    for cell in cells
                    if cell["horizon_multiplier"] == horizon
                ]
            )
        )
        for horizon in CONFIRMATION_HORIZONS
    }
    checks = {
        "all_exact_reference_runs_are_finite": all(
            np.isfinite(float(cell["full_trace_mse"])) for cell in cells
        ),
        "all_twelve_prespecified_audit_cells_are_present": len(cells) == 12,
        "full_trace_learns_overall_on_known_failure_teachers": float(
            np.mean([cell["full_learning_fraction"] for cell in cells])
        )
        >= config.minimum_full_learning_overall,
        "full_trace_learns_at_every_audit_horizon": min(horizon_learning.values())
        >= config.minimum_full_learning_each_horizon,
        "full_trace_clears_every_known_failure_cell": min(
            float(cell["full_learning_fraction"]) for cell in cells
        )
        >= config.minimum_full_learning_each_cell,
    }
    return {
        "stage": "reference_audit",
        "role": (
            "apparatus validity only; reuses the two diagnosed v1.0 teachers and therefore "
            "cannot contribute to candidate performance inference"
        ),
        "reproduces_v10_task_and_training_seeds": True,
        "cells": cells,
        "full_trace_learning_by_horizon": horizon_learning,
        "minimum_cell_learning_fraction": min(
            float(cell["full_learning_fraction"]) for cell in cells
        ),
        "checks": checks,
        "gate_passed": all(checks.values()),
    }


def _state_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    candidate_row = next(row for row in rows if row["method"] == CANDIDATE)
    full_row = next(row for row in rows if row["method"] == "full_trace")
    candidate_state = int(candidate_row["training"]["used_state_floats_per_episode"])
    allocated_state = int(
        candidate_row["training"]["allocated_state_floats_per_episode"]
    )
    full_state = int(full_row["training"]["used_state_floats_per_episode"])
    return {
        "candidate_used_floats": candidate_state,
        "allocated_floats": allocated_state,
        "full_trace_floats": full_state,
        "candidate_fraction_of_full": candidate_state / full_state,
        "stacked_rank": _stacked_right_subspace_rank_for_budget(
            make_bridge_task(17, 2, 1, split="analysis"), allocated_state
        ),
    }


def _wins(cells: list[dict[str, object]]) -> dict[str, int]:
    return {
        "candidate_better_than_full_trace": sum(
            cell["candidate_mse"] < cell["full_trace_mse"] for cell in cells
        ),
        "candidate_better_than_layerwise": sum(
            cell["candidate_mse"] < cell["layerwise_mse"] for cell in cells
        ),
        "candidate_better_than_global_oracle": sum(
            cell["candidate_mse"] < cell["global_oracle_mse"] for cell in cells
        ),
        "candidate_better_than_replay": sum(
            cell["candidate_mse"] < cell["replay_mse"] for cell in cells
        ),
        "total_cells": len(cells),
    }


def analyze(
    rows: list[dict[str, object]],
    initial: dict[str, dict[str, object]],
    *,
    teachers: tuple[int, ...],
    train_replicates: tuple[int, ...],
    horizons: tuple[int, ...],
    selected_learning_rates: dict[str, dict[int, float]],
    config: Config,
    stage: str,
) -> dict[str, object]:
    lookup = _row_map(rows)
    latent_ranks = sorted({int(row["latent_rank"]) for row in rows})
    cells: list[dict[str, object]] = []
    for teacher in teachers:
        for train_replicate in train_replicates:
            for latent_rank in latent_ranks:
                for horizon in horizons:
                    base = (teacher, train_replicate, latent_rank, horizon)
                    full = lookup[base + ("full_trace",)]
                    candidate = lookup[base + (CANDIDATE,)]
                    layerwise = lookup[base + (LAYERWISE_CONTROL,)]
                    oracle = lookup[base + (GLOBAL_ORACLE,)]
                    replay_method = REPLAY_BY_HORIZON[horizon]
                    replay = lookup[base + (replay_method,)]
                    initial_mse = float(
                        initial[_initial_key(teacher, latent_rank, horizon)]["aggregate"][
                            "mean_mse"
                        ]
                    )
                    scale = max(initial_mse, 1e-12)
                    full_mse = float(full["evaluation"]["aggregate"]["mean_mse"])
                    candidate_mse = float(
                        candidate["evaluation"]["aggregate"]["mean_mse"]
                    )
                    layerwise_mse = float(
                        layerwise["evaluation"]["aggregate"]["mean_mse"]
                    )
                    oracle_mse = float(oracle["evaluation"]["aggregate"]["mean_mse"])
                    replay_mse = float(replay["evaluation"]["aggregate"]["mean_mse"])
                    endpoint_mse = float(
                        candidate["endpoint_evaluation"]["aggregate"]["mean_mse"]
                    )
                    full_learning = (initial_mse - full_mse) / scale
                    candidate_learning = (initial_mse - candidate_mse) / scale
                    cells.append(
                        {
                            "teacher": teacher,
                            "train_replicate": train_replicate,
                            "latent_rank": latent_rank,
                            "horizon_multiplier": horizon,
                            "sequence_length": BASE_LENGTH * horizon,
                            "selected_replay": replay_method,
                            "learning_rates": {
                                "full_trace": float(full["learning_rate"]),
                                CANDIDATE: float(candidate["learning_rate"]),
                                LAYERWISE_CONTROL: float(layerwise["learning_rate"]),
                                GLOBAL_ORACLE: float(oracle["learning_rate"]),
                                replay_method: float(replay["learning_rate"]),
                            },
                            "initial_mse": initial_mse,
                            "full_trace_mse": full_mse,
                            "candidate_mse": candidate_mse,
                            "layerwise_mse": layerwise_mse,
                            "global_oracle_mse": oracle_mse,
                            "replay_mse": replay_mse,
                            "candidate_endpoint_mse": endpoint_mse,
                            "full_learning_fraction": full_learning,
                            "candidate_learning_fraction": candidate_learning,
                            # This ratio is descriptive only until the reference gate below passes.
                            "candidate_full_gain_retention": (
                                candidate_learning / full_learning
                                if full_learning >= config.minimum_full_learning_each_cell
                                else None
                            ),
                            # Stable comparison scale: the untrained model's initial error.
                            "candidate_gain_vs_layerwise": (
                                layerwise_mse - candidate_mse
                            )
                            / scale,
                            "candidate_gain_vs_global_oracle": (
                                oracle_mse - candidate_mse
                            )
                            / scale,
                            "candidate_gain_vs_replay": (replay_mse - candidate_mse)
                            / scale,
                            "averaging_gain": (endpoint_mse - candidate_mse) / scale,
                            "candidate_gradient_cosine": float(
                                candidate["training"]["mean_gradient_cosine"]
                            ),
                        }
                    )

    horizon_raw = []
    for horizon in horizons:
        selected = [cell for cell in cells if cell["horizon_multiplier"] == horizon]
        horizon_raw.append(
            {
                "horizon_multiplier": horizon,
                "sequence_length": BASE_LENGTH * horizon,
                "full_learning_fraction": float(
                    np.mean([cell["full_learning_fraction"] for cell in selected])
                ),
                "candidate_learning_fraction": float(
                    np.mean([cell["candidate_learning_fraction"] for cell in selected])
                ),
                "candidate_gain_vs_layerwise": float(
                    np.mean([cell["candidate_gain_vs_layerwise"] for cell in selected])
                ),
                "candidate_gain_vs_global_oracle": float(
                    np.mean(
                        [cell["candidate_gain_vs_global_oracle"] for cell in selected]
                    )
                ),
                "candidate_gain_vs_replay": float(
                    np.mean([cell["candidate_gain_vs_replay"] for cell in selected])
                ),
                "candidate_gradient_cosine": float(
                    np.mean([cell["candidate_gradient_cosine"] for cell in selected])
                ),
            }
        )

    reference_checks = {
        "all_primary_errors_are_finite": all(
            np.isfinite(float(cell[field]))
            for cell in cells
            for field in (
                "initial_mse",
                "full_trace_mse",
                "candidate_mse",
                "layerwise_mse",
                "global_oracle_mse",
                "replay_mse",
            )
        ),
        "full_trace_learns_overall": float(
            np.mean([cell["full_learning_fraction"] for cell in cells])
        )
        >= config.minimum_full_learning_overall,
        "full_trace_learns_at_every_horizon": min(
            row["full_learning_fraction"] for row in horizon_raw
        )
        >= config.minimum_full_learning_each_horizon,
        "full_trace_has_minimum_cell_headroom": min(
            float(cell["full_learning_fraction"]) for cell in cells
        )
        >= config.minimum_full_learning_each_cell,
    }
    reference_valid = all(reference_checks.values())
    state = _state_summary(rows)
    wins = _wins(cells)
    metric_definitions = {
        "learning_fraction": "(initial_mse - method_mse) / initial_mse",
        "candidate_gain_vs_control": "(control_mse - candidate_mse) / initial_mse",
        "candidate_full_gain_retention": (
            "candidate_learning_fraction / full_learning_fraction, computed for inference only "
            "after every full-trace cell learns by at least 20%"
        ),
        "bootstrap_unit": "independent teacher; nested training replicates stay within teacher",
    }
    if not reference_valid:
        return {
            "stage": stage,
            "analysis_status": "invalid_reference_no_performance_inference",
            "horizons": list(horizons),
            "metric_definitions": metric_definitions,
            "state": state,
            "cells": cells,
            "horizon_aggregates": horizon_raw,
            "teacher_aggregates": None,
            "bootstrap_over_independent_teachers": None,
            "collapse_fraction": None,
            "wins_descriptive_only": wins,
            "reference_checks": reference_checks,
            "core_checks": {},
            "strong_checks": {},
            "screen_checks": {"reference_is_valid": False},
            "reference_valid": False,
            "continue_after_screen": False,
            "supportive": False,
            "strong_positive": False,
            "gate_passed": False,
        }

    horizon_aggregates = []
    for row in horizon_raw:
        horizon = int(row["horizon_multiplier"])
        selected = [cell for cell in cells if cell["horizon_multiplier"] == horizon]
        horizon_aggregates.append(
            {
                **row,
                "candidate_full_gain_retention": float(
                    np.mean([cell["candidate_full_gain_retention"] for cell in selected])
                ),
            }
        )

    longest = max(horizons)
    teacher_aggregates = []
    for teacher in teachers:
        selected = [cell for cell in cells if cell["teacher"] == teacher]
        long = [cell for cell in selected if cell["horizon_multiplier"] == longest]
        teacher_aggregates.append(
            {
                "teacher": teacher,
                "candidate_full_gain_retention": float(
                    np.mean([cell["candidate_full_gain_retention"] for cell in selected])
                ),
                "candidate_gradient_cosine": float(
                    np.mean([cell["candidate_gradient_cosine"] for cell in selected])
                ),
                "candidate_gain_vs_layerwise": float(
                    np.mean([cell["candidate_gain_vs_layerwise"] for cell in selected])
                ),
                "candidate_gain_vs_global_oracle": float(
                    np.mean(
                        [cell["candidate_gain_vs_global_oracle"] for cell in selected]
                    )
                ),
                "candidate_gain_vs_replay": float(
                    np.mean([cell["candidate_gain_vs_replay"] for cell in selected])
                ),
                "long_gain_vs_layerwise": float(
                    np.mean([cell["candidate_gain_vs_layerwise"] for cell in long])
                ),
                "long_gain_vs_global_oracle": float(
                    np.mean([cell["candidate_gain_vs_global_oracle"] for cell in long])
                ),
                "long_gain_vs_replay": float(
                    np.mean([cell["candidate_gain_vs_replay"] for cell in long])
                ),
                "averaging_gain": float(
                    np.mean([cell["averaging_gain"] for cell in selected])
                ),
            }
        )
    fields = (
        "candidate_full_gain_retention",
        "candidate_gradient_cosine",
        "candidate_gain_vs_layerwise",
        "candidate_gain_vs_global_oracle",
        "candidate_gain_vs_replay",
        "long_gain_vs_layerwise",
        "long_gain_vs_global_oracle",
        "long_gain_vs_replay",
        "averaging_gain",
    )
    bootstrap = {
        field: _bootstrap_mean(
            [float(row[field]) for row in teacher_aggregates],
            config.bootstrap_samples,
            stable_seed("n08-o10-repair-bootstrap", stage, field),
        )
        for field in fields
    }

    candidate_row = next(row for row in rows if row["method"] == CANDIDATE)
    allocated_state = int(
        candidate_row["training"]["allocated_state_floats_per_episode"]
    )
    matched_parameters = len({int(row["training"]["parameters"]) for row in rows}) == 1
    matched_steps = len(
        {int(row["training"]["optimizer_steps"]) for row in rows}
    ) == 1
    matched_allocations = all(
        int(row["training"]["allocated_state_floats_per_episode"]) == allocated_state
        for row in rows
        if row["method"] != "full_trace"
    )
    selected_rates_applied = all(
        float(row["learning_rate"])
        == _selected_rate(
            selected_learning_rates,
            str(row["method"]),
            int(row["horizon_multiplier"]),
        )
        for row in rows
    )
    collapse_fraction = float(
        np.mean(
            [
                float(cell["candidate_full_gain_retention"]) < 0.80
                for cell in cells
            ]
        )
    )
    core_checks = {
        "candidate_mean_gain_retention_is_at_least_0_95": bootstrap[
            "candidate_full_gain_retention"
        ]["mean"]
        >= config.minimum_retention_mean,
        "candidate_retention_lower_bound_is_at_least_0_90": bootstrap[
            "candidate_full_gain_retention"
        ]["lower_95"]
        >= config.minimum_retention_lower_95,
        "candidate_retains_at_least_0_85_at_every_horizon": min(
            row["candidate_full_gain_retention"] for row in horizon_aggregates
        )
        >= config.minimum_retention_each_horizon,
        "candidate_mean_gradient_cosine_is_at_least_0_90": bootstrap[
            "candidate_gradient_cosine"
        ]["mean"]
        >= config.minimum_mean_cosine,
        "candidate_cosine_is_at_least_0_85_at_every_horizon": min(
            row["candidate_gradient_cosine"] for row in horizon_aggregates
        )
        >= config.minimum_horizon_cosine,
        "candidate_is_noninferior_to_replay": bootstrap["candidate_gain_vs_replay"][
            "lower_95"
        ]
        >= -config.noninferiority_margin_replay,
        "candidate_is_noninferior_to_global_oracle": bootstrap[
            "candidate_gain_vs_global_oracle"
        ]["lower_95"]
        >= -config.noninferiority_margin_global,
        "candidate_is_noninferior_to_equal_layer_sketch": bootstrap[
            "candidate_gain_vs_layerwise"
        ]["lower_95"]
        >= -config.noninferiority_margin_layerwise,
        "candidate_is_noninferior_at_longest_horizon": min(
            bootstrap["long_gain_vs_replay"]["lower_95"]
            + config.noninferiority_margin_replay,
            bootstrap["long_gain_vs_global_oracle"]["lower_95"]
            + config.noninferiority_margin_global,
            bootstrap["long_gain_vs_layerwise"]["lower_95"]
            + config.noninferiority_margin_layerwise,
        )
        >= 0.0,
        "candidate_respects_state_ceiling": state["candidate_fraction_of_full"]
        <= config.maximum_candidate_state_fraction,
        "all_compressed_methods_receive_matched_allocations": matched_allocations,
        "parameters_are_matched": matched_parameters,
        "optimizer_steps_are_matched": matched_steps,
        "method_specific_selected_rates_are_applied": selected_rates_applied,
        "weight_averaging_is_present": min(
            int(row["training"]["averaged_steps"]) for row in rows
        )
        > 0,
    }
    strong_checks = {
        "candidate_layerwise_gain_is_at_least_0_01_of_initial_error": bootstrap[
            "candidate_gain_vs_layerwise"
        ]["mean"]
        >= config.minimum_strong_layerwise_gain,
        "candidate_layerwise_gain_lower_bound_is_positive": bootstrap[
            "candidate_gain_vs_layerwise"
        ]["lower_95"]
        > 0.0,
        "candidate_long_layerwise_gain_lower_bound_is_positive": bootstrap[
            "long_gain_vs_layerwise"
        ]["lower_95"]
        > 0.0,
    }
    screen_checks = {
        "reference_is_valid": True,
        "screen_retention_is_adequate": bootstrap["candidate_full_gain_retention"][
            "mean"
        ]
        >= config.screen_retention_floor,
        "screen_cosine_is_adequate": bootstrap["candidate_gradient_cosine"]["mean"]
        >= config.screen_cosine_floor,
        "screen_layerwise_comparison_is_not_decisively_bad": bootstrap[
            "candidate_gain_vs_layerwise"
        ]["mean"]
        >= config.screen_control_gain_floor,
        "screen_global_comparison_is_not_decisively_bad": bootstrap[
            "candidate_gain_vs_global_oracle"
        ]["mean"]
        >= config.screen_control_gain_floor,
        "screen_replay_comparison_is_not_decisively_bad": bootstrap[
            "candidate_gain_vs_replay"
        ]["mean"]
        >= config.screen_control_gain_floor,
        "screen_collapse_fraction_is_bounded": collapse_fraction
        <= config.screen_maximum_collapse_fraction,
        "candidate_respects_state_ceiling": core_checks[
            "candidate_respects_state_ceiling"
        ],
        "comparison_process_is_matched": matched_allocations
        and matched_parameters
        and matched_steps
        and selected_rates_applied,
    }
    supportive = all(core_checks.values())
    strong_positive = supportive and all(strong_checks.values())
    continue_after_screen = all(screen_checks.values())
    return {
        "stage": stage,
        "analysis_status": "valid_inference",
        "horizons": list(horizons),
        "metric_definitions": metric_definitions,
        "state": state,
        "cells": cells,
        "horizon_aggregates": horizon_aggregates,
        "teacher_aggregates": teacher_aggregates,
        "bootstrap_over_independent_teachers": bootstrap,
        "collapse_fraction": collapse_fraction,
        "wins": wins,
        "reference_checks": reference_checks,
        "core_checks": core_checks,
        "strong_checks": strong_checks,
        "screen_checks": screen_checks,
        "reference_valid": True,
        "continue_after_screen": continue_after_screen,
        "supportive": supportive,
        "strong_positive": strong_positive,
        "gate_passed": strong_positive,
    }


def _stage_summary(
    *,
    stage: str,
    rows: list[dict[str, object]],
    initial: dict[str, dict[str, object]],
    analysis: dict[str, object],
) -> dict[str, object]:
    return {
        "project": PROJECT,
        "version": VERSION,
        "stage": stage,
        "claim_boundary": CLAIM_BOUNDARY,
        "initial_evaluations": initial,
        "rows": rows,
        "analysis": analysis,
        "gate_passed": analysis["gate_passed"],
    }


def run_cascade(output_dir: Path, config: Config, device: torch.device) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    progress_path = output_dir / "progress.jsonl"
    started = time.perf_counter()

    premise = premise_summary(device)
    _write_json(output_dir / "premise_summary.json", premise)
    calibration = None
    holdout_summary = None
    reference_audit_summary = None
    screen_summary = None
    confirmation_summary = None
    boundary_summary = None
    selected_learning_rates = None
    confirmation_analysis = None

    if not premise["gate_passed"]:
        decision = "stop_optimizer_repair_after_premise"
    else:
        selected_learning_rates, calibration = calibrate_all_methods(
            output_dir=output_dir,
            progress_path=progress_path,
            config=config,
            device=device,
        )
        if not calibration["gate_passed"]:
            decision = "stop_optimizer_repair_after_calibration"
        else:
            holdout_rows, holdout_initial = run_stage(
                stage="calibration_holdout",
                task_split="optimizer_repair_holdout",
                seed_stage="optimizer_repair_holdout",
                evaluation_seed_stage="optimizer_repair_holdout",
                teachers=CALIBRATION_HOLDOUT_TEACHERS,
                train_replicates=CALIBRATION_HOLDOUT_TRAIN_REPLICATES,
                latent_ranks=LATENT_RANKS,
                horizons=CALIBRATION_HORIZONS,
                steps=config.calibration_holdout_steps,
                selected_learning_rates=selected_learning_rates,
                output_dir=output_dir,
                progress_path=progress_path,
                config=config,
                device=device,
            )
            holdout_analysis = analyze_calibration_holdout(
                holdout_rows, holdout_initial, config=config
            )
            holdout_summary = _stage_summary(
                stage="calibration_holdout",
                rows=holdout_rows,
                initial=holdout_initial,
                analysis=holdout_analysis,
            )
            _write_json(output_dir / "calibration_holdout_summary.json", holdout_summary)
            if not holdout_analysis["gate_passed"]:
                decision = (
                    "optimizer_repair_invalid_after_calibration_holdout_"
                    "preserve_n08_and_o10_evidence"
                )
            else:
                audit_rows, audit_initial = run_stage(
                    stage="reference_audit",
                    task_split="confirmation",
                    seed_stage="confirmation",
                    evaluation_seed_stage="confirmation",
                    teachers=REFERENCE_AUDIT_TEACHERS,
                    train_replicates=REFERENCE_AUDIT_TRAIN_REPLICATES,
                    latent_ranks=(2,),
                    horizons=CONFIRMATION_HORIZONS,
                    steps=config.reference_audit_steps,
                    selected_learning_rates=selected_learning_rates,
                    output_dir=output_dir,
                    progress_path=progress_path,
                    config=config,
                    device=device,
                    reference_only=True,
                )
                reference_audit_analysis = analyze_reference_audit(
                    audit_rows, audit_initial, config=config
                )
                reference_audit_summary = _stage_summary(
                    stage="reference_audit",
                    rows=audit_rows,
                    initial=audit_initial,
                    analysis=reference_audit_analysis,
                )
                _write_json(
                    output_dir / "reference_audit_summary.json",
                    reference_audit_summary,
                )
                if not reference_audit_analysis["gate_passed"]:
                    decision = (
                        "optimizer_repair_invalid_reference_on_known_failure_cells_"
                        "no_bridge_inference"
                    )
                else:
                    screen_rows, screen_initial = run_stage(
                        stage="screen",
                        task_split="optimizer_repair_screen",
                        seed_stage="optimizer_repair_screen",
                        evaluation_seed_stage="optimizer_repair_screen",
                        teachers=SCREEN_TEACHERS,
                        train_replicates=SCREEN_TRAIN_REPLICATES,
                        latent_ranks=LATENT_RANKS,
                        horizons=SCREEN_HORIZONS,
                        steps=config.screen_steps,
                        selected_learning_rates=selected_learning_rates,
                        output_dir=output_dir,
                        progress_path=progress_path,
                        config=config,
                        device=device,
                    )
                    screen_analysis = analyze(
                        screen_rows,
                        screen_initial,
                        teachers=SCREEN_TEACHERS,
                        train_replicates=SCREEN_TRAIN_REPLICATES,
                        horizons=SCREEN_HORIZONS,
                        selected_learning_rates=selected_learning_rates,
                        config=config,
                        stage="screen",
                    )
                    screen_summary = _stage_summary(
                        stage="screen",
                        rows=screen_rows,
                        initial=screen_initial,
                        analysis=screen_analysis,
                    )
                    _write_json(output_dir / "screen_summary.json", screen_summary)
                    if not screen_analysis["reference_valid"]:
                        decision = (
                            "optimizer_repair_screen_invalid_reference_no_bridge_inference"
                        )
                    elif not screen_analysis["continue_after_screen"]:
                        decision = (
                            "stacked_trace_optimizer_fair_screen_negative_stop_"
                            "preserve_n08_and_o10"
                        )
                    else:
                        confirmation_rows, confirmation_initial = run_stage(
                            stage="confirmation",
                            task_split="optimizer_repair_confirmation",
                            seed_stage="optimizer_repair_confirmation",
                            evaluation_seed_stage="optimizer_repair_confirmation",
                            teachers=CONFIRMATION_TEACHERS,
                            train_replicates=CONFIRMATION_TRAIN_REPLICATES,
                            latent_ranks=LATENT_RANKS,
                            horizons=CONFIRMATION_HORIZONS,
                            steps=config.confirmation_steps,
                            selected_learning_rates=selected_learning_rates,
                            output_dir=output_dir,
                            progress_path=progress_path,
                            config=config,
                            device=device,
                        )
                        confirmation_analysis = analyze(
                            confirmation_rows,
                            confirmation_initial,
                            teachers=CONFIRMATION_TEACHERS,
                            train_replicates=CONFIRMATION_TRAIN_REPLICATES,
                            horizons=CONFIRMATION_HORIZONS,
                            selected_learning_rates=selected_learning_rates,
                            config=config,
                            stage="confirmation",
                        )
                        confirmation_summary = _stage_summary(
                            stage="confirmation",
                            rows=confirmation_rows,
                            initial=confirmation_initial,
                            analysis=confirmation_analysis,
                        )
                        _write_json(
                            output_dir / "confirmation_summary.json",
                            confirmation_summary,
                        )
                        if not confirmation_analysis["reference_valid"]:
                            decision = (
                                "optimizer_repair_confirmation_invalid_reference_"
                                "no_bridge_inference"
                            )
                        else:
                            if confirmation_analysis["supportive"]:
                                boundary_rows, boundary_initial = run_stage(
                                    stage="boundary",
                                    task_split="optimizer_repair_boundary",
                                    seed_stage="optimizer_repair_boundary",
                                    evaluation_seed_stage="optimizer_repair_boundary",
                                    teachers=BOUNDARY_TEACHERS,
                                    train_replicates=BOUNDARY_TRAIN_REPLICATES,
                                    latent_ranks=(8, 12),
                                    horizons=BOUNDARY_HORIZONS,
                                    steps=config.boundary_steps,
                                    selected_learning_rates=selected_learning_rates,
                                    output_dir=output_dir,
                                    progress_path=progress_path,
                                    config=config,
                                    device=device,
                                )
                                boundary_analysis = analyze(
                                    boundary_rows,
                                    boundary_initial,
                                    teachers=BOUNDARY_TEACHERS,
                                    train_replicates=BOUNDARY_TRAIN_REPLICATES,
                                    horizons=BOUNDARY_HORIZONS,
                                    selected_learning_rates=selected_learning_rates,
                                    config=config,
                                    stage="boundary",
                                )
                                boundary_summary = _stage_summary(
                                    stage="boundary",
                                    rows=boundary_rows,
                                    initial=boundary_initial,
                                    analysis=boundary_analysis,
                                )
                                _write_json(
                                    output_dir / "boundary_summary.json", boundary_summary
                                )
                            if confirmation_analysis["strong_positive"]:
                                decision = (
                                    "stacked_trace_optimizer_fair_strong_positive_"
                                    "proceed_to_online_tracker_and_external_benchmarks"
                                )
                            elif confirmation_analysis["supportive"]:
                                decision = (
                                    "stacked_trace_optimizer_fair_supportive_without_"
                                    "specific_advantage_preserve_claim_boundary"
                                )
                            else:
                                decision = (
                                    "stacked_trace_optimizer_fair_not_supported_stop_bridge_"
                                    "and_build_n08_centered_paper"
                                )

    summary = {
        "project": PROJECT,
        "version": VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "source_result_archive": SOURCE_RESULT_ARCHIVE,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "v10_result_classification": (
            "invalid_inconclusive_confirmation; no negative mechanism inference"
        ),
        "frozen_evidence_entering_repair": {
            "n08_v2": (
                "rank-four streaming low-rank trace passed all gates at 34.03% of full "
                "persistent state and beat matched replay at long horizons"
            ),
            "o10_v2": (
                "rank-eight equal-layer trace retained about 99% of full-trace gain but "
                "did not establish a reliable advantage over the same-budget global oracle"
            ),
            "stacked_v10": (
                "screen was strong, but confirmation was invalid because the exact trace "
                "failed 10 of 96 cell headroom checks while only the candidate had been tuned"
            ),
        },
        "frozen_priors_before_repair": {
            "p_reference_repair_valid": 0.90,
            "p_fair_screen_advances": 0.65,
            "p_supportive_confirmation": 0.45,
            "p_strong_stacked_advantage": 0.20,
            "p_big3_given_strong_positive": 0.45,
            "p_jmlr_given_positive": 0.62,
            "p_tmlr_given_positive": 0.82,
        },
        "config": asdict(config),
        "premise_passed": premise["gate_passed"],
        "calibration_passed": calibration is not None and calibration["gate_passed"],
        "calibration_holdout_passed": holdout_summary is not None
        and holdout_summary["analysis"]["gate_passed"],
        "reference_audit_passed": reference_audit_summary is not None
        and reference_audit_summary["analysis"]["gate_passed"],
        "screen_ran": screen_summary is not None,
        "screen_reference_valid": screen_summary is not None
        and screen_summary["analysis"]["reference_valid"],
        "screen_continue": screen_summary is not None
        and screen_summary["analysis"]["continue_after_screen"],
        "confirmation_ran": confirmation_summary is not None,
        "confirmation_reference_valid": confirmation_analysis is not None
        and confirmation_analysis["reference_valid"],
        "confirmation_supportive": confirmation_analysis is not None
        and confirmation_analysis["supportive"],
        "strong_positive": confirmation_analysis is not None
        and confirmation_analysis["strong_positive"],
        "boundary_ran": boundary_summary is not None,
        "selected_learning_rates": (
            {
                method: {str(horizon): rate for horizon, rate in by_horizon.items()}
                for method, by_horizon in selected_learning_rates.items()
            }
            if selected_learning_rates is not None
            else None
        ),
        "decision": decision,
        "gate_passed": confirmation_analysis is not None
        and confirmation_analysis["strong_positive"],
        "project_viable": confirmation_analysis is not None
        and confirmation_analysis["supportive"],
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(output_dir / "cascade_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=PROJECT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration-steps", type=int, default=Config.calibration_steps)
    parser.add_argument(
        "--calibration-holdout-steps",
        type=int,
        default=Config.calibration_holdout_steps,
    )
    parser.add_argument(
        "--reference-audit-steps", type=int, default=Config.reference_audit_steps
    )
    parser.add_argument("--screen-steps", type=int, default=Config.screen_steps)
    parser.add_argument(
        "--confirmation-steps", type=int, default=Config.confirmation_steps
    )
    parser.add_argument("--boundary-steps", type=int, default=Config.boundary_steps)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument(
        "--primary-eval-examples", type=int, default=Config.primary_eval_examples
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch is None:
        raise RuntimeError("PyTorch is required to run the N08+O10 optimizer repair")
    config = Config(
        calibration_steps=args.calibration_steps,
        calibration_holdout_steps=args.calibration_holdout_steps,
        reference_audit_steps=args.reference_audit_steps,
        screen_steps=args.screen_steps,
        confirmation_steps=args.confirmation_steps,
        boundary_steps=args.boundary_steps,
        batch_size=args.batch_size,
        primary_eval_examples=args.primary_eval_examples,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    summary = run_cascade(args.output_dir, config, device)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
