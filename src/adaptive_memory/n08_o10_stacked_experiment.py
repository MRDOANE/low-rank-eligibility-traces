from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

try:
    import torch

    from .o10_horizon_experiment import (
        evaluation_batches,
        evaluate_weights,
        initial_evaluation,
    )
    from .o10_layerwise import (
        LayerwiseAdapterModel,
        approximate_trace,
        full_trace,
        method_state,
        train_method,
    )
except ModuleNotFoundError:  # Permit CPU-only decision-rule tests.
    torch = None
from .o10_tasks import (
    LayerwiseTraceTask,
    full_state_floats,
    make_task,
    sketch_state_floats,
    stable_seed,
    task_audit,
    trace_cosine,
)


PROJECT = "N08+O10 Adaptive Stacked Right Subspace"
VERSION = "1.0"
BASE_LENGTH = 128
CANDIDATE = "stacked_right_subspace"
LAYERWISE_CONTROL = "layerwise_sketch"
GLOBAL_ORACLE = "global_sketch"
PRIMARY_RANK = 8
LATENT_RANKS = (2, 4)
CALIBRATION_HORIZONS = (8, 16, 32)
SCREEN_HORIZONS = (8, 32)
CONFIRMATION_HORIZONS = (8, 16, 32)
BOUNDARY_HORIZONS = (16, 32)
CALIBRATION_TEACHERS = (61, 73)
SCREEN_TEACHERS = (607, 701, 809)
CONFIRMATION_TEACHERS = (907, 1009, 1103, 1201, 1301, 1409, 1511, 1601)
BOUNDARY_TEACHERS = (1709, 1801, 1901)
SCREEN_TRAIN_REPLICATES = (0,)
CONFIRMATION_TRAIN_REPLICATES = (0, 1)
BOUNDARY_TRAIN_REPLICATES = (0,)
REPLAY_BY_HORIZON = {8: "gradient_replay", 16: "gradient_replay", 32: "reservoir_replay"}
CLAIM_BOUNDARY = (
    "Persistent-state counts describe the eligibility representation. The reference code "
    "materializes event factors and uses two passes to select and apply the shared subspace, so "
    "this bridge does not yet establish one-pass online operation, lower peak CUDA memory, or "
    "lower wall time in an end-to-end system."
)


@dataclass(frozen=True)
class Config:
    learning_rates: tuple[float, ...] = (0.001, 0.002, 0.004)
    calibration_steps: int = 160
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


def _stacked_right_subspace_rank_for_budget(
    task: LayerwiseTraceTask, budget: int
) -> int:
    return min(task.dim, budget // (task.layers * task.dim + task.dim))


def make_bridge_task(
    teacher: int,
    latent_rank: int,
    horizon: int,
    *,
    split: str,
) -> LayerwiseTraceTask:
    task_seed = stable_seed("n08-o10-stacked-teacher", split, teacher, latent_rank)
    base = make_task(
        latent_rank,
        train_length=BASE_LENGTH,
        sketch_rank=32,
        seed=task_seed,
    )
    return replace(
        base,
        name=f"{split}_teacher{teacher}_latent{latent_rank:02d}_h{horizon:02d}",
        train_length=BASE_LENGTH * horizon,
    )


def _trace_cosine_torch(estimate: torch.Tensor, exact: torch.Tensor) -> float:
    estimate_flat = estimate.flatten(start_dim=1)
    exact_flat = exact.flatten(start_dim=1)
    numerator = torch.sum(estimate_flat * exact_flat, dim=1)
    denominator = torch.linalg.vector_norm(
        estimate_flat, dim=1
    ) * torch.linalg.vector_norm(exact_flat, dim=1)
    values = numerator / torch.clamp_min(denominator, 1e-12)
    return float(torch.mean(values))


def _trace_norm_ratio_torch(estimate: torch.Tensor, exact: torch.Tensor) -> float:
    estimate_flat = estimate.flatten(start_dim=1)
    exact_flat = exact.flatten(start_dim=1)
    ratio = torch.linalg.vector_norm(estimate_flat, dim=1) / torch.clamp_min(
        torch.linalg.vector_norm(exact_flat, dim=1), 1e-12
    )
    return float(torch.mean(ratio))


def premise_summary(device: torch.device) -> dict[str, object]:
    audits = []
    cosine_rows = []
    teacher_checks = []
    for latent_rank in LATENT_RANKS:
        short = make_bridge_task(17, latent_rank, 1, split="premise")
        long = make_bridge_task(17, latent_rank, 8, split="premise")
        teacher_checks.append(
            all(np.array_equal(a, b) for a, b in zip(short.base_matrices, long.base_matrices))
            and all(
                np.array_equal(a, b)
                for a, b in zip(short.teacher_updates, long.teacher_updates)
            )
        )
        audits.append(task_audit(short, rank=PRIMARY_RANK, n=16, seed=17 + latent_rank))
        for horizon, task in ((1, short), (8, long)):
            batch = evaluation_batches(
                task,
                2,
                stable_seed("n08-o10-premise-batch", latent_rank, horizon),
                0.0,
            )["iid"]
            model = LayerwiseAdapterModel(task).to(device).eval()
            x = torch.as_tensor(batch.x, dtype=torch.float32, device=device)
            _, left, right = model.forward_factors(x)
            exact = full_trace(left, right)
            generator = torch.Generator(device="cpu").manual_seed(9_811 + latent_rank + horizon)
            candidate = approximate_trace(
                CANDIDATE, model, left, right, PRIMARY_RANK, generator
            )
            layerwise = approximate_trace(
                LAYERWISE_CONTROL, model, left, right, PRIMARY_RANK, generator
            )
            oracle = approximate_trace(
                GLOBAL_ORACLE, model, left, right, PRIMARY_RANK, generator
            )
            cosine_rows.append(
                {
                    "latent_rank": latent_rank,
                    "horizon_multiplier": horizon,
                    "sequence_length": task.train_length,
                    "candidate_cosine": _trace_cosine_torch(candidate, exact),
                    "candidate_norm_ratio": _trace_norm_ratio_torch(candidate, exact),
                    "layerwise_cosine": _trace_cosine_torch(layerwise, exact),
                    "global_oracle_cosine": _trace_cosine_torch(oracle, exact),
                }
            )
    example = make_bridge_task(17, 2, 1, split="premise")
    budget = sketch_state_floats(example, PRIMARY_RANK)
    allocated, candidate_used = method_state(example, PRIMARY_RANK, CANDIDATE)
    stacked_rank = _stacked_right_subspace_rank_for_budget(example, budget)
    full_state = full_state_floats(example)
    matched = {
        method: method_state(example, PRIMARY_RANK, method)
        for method in (
            CANDIDATE,
            LAYERWISE_CONTROL,
            GLOBAL_ORACLE,
            "gradient_replay",
            "reservoir_replay",
        )
    }
    checks = {
        "paired_horizons_share_the_same_teacher": all(teacher_checks),
        "manual_gradients_match_finite_differences": max(
            row["maximum_finite_difference_error"] for row in audits
        )
        <= 2e-5,
        "stacked_right_subspace_is_finite_and_informative": min(
            row["candidate_cosine"] for row in cosine_rows
        )
        >= 0.90,
        "stacked_right_subspace_has_no_norm_inflation": max(
            row["candidate_norm_ratio"] for row in cosine_rows
        )
        <= 1.01,
        "stacked_rank_exceeds_equal_per_layer_rank": stacked_rank > PRIMARY_RANK,
        "candidate_uses_no_more_than_layerwise_budget": candidate_used <= budget,
        "candidate_uses_at_most_27_percent_of_full_state": candidate_used / full_state
        <= 0.27,
        "all_controls_receive_the_same_allocated_budget": all(
            allocation == allocated for allocation, _ in matched.values()
        ),
        "all_controls_use_no_more_than_allocated": all(
            used <= allocation for allocation, used in matched.values()
        ),
        "targets_are_finite_with_learning_headroom": all(
            row["all_targets_finite"] and row["initial_student_mse"] >= 1e-4
            for row in audits
        ),
    }
    return {
        "project": PROJECT,
        "version": VERSION,
        "main_question": (
            "Can one layer-preserving stacked eligibility sketch use a fixed state budget "
            "more reliably than equal per-layer sketches under long delayed feedback?"
        ),
        "specific_interrogation": (
            "Combine N08's low-rank eligibility result with O10's cross-layer boundary by "
            "sharing an adaptive right subspace across stacked layer gradients. Compare against "
            "the exact trace, equal-layer sketch, a batch global-SVD information oracle, and "
            "prospectively selected replay at 8x, 16x, and 32x horizons."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "candidate_state": {
            "allocated_floats": allocated,
            "used_floats": candidate_used,
            "fraction_of_full": candidate_used / full_state,
            "stacked_rank": stacked_rank,
            "equal_layer_rank": PRIMARY_RANK,
            "full_trace_floats": full_state,
        },
        "matched_method_states": matched,
        "task_audits": audits,
        "trace_cosines": cosine_rows,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }


def _initial_key(teacher: int, latent_rank: int, horizon: int) -> str:
    return f"teacher{teacher}:latent{latent_rank}:h{horizon}"


def run_one(
    task: LayerwiseTraceTask,
    batches: dict[str, object],
    *,
    teacher: int,
    train_replicate: int,
    horizon: int,
    method: str,
    steps: int,
    learning_rate: float,
    stage: str,
    config: Config,
    device: torch.device,
    checkpoint_root: Path | None = None,
) -> dict[str, object]:
    train_seed = stable_seed(
        "n08-o10-stacked-train",
        stage,
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


def calibrate_candidate(
    *,
    output_dir: Path,
    progress_path: Path,
    config: Config,
    device: torch.device,
) -> tuple[dict[int, float], dict[str, object]]:
    rows = []
    initial = {}
    stage_started = time.perf_counter()
    total_runs = (
        len(CALIBRATION_TEACHERS)
        * len(LATENT_RANKS)
        * len(CALIBRATION_HORIZONS)
        * len(config.learning_rates)
    )
    for teacher in CALIBRATION_TEACHERS:
        for latent_rank in LATENT_RANKS:
            for horizon in CALIBRATION_HORIZONS:
                task = make_bridge_task(teacher, latent_rank, horizon, split="calibration")
                batches = evaluation_batches(
                    task,
                    config.calibration_eval_examples,
                    stable_seed("n08-o10-calibration-eval", teacher, latent_rank, horizon),
                    config.target_noise,
                )
                key = _initial_key(teacher, latent_rank, horizon)
                initial[key] = initial_evaluation(task, batches)
                for learning_rate in config.learning_rates:
                    row = run_one(
                        task,
                        batches,
                        teacher=teacher,
                        train_replicate=0,
                        horizon=horizon,
                        method=CANDIDATE,
                        steps=config.calibration_steps,
                        learning_rate=learning_rate,
                        stage="calibration",
                        config=config,
                        device=device,
                    )
                    rows.append(row)
                    completed = len(rows)
                    elapsed = time.perf_counter() - stage_started
                    _append_jsonl(
                        progress_path,
                        {
                            "event": "completed_run",
                            "stage": "calibration",
                            "teacher": teacher,
                            "latent_rank": latent_rank,
                            "horizon": horizon,
                            "method": CANDIDATE,
                            "learning_rate": learning_rate,
                            "runtime_seconds": row["training"]["runtime_seconds"],
                            "completed": completed,
                            "total": total_runs,
                            "eta_seconds": elapsed / completed * (total_runs - completed),
                        },
                    )
    selected = {}
    score_table = {}
    for horizon in CALIBRATION_HORIZONS:
        scores = []
        for learning_rate in config.learning_rates:
            chosen = [
                row
                for row in rows
                if row["horizon_multiplier"] == horizon
                and row["learning_rate"] == learning_rate
            ]
            normalized = []
            for row in chosen:
                key = _initial_key(row["teacher"], row["latent_rank"], horizon)
                initial_mse = initial[key]["aggregate"]["mean_mse"]
                mse = row["evaluation"]["aggregate"]["mean_mse"]
                normalized.append(mse / max(initial_mse, 1e-12))
            score = {
                "learning_rate": learning_rate,
                "median_normalized_mse": float(np.median(normalized)),
                "mean_normalized_mse": float(np.mean(normalized)),
                "maximum_normalized_mse": float(np.max(normalized)),
            }
            scores.append(score)
        scores.sort(
            key=lambda item: (
                item["maximum_normalized_mse"],
                item["mean_normalized_mse"],
                item["learning_rate"],
            )
        )
        selected[horizon] = float(scores[0]["learning_rate"])
        score_table[str(horizon)] = scores
    checks = {
        "calibration_uses_independent_teachers": set(CALIBRATION_TEACHERS).isdisjoint(
            SCREEN_TEACHERS + CONFIRMATION_TEACHERS
        ),
        "every_horizon_has_a_finite_selected_rate": all(
            np.isfinite(selected[horizon]) for horizon in CALIBRATION_HORIZONS
        ),
        "every_calibration_run_is_finite": all(
            np.isfinite(row["evaluation"]["aggregate"]["mean_mse"]) for row in rows
        ),
    }
    summary = {
        "project": PROJECT,
        "version": VERSION,
        "stage": "candidate_calibration",
        "selection_objective": (
            "Minimize worst normalized averaged-checkpoint MSE, then mean MSE, using only "
            "independent calibration teachers."
        ),
        "learning_rate_grid": list(config.learning_rates),
        "selected_learning_rates": {str(key): value for key, value in selected.items()},
        "score_table": score_table,
        "rows": rows,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }
    _write_json(output_dir / "calibration_summary.json", summary)
    return selected, summary


def _control_learning_rate(method: str, horizon: int) -> float:
    if method == "gradient_replay" and horizon == 32:
        return 0.004
    return 0.002


def run_stage(
    *,
    stage: str,
    teachers: tuple[int, ...],
    train_replicates: tuple[int, ...],
    latent_ranks: tuple[int, ...],
    horizons: tuple[int, ...],
    steps: int,
    candidate_learning_rates: dict[int, float],
    output_dir: Path,
    progress_path: Path,
    config: Config,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    rows = []
    initial = {}
    checkpoint_root = output_dir / "checkpoints"
    stage_started = time.perf_counter()
    runs_per_cell = 5
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
                task = make_bridge_task(teacher, latent_rank, horizon, split=stage)
                batches = evaluation_batches(
                    task,
                    config.primary_eval_examples,
                    stable_seed("n08-o10-primary-eval", stage, teacher, latent_rank, horizon),
                    config.target_noise,
                )
                initial[_initial_key(teacher, latent_rank, horizon)] = initial_evaluation(
                    task, batches
                )
                replay = REPLAY_BY_HORIZON[horizon]
                methods = (
                    "full_trace",
                    CANDIDATE,
                    LAYERWISE_CONTROL,
                    GLOBAL_ORACLE,
                    replay,
                )
                for train_replicate in train_replicates:
                    for method in methods:
                        learning_rate = (
                            candidate_learning_rates[horizon]
                            if method == CANDIDATE
                            else _control_learning_rate(method, horizon)
                        )
                        row = run_one(
                            task,
                            batches,
                            teacher=teacher,
                            train_replicate=train_replicate,
                            horizon=horizon,
                            method=method,
                            steps=steps,
                            learning_rate=learning_rate,
                            stage=stage,
                            config=config,
                            device=device,
                            checkpoint_root=checkpoint_root,
                        )
                        rows.append(row)
                        completed = len(rows)
                        elapsed = time.perf_counter() - stage_started
                        _append_jsonl(
                            progress_path,
                            {
                                "event": "completed_run",
                                "stage": stage,
                                "teacher": teacher,
                                "train_replicate": train_replicate,
                                "latent_rank": latent_rank,
                                "horizon": horizon,
                                "method": method,
                                "runtime_seconds": row["training"]["runtime_seconds"],
                                "completed": completed,
                                "total": total_runs,
                                "eta_seconds": elapsed
                                / completed
                                * (total_runs - completed),
                            },
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


def _gain_retention(initial: float, candidate: float, full: float) -> float:
    return (initial - candidate) / max(initial - full, 1e-12)


def analyze(
    rows: list[dict[str, object]],
    initial: dict[str, dict[str, object]],
    *,
    teachers: tuple[int, ...],
    train_replicates: tuple[int, ...],
    horizons: tuple[int, ...],
    config: Config,
    stage: str,
) -> dict[str, object]:
    lookup = _row_map(rows)
    cells = []
    for teacher in teachers:
        for train_replicate in train_replicates:
            for latent_rank in sorted({int(row["latent_rank"]) for row in rows}):
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
                    available = max(initial_mse - full_mse, 1e-12)
                    cells.append(
                        {
                            "teacher": teacher,
                            "train_replicate": train_replicate,
                            "latent_rank": latent_rank,
                            "horizon_multiplier": horizon,
                            "sequence_length": BASE_LENGTH * horizon,
                            "selected_replay": replay_method,
                            "initial_mse": initial_mse,
                            "full_trace_mse": full_mse,
                            "candidate_mse": candidate_mse,
                            "layerwise_mse": layerwise_mse,
                            "global_oracle_mse": oracle_mse,
                            "replay_mse": replay_mse,
                            "candidate_endpoint_mse": endpoint_mse,
                            "full_learning_fraction": (initial_mse - full_mse)
                            / max(initial_mse, 1e-12),
                            "candidate_full_gain_retention": _gain_retention(
                                initial_mse, candidate_mse, full_mse
                            ),
                            "candidate_gain_vs_layerwise": (
                                layerwise_mse - candidate_mse
                            )
                            / available,
                            "candidate_gain_vs_global_oracle": (
                                oracle_mse - candidate_mse
                            )
                            / available,
                            "candidate_gain_vs_replay": (replay_mse - candidate_mse)
                            / available,
                            "averaging_gain": (endpoint_mse - candidate_mse) / available,
                            "candidate_gradient_cosine": float(
                                candidate["training"]["mean_gradient_cosine"]
                            ),
                        }
                    )
    horizon_aggregates = []
    for horizon in horizons:
        selected = [cell for cell in cells if cell["horizon_multiplier"] == horizon]
        horizon_aggregates.append(
            {
                "horizon_multiplier": horizon,
                "sequence_length": BASE_LENGTH * horizon,
                "full_learning_fraction": float(
                    np.mean([cell["full_learning_fraction"] for cell in selected])
                ),
                "candidate_full_gain_retention": float(
                    np.mean([cell["candidate_full_gain_retention"] for cell in selected])
                ),
                "candidate_gain_vs_layerwise": float(
                    np.mean([cell["candidate_gain_vs_layerwise"] for cell in selected])
                ),
                "candidate_gain_vs_global_oracle": float(
                    np.mean([cell["candidate_gain_vs_global_oracle"] for cell in selected])
                ),
                "candidate_gain_vs_replay": float(
                    np.mean([cell["candidate_gain_vs_replay"] for cell in selected])
                ),
                "candidate_gradient_cosine": float(
                    np.mean([cell["candidate_gradient_cosine"] for cell in selected])
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
                    np.mean([cell["candidate_gain_vs_global_oracle"] for cell in selected])
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
            stable_seed("n08-o10-bootstrap", stage, field),
        )
        for field in fields
    }
    candidate_row = next(row for row in rows if row["method"] == CANDIDATE)
    full_row = next(row for row in rows if row["method"] == "full_trace")
    candidate_state = int(candidate_row["training"]["used_state_floats_per_episode"])
    allocated_state = int(
        candidate_row["training"]["allocated_state_floats_per_episode"]
    )
    full_state = int(full_row["training"]["used_state_floats_per_episode"])
    matched_parameters = len({int(row["training"]["parameters"]) for row in rows}) == 1
    matched_steps = len({int(row["training"]["optimizer_steps"]) for row in rows}) == 1
    matched_allocations = all(
        int(row["training"]["allocated_state_floats_per_episode"]) == allocated_state
        for row in rows
        if row["method"] != "full_trace"
    )
    collapse_fraction = float(
        np.mean([cell["candidate_full_gain_retention"] < 0.80 for cell in cells])
    )
    reference_checks = {
        "full_trace_learns_overall": float(
            np.mean([cell["full_learning_fraction"] for cell in cells])
        )
        >= config.minimum_full_learning_overall,
        "full_trace_learns_at_every_horizon": min(
            row["full_learning_fraction"] for row in horizon_aggregates
        )
        >= config.minimum_full_learning_each_horizon,
        "full_trace_has_minimum_cell_headroom": min(
            cell["full_learning_fraction"] for cell in cells
        )
        >= config.minimum_full_learning_each_cell,
    }
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
        "candidate_respects_state_ceiling": candidate_state / full_state
        <= config.maximum_candidate_state_fraction,
        "all_compressed_methods_receive_matched_allocations": matched_allocations,
        "parameters_are_matched": matched_parameters,
        "optimizer_steps_are_matched": matched_steps,
        "weight_averaging_is_present": min(
            int(row["training"]["averaged_steps"]) for row in rows
        )
        > 0,
    }
    strong_checks = {
        "candidate_layerwise_gain_is_at_least_0_01": bootstrap[
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
        "reference_is_valid": all(reference_checks.values()),
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
        "comparisons_are_matched": matched_allocations
        and matched_parameters
        and matched_steps,
    }
    reference_valid = all(reference_checks.values())
    supportive = reference_valid and all(core_checks.values())
    strong_positive = supportive and all(strong_checks.values())
    continue_after_screen = all(screen_checks.values())
    return {
        "stage": stage,
        "horizons": list(horizons),
        "state": {
            "candidate_used_floats": candidate_state,
            "allocated_floats": allocated_state,
            "full_trace_floats": full_state,
            "candidate_fraction_of_full": candidate_state / full_state,
            "stacked_rank": _stacked_right_subspace_rank_for_budget(
                make_bridge_task(17, 2, 1, split="analysis"), allocated_state
            ),
        },
        "cells": cells,
        "horizon_aggregates": horizon_aggregates,
        "teacher_aggregates": teacher_aggregates,
        "bootstrap_over_independent_teachers": bootstrap,
        "collapse_fraction": collapse_fraction,
        "wins": {
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
        },
        "reference_checks": reference_checks,
        "core_checks": core_checks,
        "strong_checks": strong_checks,
        "screen_checks": screen_checks,
        "reference_valid": reference_valid,
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
    screen_summary = None
    confirmation_summary = None
    boundary_summary = None
    candidate_learning_rates = None
    if not premise["gate_passed"]:
        decision = "stop_n08_o10_stacked_after_premise"
    else:
        candidate_learning_rates, calibration = calibrate_candidate(
            output_dir=output_dir,
            progress_path=progress_path,
            config=config,
            device=device,
        )
        if not calibration["gate_passed"]:
            decision = "stop_n08_o10_stacked_after_calibration"
        else:
            screen_rows, screen_initial = run_stage(
                stage="screen",
                teachers=SCREEN_TEACHERS,
                train_replicates=SCREEN_TRAIN_REPLICATES,
                latent_ranks=LATENT_RANKS,
                horizons=SCREEN_HORIZONS,
                steps=config.screen_steps,
                candidate_learning_rates=candidate_learning_rates,
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
            if not screen_analysis["continue_after_screen"]:
                decision = "stop_stacked_trace_after_decisive_screen_preserve_n08_core"
            else:
                confirmation_rows, confirmation_initial = run_stage(
                    stage="confirmation",
                    teachers=CONFIRMATION_TEACHERS,
                    train_replicates=CONFIRMATION_TRAIN_REPLICATES,
                    latent_ranks=LATENT_RANKS,
                    horizons=CONFIRMATION_HORIZONS,
                    steps=config.confirmation_steps,
                    candidate_learning_rates=candidate_learning_rates,
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
                    output_dir / "confirmation_summary.json", confirmation_summary
                )
                if confirmation_analysis["supportive"]:
                    boundary_rows, boundary_initial = run_stage(
                        stage="boundary",
                        teachers=BOUNDARY_TEACHERS,
                        train_replicates=BOUNDARY_TRAIN_REPLICATES,
                        latent_ranks=(8, 12),
                        horizons=BOUNDARY_HORIZONS,
                        steps=config.boundary_steps,
                        candidate_learning_rates=candidate_learning_rates,
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
                        config=config,
                        stage="boundary",
                    )
                    boundary_summary = _stage_summary(
                        stage="boundary",
                        rows=boundary_rows,
                        initial=boundary_initial,
                        analysis=boundary_analysis,
                    )
                    _write_json(output_dir / "boundary_summary.json", boundary_summary)
                if confirmation_analysis["strong_positive"]:
                    decision = (
                        "stacked_trace_strong_positive_proceed_to_external_benchmarks"
                    )
                elif confirmation_analysis["supportive"]:
                    decision = (
                        "stacked_trace_viable_without_specific_advantage_use_n08_positive_"
                        "and_o10_boundary"
                    )
                else:
                    decision = (
                        "stacked_trace_not_supported_preserve_n08_positive_and_o10_boundary"
                    )
    confirmation_analysis = (
        confirmation_summary["analysis"] if confirmation_summary is not None else None
    )
    summary = {
        "project": PROJECT,
        "version": VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "frozen_evidence_entering_bridge": {
            "n08_v2": (
                "rank-four streaming low-rank trace passed all gates at 34.03% of full "
                "persistent state and beat matched replay at long horizons"
            ),
            "o10_v2": (
                "rank-eight equal-layer trace retained about 99% of full-trace gain but "
                "did not establish a reliable advantage over the same-budget global oracle"
            ),
        },
        "frozen_priors_before_bridge": {
            "p_bridge_positive": 0.55,
            "p_big3_given_positive": 0.45,
            "p_jmlr_given_positive": 0.62,
            "p_tmlr_given_positive": 0.82,
        },
        "config": asdict(config),
        "premise_passed": premise["gate_passed"],
        "calibration_passed": calibration is not None and calibration["gate_passed"],
        "screen_ran": screen_summary is not None,
        "screen_continue": screen_summary is not None
        and screen_summary["analysis"]["continue_after_screen"],
        "confirmation_ran": confirmation_summary is not None,
        "confirmation_supportive": confirmation_analysis is not None
        and confirmation_analysis["supportive"],
        "strong_positive": confirmation_analysis is not None
        and confirmation_analysis["strong_positive"],
        "boundary_ran": boundary_summary is not None,
        "candidate_learning_rates": (
            {str(key): value for key, value in candidate_learning_rates.items()}
            if candidate_learning_rates is not None
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
    parser.add_argument("--screen-steps", type=int, default=Config.screen_steps)
    parser.add_argument("--confirmation-steps", type=int, default=Config.confirmation_steps)
    parser.add_argument("--boundary-steps", type=int, default=Config.boundary_steps)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument(
        "--primary-eval-examples", type=int, default=Config.primary_eval_examples
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch is None:
        raise RuntimeError("PyTorch is required to run the N08+O10 cascade")
    config = Config(
        calibration_steps=args.calibration_steps,
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
