from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

try:
    import torch

    from .n08_traces import predict, train_method
except ModuleNotFoundError:  # Permit CPU-only reference and decision tests.
    torch = None
    predict = None
    train_method = None

from .n08_tasks import (
    DelayedOutcomeTask,
    full_reference_traces,
    make_task,
    sample_batch,
    task_audit,
    top_rank_energy,
    trace_state_floats,
    zero_student_scores,
)
PROJECT = "N08 Low-Rank Eligibility Traces for Delayed Outcomes"
VERSION = "2.0"
BASE_LENGTH = 24
HORIZONS = (1, 4, 8)
PRIMARY_LATENT_RANKS = (2, 4)
TRACE_RANKS = (1, 2, 4)
METHODS = ("full_trace", "lowrank_svd", "replay_reservoir", "multi_timescale")
COMPRESSED_METHODS = METHODS[1:]
CALIBRATION_SEEDS = (61, 73)
PRIMARY_SEEDS = (101, 211, 307, 401, 503)
FRONTIER_SEEDS = PRIMARY_SEEDS[:2]
BOUNDARY_SEEDS = (607, 701, 809)
OOD_CONDITIONS = ("noise_shift", "scale_shift", "joint_shift")


@dataclass(frozen=True)
class Config:
    calibration_steps: int = 250
    primary_steps: int = 550
    frontier_steps: int = 550
    boundary_steps: int = 550
    batch_size: int = 64
    calibration_eval_examples: int = 256
    primary_eval_examples: int = 512
    learning_rates: tuple[float, ...] = (0.003, 0.01, 0.03)
    weight_decay: float = 0.0001
    target_noise: float = 0.003
    bootstrap_samples: int = 5000
    minimum_full_learning_fraction: float = 0.50
    minimum_horizon_full_learning_fraction: float = 0.35
    minimum_mean_gain_retention: float = 0.97
    minimum_gain_retention_lower_95: float = 0.94
    minimum_task_horizon_retention: float = 0.90
    maximum_replay_excess_upper_95: float = 0.03
    maximum_strongest_control_excess_upper_95: float = 0.05
    minimum_mean_gradient_cosine: float = 0.90
    minimum_horizon_gradient_cosine: float = 0.85
    minimum_strong_pareto_gain: float = 0.03
    decisive_failure_retention: float = 0.80
    decisive_failure_cosine: float = 0.75


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _method_key(method: str, rank: int) -> str:
    return method if method == "full_trace" else f"{method}:r{rank}"


def _method_configs(rank: int, *, include_full: bool = True) -> tuple[tuple[str, int], ...]:
    values: list[tuple[str, int]] = []
    if include_full:
        values.append(("full_trace", 0))
    values.extend((method, rank) for method in COMPRESSED_METHODS)
    return tuple(values)


def make_v2_task(
    latent_rank: int,
    horizon: int,
    *,
    task_seed: int,
    split: str,
) -> DelayedOutcomeTask:
    task = make_task(
        latent_rank,
        train_length=BASE_LENGTH * horizon,
        seed=task_seed,
    )
    return replace(
        task,
        name=f"{split}_latent_{latent_rank:02d}_horizon_{horizon:02d}x",
    )


def primary_tasks() -> tuple[DelayedOutcomeTask, ...]:
    return tuple(
        make_v2_task(rank, horizon, task_seed=208_083, split="primary")
        for rank in PRIMARY_LATENT_RANKS
        for horizon in HORIZONS
    )


def development_tasks() -> tuple[DelayedOutcomeTask, ...]:
    return tuple(
        make_v2_task(2, horizon, task_seed=108_083, split="development")
        for horizon in HORIZONS
    )


def boundary_tasks() -> tuple[DelayedOutcomeTask, ...]:
    return tuple(
        make_v2_task(8, horizon, task_seed=308_083, split="boundary")
        for horizon in (4, 8)
    )


def _horizon(task: DelayedOutcomeTask) -> int:
    return task.train_length // BASE_LENGTH


def evaluation_batches(
    task: DelayedOutcomeTask,
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
            input_noise=0.12,
            target_noise=target_noise,
        ),
        "scale_shift": sample_batch(
            task,
            n,
            seed + 2,
            latent_scale=1.40,
            target_noise=target_noise,
        ),
        "joint_shift": sample_batch(
            task,
            n,
            seed + 3,
            input_noise=0.12,
            latent_scale=1.40,
            target_noise=target_noise,
        ),
    }


def initial_evaluation(
    task: DelayedOutcomeTask,
    batches: dict[str, object],
) -> dict[str, object]:
    rows: dict[str, object] = {}
    for name, batch in batches.items():
        prediction = zero_student_scores(task, batch.x).mean(axis=1)
        rows[name] = {
            "terminal_mse": float(np.mean((prediction - batch.target) ** 2)),
            "terminal_mae": float(np.mean(np.abs(prediction - batch.target))),
        }
    rows["aggregate"] = {
        "mean_mse": float(np.mean([rows[name]["terminal_mse"] for name in batches])),
        "ood_mean_mse": float(
            np.mean([rows[name]["terminal_mse"] for name in OOD_CONDITIONS])
        ),
    }
    return rows


def evaluate_weights(
    task: DelayedOutcomeTask,
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
        }
    rows["aggregate"] = {
        "mean_mse": float(np.mean([rows[name]["terminal_mse"] for name in batches])),
        "ood_mean_mse": float(
            np.mean([rows[name]["terminal_mse"] for name in OOD_CONDITIONS])
        ),
    }
    return rows


def run_one(
    task: DelayedOutcomeTask,
    batches: dict[str, object],
    *,
    method: str,
    rank: int,
    seed: int,
    steps: int,
    learning_rate: float,
    config: Config,
    device: torch.device,
) -> dict[str, object]:
    effective_rank = 2 if method == "full_trace" else rank
    trained = train_method(
        task,
        method=method,
        seed=seed,
        rank=effective_rank,
        steps=steps,
        batch_size=config.batch_size,
        learning_rate=learning_rate,
        weight_decay=config.weight_decay,
        device=device,
        target_noise=config.target_noise,
    )
    return {
        "task": task.name,
        "latent_rank": task.latent_rank,
        "horizon_multiplier": _horizon(task),
        "sequence_length": task.train_length,
        "method": method,
        "trace_rank": 0 if method == "full_trace" else rank,
        "method_key": _method_key(method, rank),
        "seed": seed,
        "learning_rate": learning_rate,
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


def premise_summary_v2() -> dict[str, object]:
    tasks = primary_tasks()
    audits = [task_audit(task, n=64, seed=83) for task in tasks]
    reference = tasks[0]
    same_teacher_across_horizons = True
    for latent_rank in PRIMARY_LATENT_RANKS:
        family = [task for task in tasks if task.latent_rank == latent_rank]
        for task in family[1:]:
            same_teacher_across_horizons &= all(
                np.array_equal(left, right)
                for left, right in zip(family[0].teacher_updates, task.teacher_updates)
            )
            same_teacher_across_horizons &= all(
                np.array_equal(left, right)
                for left, right in zip(family[0].base_matrices, task.base_matrices)
            )
    state_ladder = []
    full_state = trace_state_floats(reference, 2, "full_trace")
    for rank in TRACE_RANKS:
        candidate = trace_state_floats(reference, rank, "lowrank_svd")
        replay = trace_state_floats(reference, rank, "replay_reservoir")
        multiscale = trace_state_floats(reference, rank, "multi_timescale")
        state_ladder.append(
            {
                "rank": rank,
                "full_trace_floats": full_state,
                "candidate_floats": candidate,
                "candidate_fraction_of_full": candidate / full_state,
                "replay_floats": replay,
                "multi_timescale_floats": multiscale,
                "controls_use_no_more_state_than_candidate": max(replay, multiscale)
                <= candidate,
            }
        )
    coverage_rows = []
    for task in tasks:
        batch = sample_batch(task, 32, 88, target_noise=0.0)
        traces = full_reference_traces(task, batch.x)
        coverage_rows.append(
            {
                "task": task.name,
                "rank_1_energy": float(
                    np.mean([np.mean(top_rank_energy(trace, 1)) for trace in traces])
                ),
                "rank_2_energy": float(
                    np.mean([np.mean(top_rank_energy(trace, 2)) for trace in traces])
                ),
                "rank_4_energy": float(
                    np.mean([np.mean(top_rank_energy(trace, 4)) for trace in traces])
                ),
            }
        )
    checks = {
        "fresh_v2_task_family_is_finite": all(item["all_targets_finite"] for item in audits),
        "manual_trace_matches_finite_differences": max(
            item["maximum_finite_difference_error"] for item in audits
        )
        <= 2e-5,
        "teacher_is_identical_across_horizons": same_teacher_across_horizons,
        "rank_two_uses_at_most_0_18_of_full_state": state_ladder[1][
            "candidate_fraction_of_full"
        ]
        <= 0.18,
        "rank_four_uses_at_most_0_35_of_full_state": state_ladder[2][
            "candidate_fraction_of_full"
        ]
        <= 0.35,
        "all_controls_use_no_more_state_than_candidate": all(
            row["controls_use_no_more_state_than_candidate"] for row in state_ladder
        ),
        "primary_rank_two_energy_exceeds_0_55": min(
            row["rank_2_energy"] for row in coverage_rows
        )
        >= 0.55,
        "terminal_target_uses_both_sequence_halves": min(
            item["mean_absolute_first_vs_second_half_contribution_difference"]
            for item in audits
        )
        >= 0.005,
        "v2_seeds_do_not_reuse_v1_seeds": not set(PRIMARY_SEEDS).intersection(
            {7, 19, 31, 43, 59}
        ),
    }
    return {
        "project": PROJECT,
        "version": VERSION,
        "main_question": (
            "Can a small low-rank eligibility trace preserve delayed learning while using "
            "far less persistent state, especially as the delay gets longer?"
        ),
        "specific_interrogation": (
            "Train at 1x, 4x, and 8x sequence lengths, calibrate each method separately, "
            "and compare ranks 1, 2, and 4 against the exact trace and two controls that "
            "receive no more state."
        ),
        "frozen_v1_result": {
            "original_gate_passed": False,
            "candidate_state_fraction": 0.1701388888888889,
            "candidate_full_gain_retention": 0.9977202790629626,
            "candidate_relative_gain_over_replay": -0.004520576017656549,
            "interpretation": (
                "V1 remains a failure of its preregistered 10% superiority gate. V2 asks a "
                "different, efficiency-focused question and does not retroactively relabel V1."
            ),
        },
        "task_audits": audits,
        "spectral_coverage": coverage_rows,
        "state_ladder": state_ladder,
        "checks": checks,
        "gate_passed": all(checks.values()),
        "implementation_scope": (
            "State counts measure the persistent eligibility representation. The current "
            "reference implementation materializes event factors for benchmarking, so peak "
            "end-to-end GPU memory is not yet a publication claim."
        ),
    }


def calibrate(
    output_dir: Path,
    progress_path: Path,
    config: Config,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, object]]:
    rows: list[dict[str, object]] = []
    tasks = development_tasks()
    configurations = [("full_trace", 0)] + [
        (method, rank) for rank in TRACE_RANKS for method in COMPRESSED_METHODS
    ]
    total = len(tasks) * len(CALIBRATION_SEEDS) * len(configurations) * len(
        config.learning_rates
    )
    completed = 0
    started = time.perf_counter()
    for task_index, task in enumerate(tasks):
        for seed in CALIBRATION_SEEDS:
            batches = evaluation_batches(
                task,
                config.calibration_eval_examples,
                1_100_000 + task_index * 10_000 + seed,
                config.target_noise,
            )
            for method, rank in configurations:
                for learning_rate in config.learning_rates:
                    error = None
                    row = None
                    try:
                        row = run_one(
                            task,
                            batches,
                            method=method,
                            rank=rank,
                            seed=seed,
                            steps=config.calibration_steps,
                            learning_rate=learning_rate,
                            config=config,
                            device=device,
                        )
                        if not np.isfinite(row["evaluation"]["aggregate"]["ood_mean_mse"]):
                            error = "nonfinite_validation_metric"
                            row = None
                    except (RuntimeError, ValueError) as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    if row is None:
                        row = {
                            "task": task.name,
                            "latent_rank": task.latent_rank,
                            "horizon_multiplier": _horizon(task),
                            "sequence_length": task.train_length,
                            "method": method,
                            "trace_rank": rank,
                            "method_key": _method_key(method, rank),
                            "seed": seed,
                            "learning_rate": learning_rate,
                            "error": error,
                        }
                    rows.append(row)
                    completed += 1
                    elapsed = time.perf_counter() - started
                    _append_jsonl(
                        progress_path,
                        {
                            "stage": "calibration",
                            "completed": completed,
                            "total": total,
                            "elapsed_seconds": elapsed,
                            "eta_seconds": elapsed / completed * (total - completed),
                            "method_key": _method_key(method, rank),
                            "horizon_multiplier": _horizon(task),
                            "learning_rate": learning_rate,
                            "error": error,
                        },
                    )
    selected: dict[str, float] = {}
    selection_rows = []
    for horizon in HORIZONS:
        for method, rank in configurations:
            key = _method_key(method, rank)
            candidates = []
            for learning_rate in config.learning_rates:
                values = [
                    float(row["evaluation"]["aggregate"]["ood_mean_mse"])
                    for row in rows
                    if row.get("horizon_multiplier") == horizon
                    and row.get("method_key") == key
                    and row.get("learning_rate") == learning_rate
                    and "evaluation" in row
                ]
                if len(values) == len(CALIBRATION_SEEDS):
                    candidates.append((float(np.mean(values)), learning_rate, values))
            if candidates:
                mean_mse, learning_rate, values = min(candidates, key=lambda item: (item[0], item[1]))
                selected[f"h{horizon}:{key}"] = learning_rate
                selection_rows.append(
                    {
                        "horizon_multiplier": horizon,
                        "method_key": key,
                        "selected_learning_rate": learning_rate,
                        "mean_validation_ood_mse": mean_mse,
                        "seed_validation_ood_mse": values,
                    }
                )
    expected = len(HORIZONS) * len(configurations)
    checks = {
        "every_method_rank_horizon_has_a_finite_selection": len(selected) == expected,
        "both_development_seeds_contribute_to_every_selection": all(
            len(row["seed_validation_ood_mse"]) == len(CALIBRATION_SEEDS)
            for row in selection_rows
        ),
        "calibration_uses_no_primary_seed": not set(CALIBRATION_SEEDS).intersection(
            PRIMARY_SEEDS
        ),
    }
    summary = {
        "project": PROJECT,
        "version": VERSION,
        "stage": "separate-development optimizer calibration",
        "specific_interrogation": (
            "Give every method and trace rank the same three learning-rate choices on separate "
            "development data, then freeze the best setting before the five-seed comparison."
        ),
        "rows": rows,
        "selections": selection_rows,
        "selected_learning_rates": selected,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }
    _write_json(output_dir / "calibration_summary.json", summary)
    return selected, summary


def run_stage(
    *,
    stage: str,
    tasks: tuple[DelayedOutcomeTask, ...],
    seeds: tuple[int, ...],
    method_configs: tuple[tuple[str, int], ...],
    steps: int,
    eval_examples: int,
    selected_learning_rates: dict[str, float],
    config: Config,
    device: torch.device,
    progress_path: Path,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    initial: dict[str, dict[str, object]] = {}
    total = len(tasks) * len(seeds) * len(method_configs)
    completed = 0
    started = time.perf_counter()
    for task_index, task in enumerate(tasks):
        horizon = _horizon(task)
        for seed in seeds:
            batches = evaluation_batches(
                task,
                eval_examples,
                2_100_000 + task_index * 20_000 + seed,
                config.target_noise,
            )
            initial[f"{task.name}:{seed}"] = initial_evaluation(task, batches)
            for method, rank in method_configs:
                key = _method_key(method, rank)
                learning_rate = selected_learning_rates[f"h{horizon}:{key}"]
                row = run_one(
                    task,
                    batches,
                    method=method,
                    rank=rank,
                    seed=seed,
                    steps=steps,
                    learning_rate=learning_rate,
                    config=config,
                    device=device,
                )
                rows.append(row)
                completed += 1
                elapsed = time.perf_counter() - started
                _append_jsonl(
                    progress_path,
                    {
                        "stage": stage,
                        "completed": completed,
                        "total": total,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": elapsed / completed * (total - completed),
                        "task": task.name,
                        "seed": seed,
                        "method_key": key,
                        "ood_mean_mse": row["evaluation"]["aggregate"]["ood_mean_mse"],
                    },
                )
    return rows, initial


def _bootstrap_mean(values: list[float], samples: int, seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0 or not np.isfinite(array).all():
        return {"mean": float("nan"), "lower_95": float("nan"), "upper_95": float("nan")}
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(samples, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
    }


def _row_lookup(
    rows: list[dict[str, object]],
    task: str,
    seed: int,
    method: str,
    rank: int,
) -> dict[str, object]:
    matches = [
        row
        for row in rows
        if row["task"] == task
        and row["seed"] == seed
        and row["method"] == method
        and (method == "full_trace" or row["trace_rank"] == rank)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one row for task={task}, seed={seed}, method={method}, rank={rank}; "
            f"found {len(matches)}"
        )
    return matches[0]


def rank_analysis(
    rows: list[dict[str, object]],
    initial: dict[str, dict[str, object]],
    *,
    rank: int,
    seeds: tuple[int, ...],
    config: Config,
) -> dict[str, object]:
    task_names = sorted({row["task"] for row in rows})
    cells = []
    for task_name in task_names:
        example = next(row for row in rows if row["task"] == task_name)
        for seed in seeds:
            full = _row_lookup(rows, task_name, seed, "full_trace", rank)
            candidate = _row_lookup(rows, task_name, seed, "lowrank_svd", rank)
            replay = _row_lookup(rows, task_name, seed, "replay_reservoir", rank)
            multi = _row_lookup(rows, task_name, seed, "multi_timescale", rank)
            initial_mse = float(initial[f"{task_name}:{seed}"]["aggregate"]["ood_mean_mse"])
            full_mse = float(full["evaluation"]["aggregate"]["ood_mean_mse"])
            candidate_mse = float(candidate["evaluation"]["aggregate"]["ood_mean_mse"])
            replay_mse = float(replay["evaluation"]["aggregate"]["ood_mean_mse"])
            multi_mse = float(multi["evaluation"]["aggregate"]["ood_mean_mse"])
            strongest = min(replay_mse, multi_mse)
            full_gain = initial_mse - full_mse
            cells.append(
                {
                    "task": task_name,
                    "latent_rank": example["latent_rank"],
                    "horizon_multiplier": example["horizon_multiplier"],
                    "seed": seed,
                    "initial_ood_mse": initial_mse,
                    "full_trace_ood_mse": full_mse,
                    "candidate_ood_mse": candidate_mse,
                    "replay_ood_mse": replay_mse,
                    "multi_timescale_ood_mse": multi_mse,
                    "full_learning_fraction": (initial_mse - full_mse)
                    / max(initial_mse, 1e-12),
                    "candidate_full_gain_retention": (initial_mse - candidate_mse)
                    / max(full_gain, 1e-12),
                    "candidate_relative_excess_vs_replay": (candidate_mse - replay_mse)
                    / max(replay_mse, 1e-12),
                    "candidate_relative_excess_vs_strongest_control": (
                        candidate_mse - strongest
                    )
                    / max(strongest, 1e-12),
                    "candidate_gradient_cosine": candidate["training"][
                        "mean_gradient_cosine"
                    ],
                }
            )
    seed_rows = []
    for seed in seeds:
        selected = [row for row in cells if row["seed"] == seed]
        seed_rows.append(
            {
                "seed": seed,
                "full_learning_fraction": float(
                    np.mean([row["full_learning_fraction"] for row in selected])
                ),
                "candidate_full_gain_retention": float(
                    np.mean([row["candidate_full_gain_retention"] for row in selected])
                ),
                "candidate_relative_excess_vs_replay": float(
                    np.mean([row["candidate_relative_excess_vs_replay"] for row in selected])
                ),
                "candidate_relative_excess_vs_strongest_control": float(
                    np.mean(
                        [
                            row["candidate_relative_excess_vs_strongest_control"]
                            for row in selected
                        ]
                    )
                ),
                "candidate_gradient_cosine": float(
                    np.mean([row["candidate_gradient_cosine"] for row in selected])
                ),
            }
        )
    bootstrap = {
        field: _bootstrap_mean(
            [row[field] for row in seed_rows],
            config.bootstrap_samples,
            808_200 + index,
        )
        for index, field in enumerate(
            (
                "full_learning_fraction",
                "candidate_full_gain_retention",
                "candidate_relative_excess_vs_replay",
                "candidate_relative_excess_vs_strongest_control",
                "candidate_gradient_cosine",
            )
        )
    }
    horizon_rows = []
    for horizon in sorted({row["horizon_multiplier"] for row in cells}):
        selected = [row for row in cells if row["horizon_multiplier"] == horizon]
        horizon_rows.append(
            {
                "horizon_multiplier": horizon,
                "full_learning_fraction": float(
                    np.mean([row["full_learning_fraction"] for row in selected])
                ),
                "candidate_full_gain_retention": float(
                    np.mean([row["candidate_full_gain_retention"] for row in selected])
                ),
                "candidate_relative_gain_over_replay": float(
                    -np.mean([row["candidate_relative_excess_vs_replay"] for row in selected])
                ),
                "candidate_relative_gain_over_strongest_control": float(
                    -np.mean(
                        [row["candidate_relative_excess_vs_strongest_control"] for row in selected]
                    )
                ),
                "candidate_gradient_cosine": float(
                    np.mean([row["candidate_gradient_cosine"] for row in selected])
                ),
            }
        )
    task_horizon_rows = []
    for latent_rank in sorted({row["latent_rank"] for row in cells}):
        for horizon in sorted({row["horizon_multiplier"] for row in cells}):
            selected = [
                row
                for row in cells
                if row["latent_rank"] == latent_rank
                and row["horizon_multiplier"] == horizon
            ]
            if not selected:
                continue
            task_horizon_rows.append(
                {
                    "latent_rank": latent_rank,
                    "horizon_multiplier": horizon,
                    "candidate_full_gain_retention": float(
                        np.mean([row["candidate_full_gain_retention"] for row in selected])
                    ),
                    "candidate_relative_gain_over_replay": float(
                        -np.mean(
                            [row["candidate_relative_excess_vs_replay"] for row in selected]
                        )
                    ),
                }
            )
    candidate_row = next(
        row for row in rows if row["method"] == "lowrank_svd" and row["trace_rank"] == rank
    )
    full_row = next(row for row in rows if row["method"] == "full_trace")
    candidate_state = int(candidate_row["training"]["state_floats_per_episode"])
    full_state = int(full_row["training"]["state_floats_per_episode"])
    maximum_state_fraction = {1: 0.10, 2: 0.20, 4: 0.36}[rank]
    matched_parameters = len({row["training"]["parameters"] for row in rows}) == 1
    matched_steps = len({row["training"]["optimizer_steps"] for row in rows}) == 1
    checks = {
        "full_trace_learns_at_least_half_of_available_error": bootstrap[
            "full_learning_fraction"
        ]["mean"]
        >= config.minimum_full_learning_fraction,
        "full_trace_learns_at_every_horizon": min(
            row["full_learning_fraction"] for row in horizon_rows
        )
        >= config.minimum_horizon_full_learning_fraction,
        "candidate_mean_gain_retention_is_at_least_0_97": bootstrap[
            "candidate_full_gain_retention"
        ]["mean"]
        >= config.minimum_mean_gain_retention,
        "candidate_gain_retention_lower_95_is_at_least_0_94": bootstrap[
            "candidate_full_gain_retention"
        ]["lower_95"]
        >= config.minimum_gain_retention_lower_95,
        "every_task_horizon_retains_at_least_0_90": min(
            row["candidate_full_gain_retention"] for row in task_horizon_rows
        )
        >= config.minimum_task_horizon_retention,
        "candidate_is_noninferior_to_replay_with_0_03_margin": bootstrap[
            "candidate_relative_excess_vs_replay"
        ]["upper_95"]
        <= config.maximum_replay_excess_upper_95,
        "candidate_is_noninferior_to_strongest_control_with_0_05_margin": bootstrap[
            "candidate_relative_excess_vs_strongest_control"
        ]["upper_95"]
        <= config.maximum_strongest_control_excess_upper_95,
        "candidate_mean_gradient_cosine_is_at_least_0_90": bootstrap[
            "candidate_gradient_cosine"
        ]["mean"]
        >= config.minimum_mean_gradient_cosine,
        "candidate_gradient_cosine_is_at_least_0_85_at_every_horizon": min(
            row["candidate_gradient_cosine"] for row in horizon_rows
        )
        >= config.minimum_horizon_gradient_cosine,
        "candidate_respects_rank_specific_state_budget": candidate_state / full_state
        <= maximum_state_fraction,
        "parameters_are_matched": matched_parameters,
        "optimizer_steps_are_matched": matched_steps,
    }
    long_cells = [row for row in cells if row["horizon_multiplier"] >= 4]
    long_seed_gains = []
    for seed in seeds:
        selected = [row for row in long_cells if row["seed"] == seed]
        long_seed_gains.append(
            float(-np.mean([row["candidate_relative_excess_vs_replay"] for row in selected]))
        )
    long_bootstrap = _bootstrap_mean(
        long_seed_gains, config.bootstrap_samples, 808_299 + rank
    )
    strong_cell_count = sum(
        row["candidate_relative_gain_over_replay"] >= config.minimum_strong_pareto_gain
        for row in task_horizon_rows
        if row["horizon_multiplier"] >= 4
    )
    strong_pareto = (
        long_bootstrap["mean"] >= config.minimum_strong_pareto_gain
        and long_bootstrap["lower_95"] > 0.0
        and strong_cell_count >= 2
    )
    gate_passed = all(checks.values())
    reference_valid = checks["full_trace_learns_at_least_half_of_available_error"] and checks[
        "full_trace_learns_at_every_horizon"
    ]
    rescue_warranted = (
        not gate_passed
        and reference_valid
        and bootstrap["candidate_full_gain_retention"]["mean"]
        >= config.decisive_failure_retention
        and bootstrap["candidate_gradient_cosine"]["mean"]
        >= config.decisive_failure_cosine
        and checks["parameters_are_matched"]
        and checks["optimizer_steps_are_matched"]
    )
    decisive_failure = not gate_passed and reference_valid and not rescue_warranted
    return {
        "trace_rank": rank,
        "state": {
            "candidate_floats": candidate_state,
            "full_trace_floats": full_state,
            "candidate_fraction_of_full": candidate_state / full_state,
        },
        "cells": cells,
        "seed_aggregates": seed_rows,
        "bootstrap_over_seed_aggregates": bootstrap,
        "horizon_aggregates": horizon_rows,
        "task_horizon_aggregates": task_horizon_rows,
        "long_horizon_gain_over_replay_bootstrap": long_bootstrap,
        "strong_long_horizon_cells": strong_cell_count,
        "checks": checks,
        "gate_passed": gate_passed,
        "strong_pareto_signal": strong_pareto,
        "rank_four_rescue_warranted": rank == 2 and rescue_warranted,
        "decisive_failure": decisive_failure,
        "reference_invalid": not reference_valid,
    }


def run(output_dir: Path, device: torch.device, config: Config) -> dict[str, object]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=False)
    progress_path = output_dir / "progress.jsonl"
    premise = premise_summary_v2()
    _write_json(output_dir / "premise_summary.json", premise)
    _append_jsonl(progress_path, {"stage": "premise", "gate_passed": premise["gate_passed"]})
    if not premise["gate_passed"]:
        summary = {
            "project": PROJECT,
            "version": VERSION,
            "decision": "repair_n08_v2_reference_before_inference",
            "premise_passed": False,
            "gate_passed": False,
            "runtime_seconds": time.perf_counter() - started,
            "config": asdict(config),
        }
        _write_json(output_dir / "cascade_summary.json", summary)
        return summary

    selected_lrs, calibration = calibrate(output_dir, progress_path, config, device)
    if not calibration["gate_passed"]:
        summary = {
            "project": PROJECT,
            "version": VERSION,
            "decision": "repair_n08_v2_calibration_before_inference",
            "premise_passed": True,
            "calibration_passed": False,
            "gate_passed": False,
            "runtime_seconds": time.perf_counter() - started,
            "config": asdict(config),
        }
        _write_json(output_dir / "cascade_summary.json", summary)
        return summary

    primary_rows, primary_initial = run_stage(
        stage="five_seed_rank_two_primary",
        tasks=primary_tasks(),
        seeds=PRIMARY_SEEDS,
        method_configs=_method_configs(2),
        steps=config.primary_steps,
        eval_examples=config.primary_eval_examples,
        selected_learning_rates=selected_lrs,
        config=config,
        device=device,
        progress_path=progress_path,
    )
    primary_gate = rank_analysis(
        primary_rows,
        primary_initial,
        rank=2,
        seeds=PRIMARY_SEEDS,
        config=config,
    )
    primary_summary = {
        "project": PROJECT,
        "version": VERSION,
        "stage": "five-seed rank-two accuracy-memory-delay frontier",
        "main_question": premise["main_question"],
        "specific_interrogation": (
            "At 17% of full-trace state, test whether the calibrated low-rank trace stays "
            "near the exact learner and statistically noninferior to replay across 1x, 4x, "
            "and 8x learning delays."
        ),
        "initial_evaluations": primary_initial,
        "rows": primary_rows,
        "gate": primary_gate,
        "gate_passed": primary_gate["gate_passed"],
    }
    _write_json(output_dir / "primary_summary.json", primary_summary)

    selected_rank: int | None = 2 if primary_gate["gate_passed"] else None
    rescue_summary = None
    frontier_summary = None
    if primary_gate["rank_four_rescue_warranted"]:
        rescue_rows, _ = run_stage(
            stage="rank_four_capacity_rescue",
            tasks=primary_tasks(),
            seeds=PRIMARY_SEEDS,
            method_configs=_method_configs(4, include_full=False),
            steps=config.primary_steps,
            eval_examples=config.primary_eval_examples,
            selected_learning_rates=selected_lrs,
            config=config,
            device=device,
            progress_path=progress_path,
        )
        combined = primary_rows + rescue_rows
        rescue_gate = rank_analysis(
            combined,
            primary_initial,
            rank=4,
            seeds=PRIMARY_SEEDS,
            config=config,
        )
        rescue_summary = {
            "project": PROJECT,
            "version": VERSION,
            "stage": "automatic rank-four capacity rescue",
            "specific_interrogation": (
                "If rank two is informative but misses the noninferiority bar, double its rank "
                "under the preregistered 35% state ceiling before declaring the mechanism dead."
            ),
            "rows": rescue_rows,
            "gate": rescue_gate,
            "gate_passed": rescue_gate["gate_passed"],
        }
        _write_json(output_dir / "rescue_summary.json", rescue_summary)
        if rescue_gate["gate_passed"]:
            selected_rank = 4
    elif primary_gate["gate_passed"]:
        frontier_rows, _ = run_stage(
            stage="rank_ladder_characterization",
            tasks=primary_tasks(),
            seeds=FRONTIER_SEEDS,
            method_configs=tuple(
                (method, rank)
                for rank in (1, 4)
                for method in COMPRESSED_METHODS
            ),
            steps=config.frontier_steps,
            eval_examples=config.primary_eval_examples,
            selected_learning_rates=selected_lrs,
            config=config,
            device=device,
            progress_path=progress_path,
        )
        selected_primary = [row for row in primary_rows if row["seed"] in FRONTIER_SEEDS]
        combined = selected_primary + frontier_rows
        rank_one = rank_analysis(
            combined,
            {key: value for key, value in primary_initial.items() if int(key.rsplit(":", 1)[1]) in FRONTIER_SEEDS},
            rank=1,
            seeds=FRONTIER_SEEDS,
            config=config,
        )
        rank_four = rank_analysis(
            combined,
            {key: value for key, value in primary_initial.items() if int(key.rsplit(":", 1)[1]) in FRONTIER_SEEDS},
            rank=4,
            seeds=FRONTIER_SEEDS,
            config=config,
        )
        frontier_summary = {
            "project": PROJECT,
            "version": VERSION,
            "stage": "two-seed rank-one/rank-four frontier characterization",
            "specific_interrogation": (
                "After rank two passes, map whether halving or doubling the trace state gives a "
                "smooth accuracy-memory tradeoff rather than a one-budget accident."
            ),
            "rows": frontier_rows,
            "rank_one_analysis": rank_one,
            "rank_two_primary_analysis": primary_gate,
            "rank_four_analysis": rank_four,
            "informational_only": True,
        }
        _write_json(output_dir / "frontier_summary.json", frontier_summary)

    boundary_summary = None
    if selected_rank is not None:
        boundary_rows, boundary_initial = run_stage(
            stage="higher_intrinsic_rank_boundary",
            tasks=boundary_tasks(),
            seeds=BOUNDARY_SEEDS,
            method_configs=_method_configs(selected_rank),
            steps=config.boundary_steps,
            eval_examples=config.primary_eval_examples,
            selected_learning_rates=selected_lrs,
            config=config,
            device=device,
            progress_path=progress_path,
        )
        boundary_gate = rank_analysis(
            boundary_rows,
            boundary_initial,
            rank=selected_rank,
            seeds=BOUNDARY_SEEDS,
            config=config,
        )
        boundary_retention = boundary_gate["bootstrap_over_seed_aggregates"][
            "candidate_full_gain_retention"
        ]["mean"]
        boundary_excess = boundary_gate["bootstrap_over_seed_aggregates"][
            "candidate_relative_excess_vs_strongest_control"
        ]["mean"]
        boundary_supportive = boundary_retention >= 0.90 and boundary_excess <= 0.10
        boundary_noncatastrophic = boundary_retention >= 0.70 and boundary_excess <= 0.20
        boundary_summary = {
            "project": PROJECT,
            "version": VERSION,
            "stage": "three-seed latent-rank-eight boundary",
            "specific_interrogation": (
                "Raise the task's true gradient rank to eight and ask whether the selected "
                "compressed trace degrades gradually instead of failing unpredictably."
            ),
            "selected_trace_rank": selected_rank,
            "initial_evaluations": boundary_initial,
            "rows": boundary_rows,
            "analysis": boundary_gate,
            "boundary_supportive": boundary_supportive,
            "boundary_noncatastrophic": boundary_noncatastrophic,
        }
        _write_json(output_dir / "boundary_summary.json", boundary_summary)

    if selected_rank is not None:
        strong_signal = primary_gate["strong_pareto_signal"] or bool(
            rescue_summary and rescue_summary["gate"]["strong_pareto_signal"]
        )
        decision = (
            "n08_v2_strong_pareto_positive_continue_to_o10"
            if strong_signal
            else "n08_v2_compression_positive_continue_to_o10"
        )
        gate_passed = True
    elif primary_gate["reference_invalid"]:
        decision = "n08_v2_inconclusive_reference_failed_repair_before_inference"
        gate_passed = False
    else:
        decision = "stop_n08_v2_move_to_o10"
        gate_passed = False
    summary = {
        "project": PROJECT,
        "version": VERSION,
        "decision": decision,
        "premise_passed": True,
        "calibration_passed": True,
        "primary_passed": primary_gate["gate_passed"],
        "rank_four_rescue_ran": rescue_summary is not None,
        "rank_four_rescue_passed": bool(rescue_summary and rescue_summary["gate_passed"]),
        "frontier_ran": frontier_summary is not None,
        "boundary_ran": boundary_summary is not None,
        "boundary_supportive": (
            None if boundary_summary is None else boundary_summary["boundary_supportive"]
        ),
        "selected_trace_rank": selected_rank,
        "strong_pareto_signal": (
            primary_gate["strong_pareto_signal"]
            or bool(rescue_summary and rescue_summary["gate"]["strong_pareto_signal"])
        ),
        "gate_passed": gate_passed,
        "runtime_seconds": time.perf_counter() - started,
        "config": asdict(config),
        "frozen_priors_before_v2": {
            "p_positive": 0.67,
            "p_big3_given_positive": 0.38,
            "p_tmlr_given_positive": 0.72,
        },
    }
    _write_json(output_dir / "cascade_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration-steps", type=int, default=Config.calibration_steps)
    parser.add_argument("--primary-steps", type=int, default=Config.primary_steps)
    parser.add_argument("--frontier-steps", type=int, default=Config.frontier_steps)
    parser.add_argument("--boundary-steps", type=int, default=Config.boundary_steps)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    config = Config(
        calibration_steps=args.calibration_steps,
        primary_steps=args.primary_steps,
        frontier_steps=args.frontier_steps,
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
                    "primary_passed",
                    "rank_four_rescue_ran",
                    "rank_four_rescue_passed",
                    "frontier_ran",
                    "boundary_ran",
                    "selected_trace_rank",
                    "strong_pareto_signal",
                    "gate_passed",
                    "runtime_seconds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
