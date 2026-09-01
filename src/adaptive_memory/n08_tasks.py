from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DelayedOutcomeTask:
    name: str
    dim: int
    layers: int
    latent_rank: int
    train_length: int
    input_noise: float
    adapter_scale: float
    base_matrices: tuple[np.ndarray, ...]
    teacher_updates: tuple[np.ndarray, ...]
    readout: np.ndarray
    probe_matrix: np.ndarray
    input_basis: np.ndarray


@dataclass(frozen=True)
class DelayedOutcomeBatch:
    x: np.ndarray
    target: np.ndarray
    event_scores: np.ndarray


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _orthogonal(rng: np.random.Generator, dim: int) -> np.ndarray:
    matrix = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return (q * signs).astype(np.float32)


def make_task(
    latent_rank: int,
    *,
    dim: int = 24,
    layers: int = 3,
    train_length: int = 24,
    seed: int = 80_083,
) -> DelayedOutcomeTask:
    if not 1 <= latent_rank <= dim:
        raise ValueError("latent_rank must lie between one and dim")
    rng = np.random.default_rng(_stable_seed("n08-task", seed, latent_rank, dim, layers))
    basis_raw = rng.normal(size=(dim, latent_rank))
    basis, _ = np.linalg.qr(basis_raw)
    base_matrices = tuple(0.92 * _orthogonal(rng, dim) for _ in range(layers))
    update_rank = min(max(2, latent_rank), 6)
    teacher_updates = []
    for _ in range(layers):
        left = rng.normal(size=(dim, update_rank))
        right = rng.normal(size=(dim, update_rank))
        update = left @ right.T / np.sqrt(dim * update_rank)
        spectral = max(float(np.linalg.norm(update, ord=2)), 1e-8)
        teacher_updates.append((1.15 * update / spectral).astype(np.float32))
    readout = rng.normal(size=dim)
    readout = (readout / max(float(np.linalg.norm(readout)), 1e-8)).astype(np.float32)
    probe_matrix = (1.10 * _orthogonal(rng, dim)).astype(np.float32)
    return DelayedOutcomeTask(
        name=f"latent_rank_{latent_rank:02d}",
        dim=dim,
        layers=layers,
        latent_rank=latent_rank,
        train_length=train_length,
        input_noise=0.035,
        adapter_scale=0.75,
        base_matrices=base_matrices,
        teacher_updates=tuple(teacher_updates),
        readout=readout,
        probe_matrix=probe_matrix,
        input_basis=basis.astype(np.float32),
    )


def task_suite() -> dict[str, DelayedOutcomeTask]:
    tasks = (make_task(2), make_task(4))
    return {task.name: task for task in tasks}


def boundary_suite() -> dict[str, DelayedOutcomeTask]:
    tasks = (make_task(8), make_task(12))
    return {task.name: task for task in tasks}


def _forward_scores(
    task: DelayedOutcomeTask,
    x: np.ndarray,
    updates: tuple[np.ndarray, ...],
) -> np.ndarray:
    h = np.asarray(x, dtype=np.float64)
    for base, update in zip(task.base_matrices, updates):
        matrix = np.asarray(base, dtype=np.float64) + task.adapter_scale * np.asarray(
            update, dtype=np.float64
        )
        h = np.tanh(np.einsum("...j,ij->...i", h, matrix))
    probe = np.tanh(
        np.einsum("...j,ij->...i", np.asarray(x, dtype=np.float64), task.probe_matrix)
    )
    return (
        np.einsum("...i,...i->...", h, probe) / np.sqrt(task.dim)
        + 0.15 * np.einsum("...i,i->...", h, task.readout)
    )


def sample_batch(
    task: DelayedOutcomeTask,
    n: int,
    seed: int,
    *,
    length: int | None = None,
    input_noise: float | None = None,
    latent_scale: float = 1.0,
    target_noise: float = 0.01,
) -> DelayedOutcomeBatch:
    length = task.train_length if length is None else int(length)
    input_noise = task.input_noise if input_noise is None else float(input_noise)
    if length < 2:
        raise ValueError("delayed-outcome sequences must contain at least two events")
    rng = np.random.default_rng(
        _stable_seed(task.name, n, seed, length, input_noise, latent_scale, target_noise)
    )
    latent = rng.normal(
        0.0, latent_scale, size=(n, length, task.latent_rank)
    )
    x = np.einsum("btk,dk->btd", latent, task.input_basis)
    if input_noise:
        x = x + rng.normal(0.0, input_noise, size=x.shape)
    event_scores = _forward_scores(task, x, task.teacher_updates)
    target = event_scores.mean(axis=1)
    if target_noise:
        target = target + rng.normal(0.0, target_noise, size=target.shape)
    return DelayedOutcomeBatch(
        x=x.astype(np.float32),
        target=target.astype(np.float32),
        event_scores=event_scores.astype(np.float32),
    )


def zero_student_scores(task: DelayedOutcomeTask, x: np.ndarray) -> np.ndarray:
    zeros = tuple(np.zeros_like(matrix) for matrix in task.base_matrices)
    return _forward_scores(task, x, zeros).astype(np.float32)


def reference_factors(
    task: DelayedOutcomeTask,
    x: np.ndarray,
    updates: tuple[np.ndarray, ...] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    if updates is None:
        updates = tuple(np.zeros_like(matrix) for matrix in task.base_matrices)
    h = np.asarray(x, dtype=np.float64)
    states = [h]
    matrices = []
    for base, update in zip(task.base_matrices, updates):
        matrix = np.asarray(base, dtype=np.float64) + task.adapter_scale * np.asarray(
            update, dtype=np.float64
        )
        matrices.append(matrix)
        h = np.tanh(np.einsum("...j,ij->...i", h, matrix))
        states.append(h)
    probe = np.tanh(
        np.einsum("...j,ij->...i", np.asarray(x, dtype=np.float64), task.probe_matrix)
    )
    score = (
        np.einsum("...i,...i->...", h, probe) / np.sqrt(task.dim)
        + 0.15 * np.einsum("...i,i->...", h, task.readout)
    )
    delta = probe / np.sqrt(task.dim) + 0.15 * np.asarray(
        task.readout, dtype=np.float64
    )
    delta = delta * (1.0 - states[-1] ** 2)
    left_factors: list[np.ndarray] = [np.empty(0)] * task.layers
    right_factors: list[np.ndarray] = [np.empty(0)] * task.layers
    for layer in reversed(range(task.layers)):
        left_factors[layer] = task.adapter_scale * delta
        right_factors[layer] = states[layer]
        if layer:
            delta = np.einsum("...i,ij->...j", delta, matrices[layer])
            delta = delta * (1.0 - states[layer] ** 2)
    return score, tuple(left_factors), tuple(right_factors)


def full_reference_traces(
    task: DelayedOutcomeTask,
    x: np.ndarray,
    updates: tuple[np.ndarray, ...] | None = None,
) -> tuple[np.ndarray, ...]:
    _, left, right = reference_factors(task, x, updates)
    return tuple(
        np.einsum("bti,btj->bij", a, b) / x.shape[1]
        for a, b in zip(left, right)
    )


def top_rank_energy(matrix: np.ndarray, rank: int) -> np.ndarray:
    singular = np.linalg.svd(matrix, compute_uv=False)
    squared = singular**2
    return squared[..., :rank].sum(axis=-1) / np.maximum(squared.sum(axis=-1), 1e-12)


def trace_state_floats(task: DelayedOutcomeTask, rank: int, method: str) -> int:
    full = task.layers * task.dim * task.dim
    factor = task.layers * rank * (2 * task.dim + 1)
    if method == "full_trace":
        return full
    if method in {
        "lowrank_svd",
        "recent_factors",
        "reservoir_factors",
    }:
        return factor
    if method == "multi_timescale":
        return task.layers * rank * 2 * task.dim
    if method == "replay_reservoir":
        slots = max(1, factor // (task.dim + 1))
        return slots * (task.dim + 1)
    raise ValueError(method)


def task_audit(task: DelayedOutcomeTask, n: int = 192, seed: int = 17) -> dict[str, object]:
    batch = sample_batch(task, n, seed, target_noise=0.0)
    traces = full_reference_traces(task, batch.x)
    coverage = [float(np.mean(top_rank_energy(trace, 2))) for trace in traces]
    zero_prediction = zero_student_scores(task, batch.x).mean(axis=1)
    first_half = batch.event_scores[:, : task.train_length // 2].mean(axis=1)
    second_half = batch.event_scores[:, task.train_length // 2 :].mean(axis=1)
    finite_difference_errors = []
    epsilon = 1e-4
    zeros = [np.zeros_like(matrix) for matrix in task.base_matrices]
    for layer in range(task.layers):
        row = (3 * layer + 1) % task.dim
        column = (5 * layer + 2) % task.dim
        plus = [value.copy() for value in zeros]
        minus = [value.copy() for value in zeros]
        plus[layer][row, column] += epsilon
        minus[layer][row, column] -= epsilon
        plus_value = _forward_scores(task, batch.x[:8], tuple(plus)).mean(axis=1)
        minus_value = _forward_scores(task, batch.x[:8], tuple(minus)).mean(axis=1)
        numeric = (plus_value - minus_value) / (2 * epsilon)
        analytic = traces[layer][:8, row, column]
        finite_difference_errors.append(float(np.max(np.abs(numeric - analytic))))
    rank_two_state = trace_state_floats(task, 2, "lowrank_svd")
    full_state = trace_state_floats(task, 2, "full_trace")
    return {
        "task": task.name,
        "latent_rank": task.latent_rank,
        "sequence_length": task.train_length,
        "examples": n,
        "initial_student_mse": float(np.mean((zero_prediction - batch.target) ** 2)),
        "mean_absolute_first_vs_second_half_contribution_difference": float(
            np.mean(np.abs(first_half - second_half))
        ),
        "top_rank_two_energy_by_layer": coverage,
        "mean_top_rank_two_energy": float(np.mean(coverage)),
        "finite_difference_max_errors": finite_difference_errors,
        "maximum_finite_difference_error": max(finite_difference_errors),
        "full_trace_state_floats_per_episode": full_state,
        "rank_two_state_floats_per_episode": rank_two_state,
        "rank_two_state_fraction": rank_two_state / full_state,
        "all_targets_finite": bool(np.isfinite(batch.target).all()),
        "only_terminal_aggregate_target_is_exposed": True,
    }


def premise_summary() -> dict[str, object]:
    primary = [task_audit(task) for task in task_suite().values()]
    boundary = [task_audit(task) for task in boundary_suite().values()]
    primary_coverage = float(
        np.mean([item["mean_top_rank_two_energy"] for item in primary])
    )
    low_rank_coverage = float(primary[0]["mean_top_rank_two_energy"])
    high_rank_coverage = float(boundary[-1]["mean_top_rank_two_energy"])
    all_audits = primary + boundary
    checks = {
        "manual_trace_matches_finite_differences": max(
            item["maximum_finite_difference_error"] for item in all_audits
        )
        <= 2e-5,
        "primary_trace_top_two_energy_is_at_least_0_55": primary_coverage >= 0.55,
        "rank_two_coverage_declines_by_at_least_0_10_at_high_latent_rank": (
            low_rank_coverage - high_rank_coverage
        )
        >= 0.10,
        "terminal_target_depends_on_events_from_both_halves": min(
            item["mean_absolute_first_vs_second_half_contribution_difference"]
            for item in all_audits
        )
        >= 0.01,
        "zero_student_has_nontrivial_learning_headroom": min(
            item["initial_student_mse"] for item in primary
        )
        >= 1e-4,
        "rank_two_trace_uses_at_most_0_25_of_full_state": max(
            item["rank_two_state_fraction"] for item in all_audits
        )
        <= 0.25,
        "all_targets_are_finite": all(item["all_targets_finite"] for item in all_audits),
    }
    return {
        "project": "N08 Low-Rank Eligibility Traces for Delayed Outcomes",
        "version": "1.0",
        "main_question": (
            "Can a network learn from a score that arrives after a long sequence while storing "
            "only a small low-rank summary of which weights were involved?"
        ),
        "specific_interrogation": (
            "Before training, verify that exact layerwise eligibility matrices are compressible, "
            "that their effective rank changes in a controlled boundary family, and that the "
            "manual online trace equals the delayed terminal-loss gradient."
        ),
        "primary_task_audits": primary,
        "boundary_task_audits": boundary,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }
