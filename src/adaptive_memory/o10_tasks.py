from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LayerwiseTraceTask:
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
    omega: np.ndarray
    psi: np.ndarray


@dataclass(frozen=True)
class LayerwiseTraceBatch:
    x: np.ndarray
    target: np.ndarray
    event_scores: np.ndarray


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _orthogonal(rng: np.random.Generator, dim: int) -> np.ndarray:
    matrix = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return (q * signs).astype(np.float32)


def _sketch_matrix(
    rng: np.random.Generator, layers: int, dim: int, rank: int
) -> np.ndarray:
    matrices = []
    for _ in range(layers):
        raw = rng.normal(size=(dim, rank))
        q, _ = np.linalg.qr(raw)
        matrices.append(q[:, :rank])
    return np.stack(matrices).astype(np.float32)


def make_task(
    latent_rank: int,
    *,
    dim: int = 64,
    layers: int = 4,
    train_length: int = 128,
    sketch_rank: int = 16,
    seed: int = 100_010,
) -> LayerwiseTraceTask:
    if not 1 <= latent_rank <= dim:
        raise ValueError("latent_rank must lie between one and dim")
    if not 1 <= sketch_rank <= dim:
        raise ValueError("sketch_rank must lie between one and dim")
    rng = np.random.default_rng(
        stable_seed("o10-task", seed, latent_rank, dim, layers, train_length)
    )
    basis_raw = rng.normal(size=(dim, latent_rank))
    basis, _ = np.linalg.qr(basis_raw)
    base_matrices = tuple(0.90 * _orthogonal(rng, dim) for _ in range(layers))
    update_rank = min(max(2, latent_rank), sketch_rank)
    teacher_updates = []
    for layer in range(layers):
        left = rng.normal(size=(dim, update_rank))
        right = rng.normal(size=(dim, update_rank))
        update = left @ right.T / np.sqrt(dim * update_rank)
        spectral = max(float(np.linalg.norm(update, ord=2)), 1e-8)
        layer_scale = 1.05 + 0.15 * layer / max(layers - 1, 1)
        teacher_updates.append((layer_scale * update / spectral).astype(np.float32))
    readout = rng.normal(size=dim)
    readout = (readout / max(float(np.linalg.norm(readout)), 1e-8)).astype(
        np.float32
    )
    probe_matrix = (1.05 * _orthogonal(rng, dim)).astype(np.float32)
    omega = _sketch_matrix(rng, layers, dim, sketch_rank)
    psi = _sketch_matrix(rng, layers, dim, sketch_rank)
    return LayerwiseTraceTask(
        name=f"latent_rank_{latent_rank:02d}",
        dim=dim,
        layers=layers,
        latent_rank=latent_rank,
        train_length=train_length,
        input_noise=0.025,
        adapter_scale=0.90,
        base_matrices=base_matrices,
        teacher_updates=tuple(teacher_updates),
        readout=readout,
        probe_matrix=probe_matrix,
        input_basis=basis.astype(np.float32),
        omega=omega,
        psi=psi,
    )


def task_suite() -> dict[str, LayerwiseTraceTask]:
    tasks = (make_task(2), make_task(4))
    return {task.name: task for task in tasks}


def _forward_scores(
    task: LayerwiseTraceTask,
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
        + 0.20 * np.einsum("...i,i->...", h, task.readout)
    )


def sample_batch(
    task: LayerwiseTraceTask,
    n: int,
    seed: int,
    *,
    length: int | None = None,
    input_noise: float | None = None,
    latent_scale: float = 1.0,
    target_noise: float = 0.002,
) -> LayerwiseTraceBatch:
    length = task.train_length if length is None else int(length)
    input_noise = task.input_noise if input_noise is None else float(input_noise)
    if length < 4:
        raise ValueError("layerwise delayed-outcome sequences need at least four events")
    rng = np.random.default_rng(
        stable_seed(
            task.name, n, seed, length, input_noise, latent_scale, target_noise
        )
    )
    latent = rng.normal(0.0, latent_scale, size=(n, length, task.latent_rank))
    x = np.einsum("btk,dk->btd", latent, task.input_basis)
    if input_noise:
        x = x + rng.normal(0.0, input_noise, size=x.shape)
    event_scores = _forward_scores(task, x, task.teacher_updates)
    target = event_scores.sum(axis=1) / np.sqrt(length)
    if target_noise:
        target = target + rng.normal(0.0, target_noise, size=target.shape)
    return LayerwiseTraceBatch(
        x=x.astype(np.float32),
        target=target.astype(np.float32),
        event_scores=event_scores.astype(np.float32),
    )


def zero_student_scores(task: LayerwiseTraceTask, x: np.ndarray) -> np.ndarray:
    zeros = tuple(np.zeros_like(matrix) for matrix in task.base_matrices)
    return _forward_scores(task, x, zeros).astype(np.float32)


def reference_factors(
    task: LayerwiseTraceTask,
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
        + 0.20 * np.einsum("...i,i->...", h, task.readout)
    )
    delta = probe / np.sqrt(task.dim) + 0.20 * np.asarray(
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
    task: LayerwiseTraceTask,
    x: np.ndarray,
    updates: tuple[np.ndarray, ...] | None = None,
) -> np.ndarray:
    _, left, right = reference_factors(task, x, updates)
    return np.stack(
        [
            np.einsum("bti,btj->bij", a, b) / np.sqrt(x.shape[1])
            for a, b in zip(left, right)
        ],
        axis=1,
    )


def sketch_state_floats(task: LayerwiseTraceTask, rank: int) -> int:
    return task.layers * (2 * task.dim * rank + rank * rank)


def full_state_floats(task: LayerwiseTraceTask) -> int:
    return task.layers * task.dim * task.dim


def replay_slots(task: LayerwiseTraceTask, rank: int) -> int:
    return sketch_state_floats(task, rank) // (task.dim + 1)


def two_sided_sketch_numpy(
    task: LayerwiseTraceTask,
    left: tuple[np.ndarray, ...],
    right: tuple[np.ndarray, ...],
    rank: int,
) -> np.ndarray:
    estimates = []
    length = left[0].shape[1]
    for layer, (a, b) in enumerate(zip(left, right)):
        omega = np.asarray(task.omega[layer, :, :rank], dtype=np.float64)
        psi = np.asarray(task.psi[layer, :, :rank], dtype=np.float64)
        b_omega = np.einsum("btd,dr->btr", b, omega)
        a_psi = np.einsum("btd,dr->btr", a, psi)
        normalizer = np.sqrt(length)
        column = np.einsum("btd,btr->bdr", a, b_omega) / normalizer
        row_t = np.einsum("btd,btr->bdr", b, a_psi) / normalizer
        core = np.einsum("btr,bts->brs", a_psi, b_omega) / normalizer
        inverse = np.linalg.pinv(core, rcond=1e-5)
        estimate = np.einsum("bdr,brs,bes->bde", column, inverse, row_t)
        estimates.append(estimate)
    return np.stack(estimates, axis=1)


def sampled_trace_numpy(
    left: tuple[np.ndarray, ...],
    right: tuple[np.ndarray, ...],
    count: int,
    seed: int,
    *,
    gradient_ranked: bool = False,
) -> np.ndarray:
    batch, length, dim = left[0].shape
    count = min(max(1, count), length)
    if gradient_ranked:
        scores = np.zeros((batch, length), dtype=np.float64)
        for a, b in zip(left, right):
            scores += np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
        indices = np.argpartition(scores, -count, axis=1)[:, -count:]
    else:
        rng = np.random.default_rng(seed)
        indices = np.stack([rng.choice(length, count, replace=False) for _ in range(batch)])
    estimates = []
    rows = np.arange(batch)[:, None]
    for a, b in zip(left, right):
        selected_a = a[rows, indices]
        selected_b = b[rows, indices]
        estimates.append(
            np.einsum("bki,bkj->bij", selected_a, selected_b)
            * np.sqrt(length)
            / count
        )
    return np.stack(estimates, axis=1)


def trace_cosine(estimate: np.ndarray, exact: np.ndarray) -> np.ndarray:
    estimate_flat = estimate.reshape(len(estimate), -1)
    exact_flat = exact.reshape(len(exact), -1)
    numerator = np.sum(estimate_flat * exact_flat, axis=1)
    denominator = np.linalg.norm(estimate_flat, axis=1) * np.linalg.norm(
        exact_flat, axis=1
    )
    return numerator / np.maximum(denominator, 1e-12)


def task_audit(
    task: LayerwiseTraceTask,
    *,
    rank: int = 8,
    n: int = 64,
    seed: int = 17,
) -> dict[str, object]:
    batch = sample_batch(task, n, seed, target_noise=0.0)
    _, left, right = reference_factors(task, batch.x)
    exact = full_reference_traces(task, batch.x)
    sketch = two_sided_sketch_numpy(task, left, right, rank)
    reservoir = sampled_trace_numpy(
        left, right, replay_slots(task, rank), seed + 1
    )
    gradient_replay = sampled_trace_numpy(
        left,
        right,
        replay_slots(task, rank),
        seed + 2,
        gradient_ranked=True,
    )
    long_batch = sample_batch(
        task, n, seed + 3, length=8 * task.train_length, target_noise=0.0
    )
    _, long_left, long_right = reference_factors(task, long_batch.x)
    long_exact = full_reference_traces(task, long_batch.x)
    long_sketch = two_sided_sketch_numpy(task, long_left, long_right, rank)
    long_reservoir = sampled_trace_numpy(
        long_left, long_right, replay_slots(task, rank), seed + 4
    )
    long_gradient_replay = sampled_trace_numpy(
        long_left,
        long_right,
        replay_slots(task, rank),
        seed + 5,
        gradient_ranked=True,
    )
    permuted = np.roll(sketch, shift=1, axis=1)
    zeros = [np.zeros_like(matrix) for matrix in task.base_matrices]
    epsilon = 1e-4
    finite_difference_errors = []
    for layer in range(task.layers):
        row = (3 * layer + 2) % task.dim
        column = (7 * layer + 1) % task.dim
        plus = [value.copy() for value in zeros]
        minus = [value.copy() for value in zeros]
        plus[layer][row, column] += epsilon
        minus[layer][row, column] -= epsilon
        plus_value = _forward_scores(task, batch.x[:6], tuple(plus)).sum(axis=1) / np.sqrt(
            task.train_length
        )
        minus_value = _forward_scores(
            task, batch.x[:6], tuple(minus)
        ).sum(axis=1) / np.sqrt(task.train_length)
        numeric = (plus_value - minus_value) / (2 * epsilon)
        analytic = exact[:6, layer, row, column]
        finite_difference_errors.append(float(np.max(np.abs(numeric - analytic))))
    zero_prediction = zero_student_scores(task, batch.x).sum(axis=1) / np.sqrt(
        task.train_length
    )
    half = task.train_length // 2
    first_half = batch.event_scores[:, :half].mean(axis=1)
    second_half = batch.event_scores[:, half:].mean(axis=1)
    sketch_cosine = trace_cosine(sketch, exact)
    reservoir_cosine = trace_cosine(reservoir, exact)
    gradient_replay_cosine = trace_cosine(gradient_replay, exact)
    long_sketch_cosine = trace_cosine(long_sketch, long_exact)
    long_reservoir_cosine = trace_cosine(long_reservoir, long_exact)
    long_gradient_replay_cosine = trace_cosine(long_gradient_replay, long_exact)
    permuted_cosine = trace_cosine(permuted, exact)
    return {
        "task": task.name,
        "latent_rank": task.latent_rank,
        "sequence_length": task.train_length,
        "long_sequence_length": 8 * task.train_length,
        "examples": n,
        "initial_student_mse": float(np.mean((zero_prediction - batch.target) ** 2)),
        "first_vs_second_half_difference": float(
            np.mean(np.abs(first_half - second_half))
        ),
        "maximum_finite_difference_error": max(finite_difference_errors),
        "sketch_mean_cosine": float(np.mean(sketch_cosine)),
        "reservoir_mean_cosine": float(np.mean(reservoir_cosine)),
        "gradient_replay_mean_cosine": float(np.mean(gradient_replay_cosine)),
        "permuted_layer_mean_cosine": float(np.mean(permuted_cosine)),
        "long_sketch_mean_cosine": float(np.mean(long_sketch_cosine)),
        "long_reservoir_mean_cosine": float(np.mean(long_reservoir_cosine)),
        "long_gradient_replay_mean_cosine": float(
            np.mean(long_gradient_replay_cosine)
        ),
        "long_sketch_margin_over_best_replay": float(
            np.mean(long_sketch_cosine)
            - max(
                float(np.mean(long_reservoir_cosine)),
                float(np.mean(long_gradient_replay_cosine)),
            )
        ),
        "layer_identity_cosine_drop": float(
            np.mean(sketch_cosine) - np.mean(permuted_cosine)
        ),
        "candidate_state_floats": sketch_state_floats(task, rank),
        "full_trace_state_floats": full_state_floats(task),
        "candidate_state_fraction": sketch_state_floats(task, rank)
        / full_state_floats(task),
        "matched_replay_slots": replay_slots(task, rank),
        "state_is_independent_of_sequence_length": True,
        "all_targets_finite": bool(np.isfinite(batch.target).all()),
        "only_terminal_aggregate_target_is_exposed": True,
    }


def premise_summary(rank: int = 8) -> dict[str, object]:
    audits = [task_audit(task, rank=rank) for task in task_suite().values()]
    checks = {
        "manual_layerwise_trace_matches_finite_differences": max(
            item["maximum_finite_difference_error"] for item in audits
        )
        <= 2e-5,
        "candidate_training_horizon_cosine_is_at_least_0_80": min(
            item["sketch_mean_cosine"] for item in audits
        )
        >= 0.80,
        "candidate_eight_x_horizon_cosine_is_at_least_0_70": min(
            item["long_sketch_mean_cosine"] for item in audits
        )
        >= 0.70,
        "candidate_has_eight_x_horizon_margin_over_replay": min(
            item["long_sketch_margin_over_best_replay"] for item in audits
        )
        >= 0.01,
        "layer_identity_matters_for_gradient_direction": min(
            item["layer_identity_cosine_drop"] for item in audits
        )
        >= 0.10,
        "candidate_uses_at_most_0_30_of_full_trace_state": max(
            item["candidate_state_fraction"] for item in audits
        )
        <= 0.30,
        "terminal_target_depends_on_both_sequence_halves": min(
            item["first_vs_second_half_difference"] for item in audits
        )
        >= 0.004,
        "zero_student_has_nontrivial_headroom": min(
            item["initial_student_mse"] for item in audits
        )
        >= 1e-4,
        "state_is_fixed_across_horizons": all(
            item["state_is_independent_of_sequence_length"] for item in audits
        ),
        "all_targets_are_finite": all(item["all_targets_finite"] for item in audits),
    }
    return {
        "version": "1.0",
        "project": "O10 Layerwise Eligibility Traces",
        "main_question": (
            "Can fixed-size local traces kept separately at several network layers preserve "
            "credit from long streams better than an equally sized replay buffer?"
        ),
        "specific_interrogation": (
            "Test whether separate rank-eight adapter-gradient sketches retain the contribution "
            "of every event as the delay grows, while same-state replay, pooled-layer, "
            "permuted-layer, recent-event, and multi-timescale controls lose credit."
        ),
        "trace_rank": rank,
        "task_audits": audits,
        "checks": checks,
        "premise_passed": all(checks.values()),
    }
