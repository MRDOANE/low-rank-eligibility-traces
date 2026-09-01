from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .o10_tasks import (
    LayerwiseTraceTask,
    full_state_floats,
    replay_slots,
    sample_batch,
    sketch_state_floats,
)


METHODS = (
    "full_trace",
    "layerwise_sketch",
    "stacked_right_subspace",
    "global_sketch",
    "reservoir_replay",
    "stratified_replay",
    "gradient_replay",
    "recent_replay",
    "multi_timescale",
    "pooled_layer_sketch",
    "permuted_layer_sketch",
)

MATCHED_CONTROLS = (
    "stacked_right_subspace",
    "global_sketch",
    "reservoir_replay",
    "stratified_replay",
    "gradient_replay",
    "recent_replay",
    "multi_timescale",
    "pooled_layer_sketch",
    "permuted_layer_sketch",
)

REPLAY_CONTROLS = (
    "reservoir_replay",
    "stratified_replay",
    "gradient_replay",
    "recent_replay",
)


class LayerwiseAdapterModel(nn.Module):
    def __init__(self, task: LayerwiseTraceTask):
        super().__init__()
        self.task = task
        self.register_buffer("base", torch.as_tensor(np.stack(task.base_matrices)))
        self.register_buffer("readout", torch.as_tensor(task.readout))
        self.register_buffer("probe_matrix", torch.as_tensor(task.probe_matrix))
        self.register_buffer("omega", torch.as_tensor(task.omega))
        self.register_buffer("psi", torch.as_tensor(task.psi))
        self.updates = nn.ParameterList(
            [nn.Parameter(torch.zeros(task.dim, task.dim)) for _ in range(task.layers)]
        )

    @torch.no_grad()
    def forward_prediction(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer, update in enumerate(self.updates):
            matrix = self.base[layer] + self.task.adapter_scale * update
            h = torch.tanh(torch.einsum("...j,ij->...i", h, matrix))
        probe = torch.tanh(torch.einsum("...j,ij->...i", x, self.probe_matrix))
        score = (
            torch.sum(h * probe, dim=-1) / np.sqrt(self.task.dim)
            + 0.20 * torch.einsum("...i,i->...", h, self.readout)
        )
        return score.sum(dim=1) / np.sqrt(x.shape[1])

    @torch.no_grad()
    def forward_factors(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = x
        states = [h]
        matrices = []
        for layer, update in enumerate(self.updates):
            matrix = self.base[layer] + self.task.adapter_scale * update
            matrices.append(matrix)
            h = torch.tanh(torch.einsum("...j,ij->...i", h, matrix))
            states.append(h)
        probe = torch.tanh(torch.einsum("...j,ij->...i", x, self.probe_matrix))
        score = (
            torch.sum(h * probe, dim=-1) / np.sqrt(self.task.dim)
            + 0.20 * torch.einsum("...i,i->...", h, self.readout)
        )
        delta = probe / np.sqrt(self.task.dim) + 0.20 * self.readout
        delta = delta * (1.0 - states[-1].square())
        left: list[torch.Tensor] = [torch.empty(0, device=x.device)] * self.task.layers
        right: list[torch.Tensor] = [torch.empty(0, device=x.device)] * self.task.layers
        for layer in reversed(range(self.task.layers)):
            left[layer] = self.task.adapter_scale * delta
            right[layer] = states[layer]
            if layer:
                delta = torch.einsum("...i,ij->...j", delta, matrices[layer])
                delta = delta * (1.0 - states[layer].square())
        prediction = score.sum(dim=1) / np.sqrt(x.shape[1])
        return prediction, torch.stack(left, dim=1), torch.stack(right, dim=1)


@dataclass(frozen=True)
class LayerwiseTrainResult:
    weights: tuple[np.ndarray, ...]
    averaged_weights: tuple[np.ndarray, ...] | None
    averaged_steps: int
    runtime_seconds: float
    trajectory: list[dict[str, float | int]]
    parameters: int
    optimizer_steps: int
    allocated_state_floats: int
    used_state_floats: int
    mean_gradient_cosine: float
    final_gradient_cosine: float
    final_train_mse: float


def seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def full_trace(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.einsum("blti,bltj->blij", left, right) / np.sqrt(left.shape[2])


def two_sided_layerwise_sketch(
    model: LayerwiseAdapterModel,
    left: torch.Tensor,
    right: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    omega = model.omega[:, :, :rank]
    psi = model.psi[:, :, :rank]
    normalizer = np.sqrt(left.shape[2])
    right_omega = torch.einsum("bltd,ldr->bltr", right, omega)
    left_psi = torch.einsum("bltd,ldr->bltr", left, psi)
    column = torch.einsum("bltd,bltr->bldr", left, right_omega) / normalizer
    row_t = torch.einsum("bltd,bltr->bldr", right, left_psi) / normalizer
    core = torch.einsum("bltr,blts->blrs", left_psi, right_omega) / normalizer
    inverse = torch.linalg.pinv(core, rtol=1e-5)
    return torch.einsum("bldr,blrs,bles->blde", column, inverse, row_t)


def global_rank_for_budget(task: LayerwiseTraceTask, budget: int) -> int:
    """Largest stacked-matrix SVD rank that fits ``budget`` floats."""
    per_rank = task.layers * task.dim + task.dim + 1
    return budget // per_rank


def global_sketch_state_floats(task: LayerwiseTraceTask, rank: int) -> int:
    return (task.layers * task.dim + task.dim + 1) * rank


def stacked_right_subspace_state_floats(
    task: LayerwiseTraceTask, rank: int
) -> int:
    return (task.layers * task.dim + task.dim) * rank


def stacked_right_subspace_rank_for_budget(
    task: LayerwiseTraceTask, budget: int
) -> int:
    per_rank = task.layers * task.dim + task.dim
    return min(task.dim, budget // per_rank)


def adaptive_stacked_right_subspace(
    left: torch.Tensor,
    right: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Project every layer trace onto one adaptive right subspace.

    The reference estimator chooses the subspace from a left-energy-weighted
    covariance of the event right factors, then makes a second pass over the
    materialized factors. The retained representation is the shared basis plus
    layer-preserving coefficients. A future online implementation must replace
    this two-pass reference with a subspace tracker before making systems claims.
    """
    batch, layers, length, dim = left.shape
    budget = layers * (2 * dim * rank + rank * rank)
    shared_rank = min(dim, budget // (layers * dim + dim))
    if shared_rank < 1:
        raise ValueError("stacked right-subspace budget cannot support rank one")

    left_energy = torch.sum(left.square(), dim=-1)
    weighted_right = right * torch.sqrt(torch.clamp_min(left_energy, 1e-12))[..., None]
    flattened_right = weighted_right.reshape(batch, layers * length, dim)
    covariance = (
        flattened_right.transpose(-2, -1) @ flattened_right
    ) / float(layers * length)
    _, eigenvectors = torch.linalg.eigh(covariance)
    basis = eigenvectors[..., -shared_rank:]

    normalizer = float(np.sqrt(left.shape[2]))
    right_projection = torch.einsum("bltd,bdr->bltr", right, basis)
    coefficients = (
        torch.einsum("blti,bltr->blir", left, right_projection) / normalizer
    )
    return torch.einsum("blir,bjr->blij", coefficients, basis)


def two_sided_global_sketch(
    model: LayerwiseAdapterModel,
    left: torch.Tensor,
    right: torch.Tensor,
    candidate_rank: int,
) -> torch.Tensor:
    """Best-rank approximation of the stacked layer matrix at matched state.

    Unlike the historical pooled ablation, this control retains layer position
    and may spend its rank unevenly across layers. Its global rank is chosen as
    large as possible without exceeding the separate-sketch state budget. The
    reference implementation deliberately uses the exact truncated SVD, making
    this an information-state oracle rather than a streaming-systems claim.
    """
    task = model.task
    budget = sketch_state_floats(task, candidate_rank)
    rank = global_rank_for_budget(task, budget)
    if rank < 1:
        raise ValueError("global sketch budget cannot support rank one")
    exact = full_trace(left, right)
    stacked = exact.reshape(exact.shape[0], task.layers * task.dim, task.dim)
    u, singular, vh = torch.linalg.svd(stacked, full_matrices=False)
    estimate = (u[:, :, :rank] * singular[:, None, :rank]) @ vh[:, :rank, :]
    return estimate.reshape(exact.shape)


def _sample_indices(
    left: torch.Tensor,
    right: torch.Tensor,
    count: int,
    method: str,
    generator: torch.Generator,
) -> torch.Tensor:
    batch, _, length, _ = left.shape
    count = min(max(1, count), length)
    if method == "recent_replay":
        return torch.arange(length - count, length, device=left.device)[None, :].expand(
            batch, -1
        )
    if method == "gradient_replay":
        scores = torch.zeros(batch, length, device=left.device, dtype=left.dtype)
        for layer in range(left.shape[1]):
            scores = scores + torch.linalg.vector_norm(
                left[:, layer], dim=-1
            ) * torch.linalg.vector_norm(right[:, layer], dim=-1)
        return torch.topk(scores, count, dim=1, largest=True, sorted=False).indices
    if method == "stratified_replay":
        if count == length:
            return torch.arange(length, device=left.device)[None, :].expand(batch, -1)
        starts = torch.div(
            torch.arange(count, dtype=torch.int64) * length,
            count,
            rounding_mode="floor",
        )
        ends = torch.div(
            torch.arange(1, count + 1, dtype=torch.int64) * length,
            count,
            rounding_mode="floor",
        )
        widths = torch.clamp_min(ends - starts, 1)
        draws = torch.rand((batch, count), generator=generator)
        offsets = torch.floor(draws * widths[None, :]).to(torch.int64)
        return (starts[None, :] + offsets).to(left.device)
    rows = [torch.randperm(length, generator=generator)[:count] for _ in range(batch)]
    return torch.stack(rows).to(left.device)


def replay_trace(
    left: torch.Tensor,
    right: torch.Tensor,
    count: int,
    method: str,
    generator: torch.Generator,
) -> torch.Tensor:
    batch, layers, length, dim = left.shape
    indices = _sample_indices(left, right, count, method, generator)
    gather = indices[:, None, :, None].expand(-1, layers, -1, dim)
    selected_left = torch.gather(left, 2, gather)
    selected_right = torch.gather(right, 2, gather)
    return (
        torch.einsum("blki,blkj->blij", selected_left, selected_right)
        * np.sqrt(length)
        / indices.shape[1]
    )


def multi_timescale_trace(
    left: torch.Tensor,
    right: torch.Tensor,
    state_count: int,
) -> torch.Tensor:
    batch, layers, length, dim = left.shape
    traces = max(1, state_count // (layers * 2 * dim))
    decays = torch.linspace(0.35, 0.995, traces, device=left.device, dtype=left.dtype)
    left_state = torch.zeros(
        batch, layers, traces, dim, device=left.device, dtype=left.dtype
    )
    right_state = torch.zeros_like(left_state)
    normalizer = torch.zeros(traces, device=left.device, dtype=left.dtype)
    for step in range(length):
        left_state = decays[None, None, :, None] * left_state + left[
            :, :, step, None, :
        ]
        right_state = decays[None, None, :, None] * right_state + right[
            :, :, step, None, :
        ]
        normalizer = decays * normalizer + 1.0
    left_state = left_state / normalizer[None, None, :, None]
    right_state = right_state / normalizer[None, None, :, None]
    return (
        torch.einsum("blri,blrj->blij", left_state, right_state)
        * np.sqrt(length)
        / traces
    )


def approximate_trace(
    method: str,
    model: LayerwiseAdapterModel,
    left: torch.Tensor,
    right: torch.Tensor,
    rank: int,
    generator: torch.Generator,
) -> torch.Tensor:
    task = model.task
    if method == "full_trace":
        return full_trace(left, right)
    if method in {
        "layerwise_sketch",
        "pooled_layer_sketch",
        "permuted_layer_sketch",
    }:
        estimate = two_sided_layerwise_sketch(model, left, right, rank)
        if method == "pooled_layer_sketch":
            return estimate.mean(dim=1, keepdim=True).expand(-1, task.layers, -1, -1)
        if method == "permuted_layer_sketch":
            return torch.roll(estimate, shifts=1, dims=1)
        return estimate
    if method == "stacked_right_subspace":
        return adaptive_stacked_right_subspace(left, right, rank)
    if method == "global_sketch":
        return two_sided_global_sketch(model, left, right, rank)
    if method in REPLAY_CONTROLS:
        return replay_trace(
            left,
            right,
            replay_slots(task, rank),
            method,
            generator,
        )
    if method == "multi_timescale":
        return multi_timescale_trace(left, right, sketch_state_floats(task, rank))
    raise ValueError(method)


def gradient_cosine(
    error: torch.Tensor,
    estimate: torch.Tensor,
    exact: torch.Tensor,
) -> float:
    estimated_gradient = (error[:, None, None, None] * estimate).flatten(start_dim=1)
    exact_gradient = (error[:, None, None, None] * exact).flatten(start_dim=1)
    numerator = torch.sum(estimated_gradient * exact_gradient, dim=1)
    denominator = torch.linalg.vector_norm(
        estimated_gradient, dim=1
    ) * torch.linalg.vector_norm(exact_gradient, dim=1)
    cosine = torch.where(
        denominator > 1e-12,
        numerator / torch.clamp_min(denominator, 1e-12),
        torch.ones_like(denominator),
    )
    return float(torch.mean(cosine))


def method_state(task: LayerwiseTraceTask, rank: int, method: str) -> tuple[int, int]:
    if method == "full_trace":
        state = full_state_floats(task)
        return state, state
    allocated = sketch_state_floats(task, rank)
    if method == "global_sketch":
        global_rank = global_rank_for_budget(task, allocated)
        used = global_sketch_state_floats(task, global_rank)
    elif method == "stacked_right_subspace":
        stacked_rank = stacked_right_subspace_rank_for_budget(task, allocated)
        used = stacked_right_subspace_state_floats(task, stacked_rank)
    elif method in REPLAY_CONTROLS:
        used = replay_slots(task, rank) * (task.dim + 1)
    elif method == "multi_timescale":
        count = max(1, allocated // (task.layers * 2 * task.dim))
        used = count * task.layers * 2 * task.dim
    else:
        used = allocated
    return allocated, used


def train_method(
    task: LayerwiseTraceTask,
    *,
    method: str,
    seed: int,
    rank: int,
    training_length: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    average_start_step: int | None = None,
) -> LayerwiseTrainResult:
    if method not in METHODS:
        raise ValueError(method)
    seed_everything(seed, device)
    model = LayerwiseAdapterModel(task).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 10_010_003)
    trajectory: list[dict[str, float | int]] = []
    cosine_values = []
    weight_sums = [torch.zeros_like(parameter) for parameter in model.updates]
    averaged_steps = 0
    started = time.perf_counter()
    final_loss = float("nan")
    for step in range(1, steps + 1):
        batch = sample_batch(
            task,
            batch_size,
            200_000 + seed * 10_000 + step,
            length=training_length,
        )
        x = torch.as_tensor(batch.x, dtype=torch.float32, device=device)
        target = torch.as_tensor(batch.target, dtype=torch.float32, device=device)
        prediction, left, right = model.forward_factors(x)
        exact = full_trace(left, right)
        estimate = approximate_trace(method, model, left, right, rank, generator)
        error = prediction - target
        cosine = gradient_cosine(error, estimate, exact)
        cosine_values.append(cosine)
        gradients = torch.mean(error[:, None, None, None] * estimate, dim=0)
        gradient_norm = torch.linalg.vector_norm(gradients)
        clip_scale = min(1.0, 5.0 / max(float(gradient_norm), 1e-12))
        optimizer.zero_grad(set_to_none=True)
        for layer, parameter in enumerate(model.updates):
            parameter.grad = gradients[layer] * clip_scale
        optimizer.step()
        if average_start_step is not None and step >= average_start_step:
            for total, parameter in zip(weight_sums, model.updates):
                total.add_(parameter.detach())
            averaged_steps += 1
        final_loss = float(torch.mean(error.square()))
        if step == 1 or step % 50 == 0 or step == steps:
            trajectory.append(
                {
                    "step": step,
                    "terminal_mse": final_loss,
                    "gradient_cosine_to_full_trace": cosine,
                    "gradient_norm": float(gradient_norm),
                }
            )
    allocated, used = method_state(task, rank, method)
    averaged_weights = None
    if averaged_steps:
        averaged_weights = tuple(
            (total / averaged_steps).cpu().numpy().copy() for total in weight_sums
        )
    return LayerwiseTrainResult(
        weights=tuple(
            parameter.detach().cpu().numpy().copy() for parameter in model.updates
        ),
        averaged_weights=averaged_weights,
        averaged_steps=averaged_steps,
        runtime_seconds=time.perf_counter() - started,
        trajectory=trajectory,
        parameters=parameter_count(model),
        optimizer_steps=steps,
        allocated_state_floats=allocated,
        used_state_floats=used,
        mean_gradient_cosine=float(np.mean(cosine_values)),
        final_gradient_cosine=float(cosine_values[-1]),
        final_train_mse=final_loss,
    )


@torch.no_grad()
def predict(
    task: LayerwiseTraceTask,
    weights: tuple[np.ndarray, ...],
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model = LayerwiseAdapterModel(task).to(device)
    for parameter, value in zip(model.updates, weights):
        parameter.copy_(
            torch.as_tensor(value, dtype=parameter.dtype, device=parameter.device)
        )
    tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    predictions = [model.forward_prediction(chunk) for chunk in tensor.split(128)]
    return torch.cat(predictions).cpu().numpy()
