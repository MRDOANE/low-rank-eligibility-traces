from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .n08_tasks import DelayedOutcomeTask, sample_batch, trace_state_floats


METHODS = (
    "full_trace",
    "lowrank_svd",
    "recent_factors",
    "reservoir_factors",
    "replay_reservoir",
    "multi_timescale",
)
MEMORY_MATCHED_CONTROLS = (
    "recent_factors",
    "reservoir_factors",
    "replay_reservoir",
    "multi_timescale",
)


class DelayedAdapterModel(nn.Module):
    def __init__(self, task: DelayedOutcomeTask):
        super().__init__()
        self.task = task
        self.register_buffer("base", torch.as_tensor(np.stack(task.base_matrices)))
        self.register_buffer("readout", torch.as_tensor(task.readout))
        self.register_buffer("probe_matrix", torch.as_tensor(task.probe_matrix))
        self.updates = nn.ParameterList(
            [nn.Parameter(torch.zeros(task.dim, task.dim)) for _ in range(task.layers)]
        )

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
            + 0.15 * torch.einsum("...i,i->...", h, self.readout)
        )
        delta = probe / np.sqrt(self.task.dim) + 0.15 * self.readout
        delta = delta * (1.0 - states[-1].square())
        left: list[torch.Tensor] = [torch.empty(0, device=x.device)] * self.task.layers
        right: list[torch.Tensor] = [torch.empty(0, device=x.device)] * self.task.layers
        for layer in reversed(range(self.task.layers)):
            left[layer] = self.task.adapter_scale * delta
            right[layer] = states[layer]
            if layer:
                delta = torch.einsum("...i,ij->...j", delta, matrices[layer])
                delta = delta * (1.0 - states[layer].square())
        prediction = score.mean(dim=1)
        return prediction, torch.stack(left, dim=1), torch.stack(right, dim=1)


@dataclass(frozen=True)
class TraceTrainResult:
    weights: tuple[np.ndarray, ...]
    runtime_seconds: float
    trajectory: list[dict[str, float | int]]
    parameters: int
    optimizer_steps: int
    state_floats_per_episode: int
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
    return torch.einsum("blti,bltj->blij", left, right) / left.shape[2]


def streaming_lowrank_trace(
    left: torch.Tensor,
    right: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    batch, layers, length, dim = left.shape
    effective_rank = min(rank, dim)
    u = torch.zeros(
        batch, layers, dim, effective_rank, dtype=left.dtype, device=left.device
    )
    v = torch.zeros_like(u)
    singular = torch.zeros(
        batch, layers, effective_rank, dtype=left.dtype, device=left.device
    )
    for step in range(length):
        left_augmented = torch.cat(
            [u * singular.unsqueeze(-2), left[:, :, step].unsqueeze(-1) / length],
            dim=-1,
        )
        right_augmented = torch.cat(
            [v, right[:, :, step].unsqueeze(-1)], dim=-1
        )
        q_left, r_left = torch.linalg.qr(left_augmented, mode="reduced")
        q_right, r_right = torch.linalg.qr(right_augmented, mode="reduced")
        core = r_left @ r_right.transpose(-2, -1)
        core_left, core_singular, core_right_h = torch.linalg.svd(
            core, full_matrices=False
        )
        u = q_left @ core_left[..., :effective_rank]
        v = q_right @ core_right_h.transpose(-2, -1)[..., :effective_rank]
        singular = core_singular[..., :effective_rank]
    return torch.einsum("blir,blr,bljr->blij", u, singular, v)


def _sample_indices(
    batch: int,
    length: int,
    count: int,
    *,
    recent: bool,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    count = min(max(1, count), length)
    if recent:
        indices = torch.arange(length - count, length, dtype=torch.long)
        return indices[None, :].expand(batch, -1).to(device)
    rows = [torch.randperm(length, generator=generator)[:count] for _ in range(batch)]
    return torch.stack(rows).to(device)


def sampled_factor_trace(
    left: torch.Tensor,
    right: torch.Tensor,
    count: int,
    *,
    recent: bool,
    generator: torch.Generator,
) -> torch.Tensor:
    batch, layers, length, dim = left.shape
    indices = _sample_indices(
        batch,
        length,
        count,
        recent=recent,
        generator=generator,
        device=left.device,
    )
    gather = indices[:, None, :, None].expand(-1, layers, -1, dim)
    selected_left = torch.gather(left, 2, gather)
    selected_right = torch.gather(right, 2, gather)
    return torch.einsum("blki,blkj->blij", selected_left, selected_right) / count


def multi_timescale_trace(
    left: torch.Tensor,
    right: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    batch, layers, length, dim = left.shape
    decays = torch.linspace(0.45, 0.96, rank, device=left.device, dtype=left.dtype)
    left_state = torch.zeros(batch, layers, rank, dim, device=left.device, dtype=left.dtype)
    right_state = torch.zeros_like(left_state)
    normalizer = torch.zeros(rank, device=left.device, dtype=left.dtype)
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
    return torch.einsum("blri,blrj->blij", left_state, right_state) / rank


def approximate_trace(
    method: str,
    task: DelayedOutcomeTask,
    left: torch.Tensor,
    right: torch.Tensor,
    rank: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if method == "full_trace":
        return full_trace(left, right)
    if method == "lowrank_svd":
        return streaming_lowrank_trace(left, right, rank)
    if method == "recent_factors":
        return sampled_factor_trace(
            left, right, rank, recent=True, generator=generator
        )
    if method == "reservoir_factors":
        return sampled_factor_trace(
            left, right, rank, recent=False, generator=generator
        )
    if method == "replay_reservoir":
        budget = trace_state_floats(task, rank, "lowrank_svd")
        slots = max(1, budget // (task.dim + 1))
        return sampled_factor_trace(
            left, right, slots, recent=False, generator=generator
        )
    if method == "multi_timescale":
        return multi_timescale_trace(left, right, rank)
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


def train_method(
    task: DelayedOutcomeTask,
    *,
    method: str,
    seed: int,
    rank: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    target_noise: float = 0.01,
) -> TraceTrainResult:
    if method not in METHODS:
        raise ValueError(method)
    seed_everything(seed, device)
    model = DelayedAdapterModel(task).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 808_003)
    trajectory: list[dict[str, float | int]] = []
    cosine_values = []
    started = time.perf_counter()
    final_loss = float("nan")
    for step in range(1, steps + 1):
        batch = sample_batch(
            task,
            batch_size,
            100_000 + seed * 10_000 + step,
            target_noise=target_noise,
        )
        x = torch.as_tensor(batch.x, dtype=torch.float32, device=device)
        target = torch.as_tensor(batch.target, dtype=torch.float32, device=device)
        prediction, left, right = model.forward_factors(x)
        exact = full_trace(left, right)
        estimate = approximate_trace(method, task, left, right, rank, generator)
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
    state_floats = trace_state_floats(task, rank, method)
    return TraceTrainResult(
        weights=tuple(parameter.detach().cpu().numpy().copy() for parameter in model.updates),
        runtime_seconds=time.perf_counter() - started,
        trajectory=trajectory,
        parameters=parameter_count(model),
        optimizer_steps=steps,
        state_floats_per_episode=state_floats,
        mean_gradient_cosine=float(np.mean(cosine_values)),
        final_gradient_cosine=float(cosine_values[-1]),
        final_train_mse=final_loss,
    )


@torch.no_grad()
def predict(
    task: DelayedOutcomeTask,
    weights: tuple[np.ndarray, ...],
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model = DelayedAdapterModel(task).to(device)
    for parameter, value in zip(model.updates, weights):
        parameter.copy_(torch.as_tensor(value, dtype=parameter.dtype, device=device))
    prediction, _, _ = model.forward_factors(
        torch.as_tensor(x, dtype=torch.float32, device=device)
    )
    return prediction.cpu().numpy()
