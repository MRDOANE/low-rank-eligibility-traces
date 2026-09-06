from __future__ import annotations

import torch
from torch import nn


class TraceAdapter(nn.Module):
    """Small residual classifier whose logit Jacobians are outer-product sums.

    The frozen token producer is benchmark-specific. Only ``updates`` is trained.
    The last token coordinate is reserved as a constant, which gives the matrix
    stack a trainable intercept without adding an untracked parameter.  This
    keeps every learned quantity inside the delayed-credit object.
    """

    def __init__(
        self,
        dimension: int,
        layers: int,
        *,
        seed: int,
        adapter_scale: float,
        readout_scale: float,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.layers = layers
        self.adapter_scale = adapter_scale
        self.readout_scale = readout_scale

        generator = torch.Generator(device="cpu").manual_seed(seed)
        bases = []
        for _ in range(layers):
            raw = torch.randn(dimension, dimension, generator=generator)
            q, _ = torch.linalg.qr(raw)
            bases.append(0.85 * torch.eye(dimension) + 0.15 * q)
        readout = torch.randn(dimension, generator=generator)
        readout = readout / torch.linalg.vector_norm(readout)

        self.register_buffer("base", torch.stack(bases))
        self.register_buffer("readout", readout)
        self.updates = nn.Parameter(torch.zeros(layers, dimension, dimension))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_factors(tokens)[0]

    def forward_factors(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.dimension:
            raise ValueError(
                f"tokens must have shape [batch, positions, {self.dimension}], got {tuple(tokens.shape)}"
            )
        # Homogeneous-coordinate intercept.  Reserving a feature coordinate is
        # deterministic and ensures the online classifier can learn the class
        # prior instead of being trapped near a zero-logit solution.
        h = tokens.clone()
        h[..., -1] = 1.0
        states = [h]
        matrices = self.base + self.adapter_scale * self.updates
        for layer in range(self.layers):
            h = torch.tanh(torch.einsum("btj,ij->bti", h, matrices[layer]))
            states.append(h)

        score = self.readout_scale * torch.einsum("bti,i->bt", h, self.readout)
        logits = score.mean(dim=1)

        positions = tokens.shape[1]
        delta = self.readout_scale * self.readout.expand_as(states[-1])
        delta = delta * (1.0 - states[-1].square()) / float(positions)
        left: list[torch.Tensor] = [torch.empty(0, device=tokens.device)] * self.layers
        right: list[torch.Tensor] = [torch.empty(0, device=tokens.device)] * self.layers
        for layer in reversed(range(self.layers)):
            left[layer] = self.adapter_scale * delta
            right[layer] = states[layer]
            if layer:
                delta = torch.einsum("bti,ij->btj", delta, matrices[layer])
                delta = delta * (1.0 - states[layer].square())
        return logits, torch.stack(left, dim=1), torch.stack(right, dim=1)


def exact_trace(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return d(logit)/d(adapter matrix) for every example and layer."""

    return torch.einsum("blti,bltj->blij", left, right)


def model_bytes(model: nn.Module) -> int:
    tensors = list(model.parameters()) + list(model.buffers())
    return sum(t.numel() * t.element_size() for t in tensors)
