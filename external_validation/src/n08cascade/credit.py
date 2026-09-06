from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def _state_tensor(tensor: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Make an isolated CPU allocation so storage accounting is literal."""

    return tensor.detach().to(device="cpu", dtype=dtype).clone().contiguous()


@dataclass
class EncodedBatch:
    method: str
    arrays: dict[str, torch.Tensor]
    rows: int

    @property
    def nbytes(self) -> int:
        seen: set[tuple[int, int]] = set()
        total = 0
        for tensor in self.arrays.values():
            storage = tensor.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
        return total

    def take(self, indices: torch.Tensor) -> "EncodedBatch":
        return EncodedBatch(
            method=self.method,
            arrays={name: value.index_select(0, indices).contiguous() for name, value in self.arrays.items()},
            rows=int(indices.numel()),
        )


def _quality(estimate: torch.Tensor, exact: torch.Tensor) -> dict[str, list[float]]:
    estimate_flat = estimate.flatten(start_dim=1).float()
    exact_flat = exact.flatten(start_dim=1).float()
    difference = torch.linalg.vector_norm(estimate_flat - exact_flat, dim=1)
    exact_norm = torch.linalg.vector_norm(exact_flat, dim=1)
    estimate_norm = torch.linalg.vector_norm(estimate_flat, dim=1)
    relative = difference / torch.clamp_min(exact_norm, 1e-12)
    cosine = torch.sum(estimate_flat * exact_flat, dim=1) / torch.clamp_min(
        exact_norm * estimate_norm, 1e-12
    )
    cosine = torch.where(
        (exact_norm < 1e-12) & (estimate_norm < 1e-12),
        torch.ones_like(cosine),
        cosine,
    )
    return {
        "relative_error": relative.detach().cpu().tolist(),
        "gradient_cosine": cosine.detach().cpu().tolist(),
    }


class CreditCodec:
    METHODS = {
        "exact_credit",
        "global_svd",
        "random_projection",
        "o10_layerwise",
        "o10_shared",
    }

    def __init__(
        self,
        method: str,
        *,
        layers: int,
        dimension: int,
        candidate_rank: int,
        seed: int,
        state_dtype: torch.dtype = torch.float32,
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(f"Unknown credit codec {method}")
        self.method = method
        self.layers = layers
        self.dimension = dimension
        self.candidate_rank = candidate_rank
        self.seed = seed
        self.state_dtype = state_dtype
        self.target_floats = self.global_state_floats(candidate_rank)

        generator = torch.Generator(device="cpu").manual_seed(seed + 70_001)
        global_rank = self._random_projection_rank(self.target_floats)
        layer_rank = self._layerwise_rank(self.target_floats)
        self.effective_rank = {
            "exact_credit": dimension,
            "global_svd": min(candidate_rank, dimension),
            "random_projection": global_rank,
            "o10_layerwise": layer_rank,
            "o10_shared": min(dimension, self.target_floats // ((layers + 1) * dimension)),
        }[method]
        if self.effective_rank < 1:
            raise ValueError(f"Budget cannot support rank one for {method}")

        r = self.effective_rank
        if method == "random_projection":
            self.omega = torch.randn(dimension, r, generator=generator) / math.sqrt(r)
            self.psi = torch.randn(layers * dimension, r, generator=generator) / math.sqrt(r)
        elif method == "o10_layerwise":
            self.omega = torch.randn(layers, dimension, r, generator=generator) / math.sqrt(r)
            self.psi = torch.randn(layers, dimension, r, generator=generator) / math.sqrt(r)
        else:
            self.omega = None
            self.psi = None

    def global_state_floats(self, rank: int) -> int:
        return (self.layers * self.dimension + self.dimension + 1) * rank + 1

    def exact_state_floats(self) -> int:
        return self.layers * self.dimension * self.dimension + 1

    def actual_state_floats(self) -> int:
        rank = self.effective_rank
        if self.method == "exact_credit":
            return self.exact_state_floats()
        if self.method == "global_svd":
            return self.global_state_floats(rank)
        if self.method == "random_projection":
            return (self.layers * self.dimension + self.dimension) * rank + rank * rank + 1
        if self.method == "o10_layerwise":
            return self.layers * (2 * self.dimension * rank + rank * rank) + 1
        if self.method == "o10_shared":
            return (self.layers * self.dimension + self.dimension) * rank + 1
        raise AssertionError(self.method)

    def _random_projection_rank(self, budget: int) -> int:
        for rank in range(self.dimension, 0, -1):
            if (self.layers * self.dimension + self.dimension) * rank + rank * rank + 1 <= budget:
                return rank
        return 0

    def _layerwise_rank(self, budget: int) -> int:
        for rank in range(self.dimension, 0, -1):
            floats = self.layers * (2 * self.dimension * rank + rank * rank) + 1
            if floats <= budget:
                return rank
        return 0

    @property
    def static_bytes(self) -> int:
        tensors = [item for item in (self.omega, self.psi) if item is not None]
        return sum(t.numel() * t.element_size() for t in tensors)

    def encode(
        self,
        trace: torch.Tensor,
        logits: torch.Tensor,
        *,
        left: torch.Tensor | None = None,
        right: torch.Tensor | None = None,
    ) -> tuple[EncodedBatch, dict[str, list[float]]]:
        batch = trace.shape[0]
        stacked = trace.reshape(batch, self.layers * self.dimension, self.dimension)
        arrays: dict[str, torch.Tensor]

        if self.method == "exact_credit":
            estimate = trace
            arrays = {
                "trace": _state_tensor(trace, self.state_dtype),
                "logit": _state_tensor(logits),
            }
        elif self.method == "global_svd":
            u, singular, vh = torch.linalg.svd(stacked.float(), full_matrices=False)
            rank = self.effective_rank
            u = u[:, :, :rank]
            singular = singular[:, :rank]
            vh = vh[:, :rank, :]
            estimate = ((u * singular[:, None, :]) @ vh).reshape_as(trace)
            arrays = {
                "u": _state_tensor(u, self.state_dtype),
                "s": _state_tensor(singular, self.state_dtype),
                "vh": _state_tensor(vh, self.state_dtype),
                "logit": _state_tensor(logits),
            }
        elif self.method == "random_projection":
            omega = self.omega.to(stacked.device, stacked.dtype)
            psi = self.psi.to(stacked.device, stacked.dtype)
            column = stacked @ omega
            row_t = stacked.transpose(-2, -1) @ psi
            core = psi.transpose(0, 1) @ column
            inverse = torch.linalg.pinv(core.float(), rtol=1e-5).to(stacked.dtype)
            estimate_stacked = column @ inverse @ row_t.transpose(-2, -1)
            estimate = estimate_stacked.reshape_as(trace)
            arrays = {
                "column": _state_tensor(column, self.state_dtype),
                "row_t": _state_tensor(row_t, self.state_dtype),
                "core": _state_tensor(core, self.state_dtype),
                "logit": _state_tensor(logits),
            }
        elif self.method == "o10_layerwise":
            omega = self.omega.to(trace.device, trace.dtype)
            psi = self.psi.to(trace.device, trace.dtype)
            column = torch.einsum("blij,ljr->blir", trace, omega)
            row_t = torch.einsum("blji,ljr->blir", trace, psi)
            core = torch.einsum("lir,blis->blrs", psi, column)
            inverse = torch.linalg.pinv(core.float(), rtol=1e-5).to(trace.dtype)
            estimate = torch.einsum("blir,blrs,bljs->blij", column, inverse, row_t)
            arrays = {
                "column": _state_tensor(column, self.state_dtype),
                "row_t": _state_tensor(row_t, self.state_dtype),
                "core": _state_tensor(core, self.state_dtype),
                "logit": _state_tensor(logits),
            }
        else:
            if left is None or right is None:
                raise ValueError("o10_shared requires factor tensors")
            rank = self.effective_rank
            left_energy = torch.sum(left.square(), dim=-1)
            weighted_right = right * torch.sqrt(torch.clamp_min(left_energy, 1e-12))[..., None]
            flat = weighted_right.reshape(batch, -1, self.dimension)
            covariance = flat.transpose(-2, -1) @ flat / float(flat.shape[1])
            _, eigenvectors = torch.linalg.eigh(covariance.float())
            basis = eigenvectors[:, :, -rank:].to(trace.dtype)
            coefficients = torch.einsum("blij,bjr->blir", trace, basis)
            estimate = torch.einsum("blir,bjr->blij", coefficients, basis)
            arrays = {
                "coefficients": _state_tensor(coefficients, self.state_dtype),
                "basis": _state_tensor(basis, self.state_dtype),
                "logit": _state_tensor(logits),
            }

        quality = _quality(estimate, trace)
        return EncodedBatch(self.method, arrays, batch), quality

    def decode(self, encoded: EncodedBatch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        arrays = {name: tensor.to(device=device, dtype=torch.float32) for name, tensor in encoded.arrays.items()}
        logits = arrays.pop("logit")
        if self.method == "exact_credit":
            trace = arrays["trace"]
        elif self.method == "global_svd":
            stacked = (arrays["u"] * arrays["s"][:, None, :]) @ arrays["vh"]
            trace = stacked.reshape(encoded.rows, self.layers, self.dimension, self.dimension)
        elif self.method in {"random_projection", "o10_layerwise"}:
            inverse = torch.linalg.pinv(arrays["core"], rtol=1e-5)
            if self.method == "random_projection":
                stacked = arrays["column"] @ inverse @ arrays["row_t"].transpose(-2, -1)
                trace = stacked.reshape(encoded.rows, self.layers, self.dimension, self.dimension)
            else:
                trace = torch.einsum(
                    "blir,blrs,bljs->blij", arrays["column"], inverse, arrays["row_t"]
                )
        else:
            trace = torch.einsum("blir,bjr->blij", arrays["coefficients"], arrays["basis"])
        return trace, logits

    def metadata(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "candidate_rank": self.candidate_rank,
            "effective_rank": self.effective_rank,
            "target_state_floats_per_event": self.target_floats,
            "actual_state_floats_per_event": self.actual_state_floats(),
            "exact_state_floats_per_event": self.exact_state_floats(),
            "static_bytes": self.static_bytes,
        }


def combine_encoded(items: list[EncodedBatch]) -> EncodedBatch:
    if not items:
        raise ValueError("Cannot combine an empty list")
    method = items[0].method
    if any(item.method != method for item in items):
        raise ValueError("Mixed encoded methods")
    names = items[0].arrays.keys()
    arrays = {name: torch.cat([item.arrays[name] for item in items], dim=0) for name in names}
    return EncodedBatch(method, arrays, sum(item.rows for item in items))


def summarize_quality(values: dict[str, list[float]]) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {}
    for name, raw in values.items():
        array = np.asarray(raw, dtype=np.float64)
        result[f"{name}_count"] = int(array.size)
        result[f"{name}_mean"] = float(np.mean(array)) if array.size else None
        result[f"{name}_median"] = float(np.median(array)) if array.size else None
        result[f"{name}_p05"] = float(np.percentile(array, 5)) if array.size else None
        result[f"{name}_p95"] = float(np.percentile(array, 95)) if array.size else None
    return result
