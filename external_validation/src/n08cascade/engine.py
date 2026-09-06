from __future__ import annotations

import heapq
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import torch

from .credit import CreditCodec, EncodedBatch, combine_encoded
from .data import StreamBatch
from .metrics import MetricAccumulator, Reservoir, ResourceMonitor, StateBytes
from .model import TraceAdapter, exact_trace, model_bytes
from .util import atomic_json, digest_json, seed_all


CREDIT_METHODS = set(CreditCodec.METHODS)
REPLAY_METHODS = {"full_replay", "equal_byte_replay", "naive_delayed"}
ALL_METHODS = CREDIT_METHODS | REPLAY_METHODS | {"immediate_oracle"}


@dataclass
class Pending:
    due_key: int
    sequence: int
    payload: EncodedBatch | torch.Tensor
    labels: torch.Tensor
    nbytes: int


class LabeledReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.tokens: deque[torch.Tensor] = deque()
        self.labels: deque[torch.Tensor] = deque()
        self.rows = 0
        self.bytes = 0
        self.generator = torch.Generator(device="cpu").manual_seed(seed + 61_003)

    def add(self, tokens: torch.Tensor, labels: torch.Tensor) -> int:
        tokens_cpu = tokens.detach().cpu().to(torch.float16).contiguous()
        labels_cpu = labels.detach().cpu().to(torch.float32).contiguous()
        added = tokens_cpu.untyped_storage().nbytes() + labels_cpu.untyped_storage().nbytes()
        self.tokens.append(tokens_cpu)
        self.labels.append(labels_cpu)
        self.rows += len(labels_cpu)
        self.bytes += added
        removed = 0
        while self.rows > self.capacity and self.tokens:
            excess = self.rows - self.capacity
            first_tokens = self.tokens[0]
            first_labels = self.labels[0]
            if len(first_labels) <= excess:
                self.tokens.popleft()
                self.labels.popleft()
                self.rows -= len(first_labels)
                delta = first_tokens.untyped_storage().nbytes() + first_labels.untyped_storage().nbytes()
                self.bytes -= delta
                removed += delta
            else:
                keep = torch.arange(excess, len(first_labels), dtype=torch.long)
                old_bytes = first_tokens.untyped_storage().nbytes() + first_labels.untyped_storage().nbytes()
                self.tokens[0] = first_tokens.index_select(0, keep).contiguous()
                self.labels[0] = first_labels.index_select(0, keep).contiguous()
                new_bytes = self.tokens[0].untyped_storage().nbytes() + self.labels[0].untyped_storage().nbytes()
                self.rows -= excess
                self.bytes += new_bytes - old_bytes
                removed += old_bytes - new_bytes
        return added - removed

    def sample(self, count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.rows == 0 or count <= 0:
            return None
        count = min(count, self.rows)
        positions = torch.randint(0, self.rows, (count,), generator=self.generator).numpy()
        token_chunks = list(self.tokens)
        label_chunks = list(self.labels)
        cumulative = np.cumsum([len(chunk) for chunk in label_chunks])
        chunk_ids = np.searchsorted(cumulative, positions, side="right")
        sampled_tokens = []
        sampled_labels = []
        for chunk_id in np.unique(chunk_ids):
            selected_positions = positions[chunk_ids == chunk_id]
            start = 0 if chunk_id == 0 else int(cumulative[chunk_id - 1])
            local = torch.from_numpy((selected_positions - start).astype(np.int64))
            sampled_tokens.append(token_chunks[int(chunk_id)].index_select(0, local))
            sampled_labels.append(label_chunks[int(chunk_id)].index_select(0, local))
        return (
            torch.cat(sampled_tokens, dim=0).to(device=device, dtype=torch.float32),
            torch.cat(sampled_labels, dim=0).to(device=device, dtype=torch.float32),
        )


def make_model(model_config: dict[str, Any], seed: int, device: torch.device) -> TraceAdapter:
    model = TraceAdapter(
        int(model_config["dimension"]),
        int(model_config["layers"]),
        seed=seed,
        adapter_scale=float(model_config["adapter_scale"]),
        readout_scale=float(model_config["readout_scale"]),
    )
    return model.to(device)


def make_optimizer(
    model: TraceAdapter, model_config: dict[str, Any]
) -> torch.optim.Optimizer:
    name = str(model_config.get("optimizer", "sgd")).lower()
    common = {
        "lr": float(model_config["learning_rate"]),
        "weight_decay": float(model_config["weight_decay"]),
    }
    if name == "adamw":
        return torch.optim.AdamW([model.updates], **common)
    if name == "sgd":
        return torch.optim.SGD(
            [model.updates],
            momentum=float(model_config.get("momentum", 0.0)),
            **common,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def apply_trace_step(
    model: TraceAdapter,
    optimizer: torch.optim.Optimizer,
    trace: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    gradient_clip: float,
) -> None:
    residual = torch.sigmoid(logits) - labels
    gradient = torch.mean(residual[:, None, None, None] * trace, dim=0)
    norm = torch.linalg.vector_norm(gradient)
    if gradient_clip > 0 and norm > gradient_clip:
        gradient = gradient * (gradient_clip / torch.clamp_min(norm, 1e-12))
    optimizer.zero_grad(set_to_none=True)
    model.updates.grad = gradient
    optimizer.step()


def warm_start(
    model: TraceAdapter,
    batches_factory: Callable[[], Iterable[StreamBatch]],
    model_config: dict[str, Any],
    epochs: int,
) -> None:
    optimizer = make_optimizer(model, model_config)
    model.train()
    for _ in range(epochs):
        for batch in batches_factory():
            with torch.no_grad():
                logits, left, right = model.forward_factors(batch.tokens)
                trace = exact_trace(left, right)
            apply_trace_step(
                model,
                optimizer,
                trace,
                logits,
                batch.labels,
                float(model_config["gradient_clip"]),
            )


def run_trial(
    *,
    trial: dict[str, Any],
    model_config: dict[str, Any],
    batches: Iterator[StreamBatch],
    device: torch.device,
    output_path: Path,
    initial_state: dict[str, torch.Tensor] | None = None,
    naive_buffer_records: int = 524_288,
    quality_capacity: int = 100_000,
) -> dict[str, Any]:
    trial_digest = digest_json(trial)
    if output_path.exists():
        existing = __import__("json").loads(output_path.read_text(encoding="utf-8"))
        if existing.get("trial_digest") == trial_digest and existing.get("completed") is True:
            print(f"Resume: {output_path.name}", flush=True)
            return existing

    method = trial["method"]
    if method not in ALL_METHODS:
        raise ValueError(method)
    seed = int(trial["seed"])
    seed_all(seed)
    model = make_model(model_config, seed, device)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    optimizer = make_optimizer(model, model_config)
    codec = None
    if method in CREDIT_METHODS:
        codec = CreditCodec(
            method,
            layers=model.layers,
            dimension=model.dimension,
            candidate_rank=int(trial.get("rank", 1)),
            seed=seed,
            state_dtype=torch.float32,
        )
    target_bytes_per_event = int(trial.get("target_bytes_per_event", 0))
    if method == "equal_byte_replay" and target_bytes_per_event <= 0:
        raise ValueError("equal_byte_replay needs target_bytes_per_event")

    metrics = MetricAccumulator()
    cosine = Reservoir(quality_capacity, seed + 1)
    relative_error = Reservoir(quality_capacity, seed + 2)
    latency = Reservoir(quality_capacity, seed + 3)
    state = StateBytes(codec.static_bytes if codec else 0)
    heap: list[tuple[int, int, Pending]] = []
    sequence = 0
    observed = 0
    updated = 0
    updated_online_positive = 0
    updated_online_negative = 0
    predictions_after_feedback = 0
    skipped_budget = 0
    replay_allowance = 0
    labeled_buffer = LabeledReplayBuffer(naive_buffer_records, seed) if method == "naive_delayed" else None

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def step_replay(tokens_cpu: torch.Tensor, labels_cpu: torch.Tensor, use_labeled_memory: bool) -> None:
        nonlocal updated
        tokens = tokens_cpu.to(device=device, dtype=torch.float32)
        labels = labels_cpu.to(device=device, dtype=torch.float32)
        if use_labeled_memory and labeled_buffer is not None:
            sampled = labeled_buffer.sample(len(labels), device)
            if sampled is not None:
                tokens = torch.cat([tokens, sampled[0]], dim=0)
                labels = torch.cat([labels, sampled[1]], dim=0)
        with torch.no_grad():
            logits, left, right = model.forward_factors(tokens)
            trace = exact_trace(left, right)
        apply_trace_step(model, optimizer, trace, logits, labels, float(model_config["gradient_clip"]))
        updated += len(labels_cpu)

    def release(current_key: int, flush_all: bool = False) -> None:
        nonlocal updated, updated_online_positive, updated_online_negative
        due_items: list[Pending] = []
        while heap and (flush_all or heap[0][0] <= current_key):
            _, _, pending = heapq.heappop(heap)
            due_items.append(pending)
        if not due_items:
            return
        if not flush_all:
            online_labels = torch.cat([item.labels for item in due_items], dim=0)
            updated_online_positive += int(torch.sum(online_labels >= 0.5).item())
            updated_online_negative += int(torch.sum(online_labels < 0.5).item())
        update_limit = int(trial.get("update_batch_size", 2048))
        cursor = 0
        while cursor < len(due_items):
            stop = cursor
            rows = 0
            while stop < len(due_items) and (rows == 0 or rows + len(due_items[stop].labels) <= update_limit):
                rows += len(due_items[stop].labels)
                stop += 1
            chunk = due_items[cursor:stop]
            synchronize()
            started = time.perf_counter()
            if method in CREDIT_METHODS:
                assert codec is not None
                combined = combine_encoded([item.payload for item in chunk if isinstance(item.payload, EncodedBatch)])
                labels = torch.cat([item.labels for item in chunk], dim=0).to(device)
                trace, logits = codec.decode(combined, device)
                apply_trace_step(model, optimizer, trace, logits, labels, float(model_config["gradient_clip"]))
                updated += len(labels)
            else:
                token_batches = [item.payload for item in chunk if isinstance(item.payload, torch.Tensor)]
                tokens_cpu = torch.cat(token_batches, dim=0)
                labels_cpu = torch.cat([item.labels for item in chunk], dim=0)
                step_replay(tokens_cpu, labels_cpu, method == "naive_delayed")
                if method == "naive_delayed" and labeled_buffer is not None:
                    before = labeled_buffer.bytes
                    labeled_buffer.add(tokens_cpu, labels_cpu)
                    delta = labeled_buffer.bytes - before
                    if delta >= 0:
                        state.add(delta)
                    else:
                        state.remove(-delta)
            synchronize()
            latency.add_many([time.perf_counter() - started])
            state.remove(sum(item.nbytes for item in chunk))
            cursor = stop

    model.train()
    with ResourceMonitor(device) as resources:
        for batch in batches:
            observed += len(batch.labels)
            with torch.no_grad():
                logits, left, right = model.forward_factors(batch.tokens)
                trace = exact_trace(left, right)
            if updated > 0:
                predictions_after_feedback += len(batch.labels)
            metrics.update(logits, batch.labels, batch.strata, batch.splits)

            if method == "immediate_oracle":
                synchronize()
                started = time.perf_counter()
                apply_trace_step(
                    model,
                    optimizer,
                    trace,
                    logits,
                    batch.labels,
                    float(model_config["gradient_clip"]),
                )
                synchronize()
                latency.add_many([time.perf_counter() - started])
                updated += len(batch.labels)
                updated_online_positive += int(torch.sum(batch.labels >= 0.5).item())
                updated_online_negative += int(torch.sum(batch.labels < 0.5).item())
                continue

            if method in CREDIT_METHODS:
                assert codec is not None
                encoded, quality = codec.encode(trace, logits, left=left, right=right)
                cosine.add_many(quality["gradient_cosine"])
                relative_error.add_many(quality["relative_error"])
                admitted_indices = np.arange(encoded.rows, dtype=np.int64)
            else:
                tokens_cpu = batch.tokens.detach().cpu().to(torch.float16).contiguous()
                bytes_per_event = tokens_cpu[0].numel() * tokens_cpu.element_size()
                if method == "equal_byte_replay":
                    admitted = []
                    for index in range(len(batch.labels)):
                        replay_allowance += target_bytes_per_event
                        if replay_allowance >= bytes_per_event:
                            admitted.append(index)
                            replay_allowance -= bytes_per_event
                        else:
                            skipped_budget += 1
                    admitted_indices = np.asarray(admitted, dtype=np.int64)
                else:
                    admitted_indices = np.arange(len(batch.labels), dtype=np.int64)
                encoded = tokens_cpu

            if admitted_indices.size:
                admitted_due = batch.due_keys[admitted_indices]
                for due_key in np.unique(admitted_due):
                    local_np = admitted_indices[admitted_due == due_key]
                    local = torch.from_numpy(local_np.astype(np.int64))
                    if isinstance(encoded, EncodedBatch):
                        payload: EncodedBatch | torch.Tensor = encoded.take(local)
                        payload_bytes = payload.nbytes
                    else:
                        payload = encoded.index_select(0, local).contiguous()
                        payload_bytes = payload.untyped_storage().nbytes()
                    labels_cpu = batch.labels.detach().cpu().index_select(0, local).float().contiguous()
                    pending = Pending(int(due_key), sequence, payload, labels_cpu, payload_bytes)
                    heapq.heappush(heap, (pending.due_key, sequence, pending))
                    sequence += 1
                    state.add(payload_bytes)

            release(batch.current_key)
            state.observe()
        updated_before_flush = updated
        release(0, flush_all=True)

    result: dict[str, Any] = {
        "completed": True,
        "trial": trial,
        "trial_digest": trial_digest,
        "device": str(device),
        "torch_version": torch.__version__,
        "model_bytes": model_bytes(model),
        "observed_examples": observed,
        "updated_examples": updated,
        "feedback_diagnostics": {
            "updated_before_flush": updated_before_flush,
            "updated_only_during_flush": updated - updated_before_flush,
            "online_positive_labels": updated_online_positive,
            "online_negative_labels": updated_online_negative,
            "predictions_after_first_feedback": predictions_after_feedback,
        },
        "budget_skipped_examples": skipped_budget,
        "predictive": metrics.summary(),
        "credit_quality": {
            **cosine.summary("gradient_cosine"),
            **relative_error.summary("relative_error"),
        },
        "latency": latency.summary("update_latency_seconds"),
        "state": state.summary(),
        "resources": resources.summary(),
    }
    result["throughput_examples_per_second"] = observed / max(float(result["resources"]["wall_seconds"]), 1e-12)
    if codec is not None:
        result["codec"] = codec.metadata()
        result["memory_reduction_vs_exact_per_event"] = codec.exact_state_floats() / max(
            1, codec.actual_state_floats()
        )
    atomic_json(output_path, result)
    return result
