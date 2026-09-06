from __future__ import annotations

import math
import os
import resource
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import psutil
import torch


@dataclass
class BinarySums:
    count: int = 0
    loss_sum: float = 0.0
    correct: int = 0
    positive_count: int = 0
    negative_count: int = 0
    true_positive: int = 0
    true_negative: int = 0

    def update(self, logits: np.ndarray, labels: np.ndarray) -> None:
        logits64 = logits.astype(np.float64, copy=False)
        labels64 = labels.astype(np.float64, copy=False)
        losses = np.maximum(logits64, 0.0) - logits64 * labels64 + np.log1p(np.exp(-np.abs(logits64)))
        predictions = logits64 >= 0.0
        positives = labels64 >= 0.5
        self.count += int(labels64.size)
        self.loss_sum += float(losses.sum())
        self.correct += int(np.sum(predictions == positives))
        self.positive_count += int(np.sum(positives))
        self.negative_count += int(np.sum(~positives))
        self.true_positive += int(np.sum(predictions & positives))
        self.true_negative += int(np.sum((~predictions) & (~positives)))

    def summary(self) -> dict[str, float | int | None]:
        positive_recall = self.true_positive / self.positive_count if self.positive_count else None
        negative_recall = self.true_negative / self.negative_count if self.negative_count else None
        balanced = None
        if positive_recall is not None and negative_recall is not None:
            balanced = 0.5 * (positive_recall + negative_recall)
        return {
            "count": self.count,
            "log_loss": self.loss_sum / self.count if self.count else None,
            "accuracy": self.correct / self.count if self.count else None,
            "balanced_accuracy": balanced,
            "positive_rate": self.positive_count / self.count if self.count else None,
        }


class MetricAccumulator:
    def __init__(self) -> None:
        self.groups: dict[str, BinarySums] = defaultdict(BinarySums)

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        strata: list[str],
        splits: list[str],
    ) -> None:
        logits_np = logits.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        self.groups["overall"].update(logits_np, labels_np)
        for split in sorted(set(splits)):
            indices = np.asarray([value == split for value in splits])
            self.groups[f"split:{split}"].update(logits_np[indices], labels_np[indices])
        for stratum in sorted(set(strata)):
            indices = np.asarray([value == stratum for value in strata])
            self.groups[f"delay:{stratum}"].update(logits_np[indices], labels_np[indices])
        for split, stratum in sorted(set(zip(splits, strata))):
            indices = np.asarray(
                [(split_value == split and stratum_value == stratum) for split_value, stratum_value in zip(splits, strata)]
            )
            self.groups[f"split:{split}|delay:{stratum}"].update(logits_np[indices], labels_np[indices])

    def summary(self) -> dict[str, dict[str, float | int | None]]:
        return {name: values.summary() for name, values in sorted(self.groups.items())}


class Reservoir:
    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.values: list[float] = []
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def add_many(self, values: Iterable[float]) -> None:
        for raw in values:
            value = float(raw)
            self.seen += 1
            if len(self.values) < self.capacity:
                self.values.append(value)
            else:
                position = int(self.rng.integers(0, self.seen))
                if position < self.capacity:
                    self.values[position] = value

    def summary(self, prefix: str) -> dict[str, float | int | None]:
        array = np.asarray(self.values, dtype=np.float64)
        return {
            f"{prefix}_count": self.seen,
            f"{prefix}_sampled": int(array.size),
            f"{prefix}_mean": float(np.mean(array)) if array.size else None,
            f"{prefix}_median": float(np.median(array)) if array.size else None,
            f"{prefix}_p05": float(np.percentile(array, 5)) if array.size else None,
            f"{prefix}_p50": float(np.percentile(array, 50)) if array.size else None,
            f"{prefix}_p95": float(np.percentile(array, 95)) if array.size else None,
        }


class StateBytes:
    def __init__(self, static_bytes: int = 0) -> None:
        self.static_bytes = int(static_bytes)
        self.current = int(static_bytes)
        self.peak = int(static_bytes)
        self.samples = 0
        self.sum_bytes = 0

    def add(self, count: int) -> None:
        self.current += int(count)
        self.peak = max(self.peak, self.current)
        self.observe()

    def remove(self, count: int) -> None:
        self.current -= int(count)
        if self.current < self.static_bytes:
            raise RuntimeError("State-byte accounting underflow")
        self.observe()

    def observe(self) -> None:
        self.samples += 1
        self.sum_bytes += self.current

    def summary(self) -> dict[str, int | float]:
        return {
            "static_bytes": self.static_bytes,
            "current_bytes": self.current,
            "peak_bytes": self.peak,
            "mean_observed_bytes": self.sum_bytes / self.samples if self.samples else float(self.current),
        }


class ResourceMonitor:
    def __init__(self, device: torch.device, interval_seconds: float = 0.05) -> None:
        self.device = device
        self.interval_seconds = interval_seconds
        try:
            self.process: psutil.Process | None = psutil.Process(os.getpid())
            self.baseline_rss = self.process.memory_info().rss
        except psutil.Error:
            self.process = None
            self.baseline_rss = self._rusage_peak()
        self.peak_rss = self.baseline_rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self.started = 0.0

    def _sample(self) -> None:
        if self.process is None:
            return
        while not self._stop.wait(self.interval_seconds):
            try:
                self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            except psutil.Error:
                return

    @staticmethod
    def _rusage_peak() -> int:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024

    def __enter__(self) -> "ResourceMonitor":
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        self.started = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.elapsed_seconds = time.perf_counter() - self.started
        self._stop.set()
        self._thread.join(timeout=1.0)
        if self.process is not None:
            try:
                self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            except psutil.Error:
                self.peak_rss = max(self.peak_rss, self._rusage_peak())
        else:
            self.peak_rss = max(self.peak_rss, self._rusage_peak())

    def summary(self) -> dict[str, int | float | None]:
        cuda_allocated = None
        cuda_reserved = None
        if self.device.type == "cuda":
            cuda_allocated = int(torch.cuda.max_memory_allocated(self.device))
            cuda_reserved = int(torch.cuda.max_memory_reserved(self.device))
        return {
            "wall_seconds": float(self.elapsed_seconds),
            "baseline_rss_bytes": int(self.baseline_rss),
            "peak_rss_bytes": int(self.peak_rss),
            "rss_increase_bytes": int(max(0, self.peak_rss - self.baseline_rss)),
            "peak_vram_allocated_bytes": cuda_allocated,
            "peak_vram_reserved_bytes": cuda_reserved,
        }


def t_interval(values: list[float], level: float = 0.95) -> dict[str, float | int | None]:
    from scipy.stats import t

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "lower": None, "upper": None}
    mean = float(np.mean(array))
    if array.size == 1:
        return {"n": 1, "mean": mean, "lower": None, "upper": None}
    standard_error = float(np.std(array, ddof=1) / math.sqrt(array.size))
    critical = float(t.ppf(0.5 + level / 2.0, df=array.size - 1))
    return {
        "n": int(array.size),
        "mean": mean,
        "lower": mean - critical * standard_error,
        "upper": mean + critical * standard_error,
    }
