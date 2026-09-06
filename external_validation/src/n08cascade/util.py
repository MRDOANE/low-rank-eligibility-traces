from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def run_checked(command: list[str], cwd: Path | None = None) -> str:
    process = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process.stdout


def ensure_git_checkout(url: str, commit: str, destination: Path) -> None:
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        run_checked(["git", "clone", "--filter=blob:none", url, str(destination)])
    actual_url = run_checked(["git", "remote", "get-url", "origin"], destination).strip()
    if actual_url.rstrip("/").removesuffix(".git") != url.rstrip("/").removesuffix(".git"):
        raise RuntimeError(f"Refusing checkout with unexpected origin: {actual_url}")
    # A completed earlier run already has the immutable object.  Avoid a
    # needless network request so v1.1 can reuse a v1.0 data cache offline.
    actual = run_checked(["git", "rev-parse", "HEAD"], destination).strip()
    if actual == commit:
        return
    run_checked(["git", "fetch", "--depth", "1", "origin", commit], destination)
    run_checked(["git", "checkout", "--detach", commit], destination)
    actual = run_checked(["git", "rev-parse", "HEAD"], destination).strip()
    if actual != commit:
        raise RuntimeError(f"Pinned commit mismatch: expected {commit}, got {actual}")


def human_bytes(value: int | float) -> str:
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(number) < 1024.0 or unit == "TiB":
            return f"{number:.2f} {unit}"
        number /= 1024.0
    raise AssertionError("unreachable")


def percentile(values: Iterable[float], q: float) -> float | None:
    materialized = list(values)
    if not materialized:
        return None
    return float(np.percentile(np.asarray(materialized, dtype=np.float64), q))


class Timer:
    def __enter__(self) -> "Timer":
        self.started = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed = time.perf_counter() - self.started
