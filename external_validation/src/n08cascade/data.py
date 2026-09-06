from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import shutil
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from .util import atomic_json, digest_json, ensure_git_checkout, sha256_file


@dataclass
class StreamBatch:
    tokens: torch.Tensor
    labels: torch.Tensor
    event_times: np.ndarray
    reveal_times: np.ndarray
    current_key: int
    due_keys: np.ndarray
    strata: list[str]
    splits: list[str]
    event_ids: np.ndarray


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "n08-external-validation/1.1"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
            os.replace(temporary, destination)
            return
        except Exception as error:
            last_error = error
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Download failed after three attempts: {url}") from last_error


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe archive path: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Refusing linked archive member: {member.name}")
        for member in handle.getmembers():
            handle.extract(member, destination)


def _load_upstream_yearbook(module_path: Path):
    spec = importlib.util.spec_from_file_location("pinned_label_delay_yearbook", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import upstream Yearbook loader at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.YEARBOOK


def prepare_yearbook(config: dict[str, Any], data_dir: Path, device: torch.device) -> dict[str, Any]:
    root = data_dir / "yearbook"
    source_root = root / "upstream"
    label_delay = source_root / "label-delay-exp"
    split_repo = source_root / "yearbook-dating"
    ensure_git_checkout(config["label_delay_repo"], config["label_delay_commit"], label_delay)
    ensure_git_checkout(config["split_repo"], config["split_commit"], split_repo)

    downloads = root / "downloads"
    archive = downloads / "faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz"
    if not archive.exists():
        print(f"Downloading Yearbook archive to {archive}", flush=True)
        _download(config["archive_url"], archive)

    dataset_parent = root / "cldatasets" / "YEARBOOK"
    image_root = dataset_parent / "faces_aligned_small_mirrored_co_aligned_cropped_cleaned"
    if not image_root.exists():
        extract_root = root / "extract"
        _safe_extract(archive, extract_root)
        candidates = [path for path in extract_root.rglob(image_root.name) if path.is_dir()]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one extracted Yearbook directory, found {len(candidates)}")
        dataset_parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(candidates[0]), str(image_root))

    split_sources = {
        "train_M.txt": split_repo / "data" / "faces" / "men" / "train.txt",
        "test_M.txt": split_repo / "data" / "faces" / "men" / "test.txt",
        "train_F.txt": split_repo / "data" / "faces" / "women" / "train.txt",
        "test_F.txt": split_repo / "data" / "faces" / "women" / "test.txt",
    }
    for target_name, source in split_sources.items():
        target = image_root / target_name
        if not target.exists():
            shutil.copyfile(source, target)

    image_extensions = {".png", ".jpg", ".jpeg"}
    physical_images = [path for path in image_root.rglob("*") if path.suffix.lower() in image_extensions]
    expected = int(config["expected_archive_images"])
    if len(physical_images) != expected:
        raise RuntimeError(
            f"Yearbook integrity failure: expected {expected} images, found {len(physical_images)}"
        )

    manifest_path = root / "frozen_stream_manifest.json"
    manifest_key = {
        "label_delay_commit": config["label_delay_commit"],
        "split_commit": config["split_commit"],
        "order_seed": config["order_seed"],
        "archive_sha256": sha256_file(archive),
        "split_sha256": {name: sha256_file(image_root / name) for name in split_sources},
    }
    manifest_digest = digest_json(manifest_key)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("manifest_digest") != manifest_digest:
            raise RuntimeError("Existing Yearbook manifest does not match the frozen protocol")
    else:
        Yearbook = _load_upstream_yearbook(label_delay / "solo" / "data" / "yearbook.py")
        relative_parent = os.path.relpath(dataset_parent, Path.home())
        records: list[dict[str, Any]] = []
        for split_offset, split in enumerate(("train", "test")):
            random.seed(int(config["order_seed"]) + split_offset)
            dataset = Yearbook(relative_parent, transform=None, split=split)
            for position, relative in enumerate(dataset.ordered_files):
                absolute = image_root / relative
                if not absolute.exists():
                    raise RuntimeError(f"Split references missing image: {absolute}")
                records.append(
                    {
                        "event_id": len(records),
                        "split_position": position,
                        "split": split,
                        "relative_path": relative,
                        "label": 1 if relative.startswith("F/") else 0,
                        "year": int(Path(relative).name.split("_")[0]),
                    }
                )
        paths = [item["relative_path"] for item in records]
        if len(paths) != len(set(paths)):
            raise RuntimeError("Yearbook split files contain duplicate image paths")
        manifest = {
            "manifest_digest": manifest_digest,
            "source": manifest_key,
            "physical_image_count": len(physical_images),
            "referenced_image_count": len(records),
            "train_image_count": sum(item["split"] == "train" for item in records),
            "test_image_count": sum(item["split"] == "test" for item in records),
            "paper_reported_train_image_count": config.get("upstream_reported_train_images"),
            "records": records,
        }
        atomic_json(manifest_path, manifest)

    feature_key = {
        "manifest_digest": manifest_digest,
        "backbone": config["backbone"],
        "preprocessing_seed": config["preprocessing_seed"],
        "crop_size": config["crop_size"],
        "crop_scale": config["crop_scale"],
        "pooled_side": config["pooled_side"],
        "dimension": config["dimension"],
    }
    feature_digest = digest_json(feature_key)
    feature_manifest_path = root / "frozen_feature_manifest.json"
    expected_arrays = [
        root / "train_tokens.npy",
        root / "train_labels.npy",
        root / "train_years.npy",
        root / "test_tokens.npy",
        root / "test_labels.npy",
        root / "test_years.npy",
    ]
    cached = False
    if feature_manifest_path.exists() and all(path.exists() for path in expected_arrays):
        existing = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
        cached = existing.get("feature_digest") == feature_digest
    if not cached:
        _extract_yearbook_features(
            config=config,
            root=root,
            image_root=image_root,
            records=manifest["records"],
            device=device,
        )
        atomic_json(
            feature_manifest_path,
            {
                "feature_digest": feature_digest,
                "definition": feature_key,
                "arrays": {path.name: sha256_file(path) for path in expected_arrays},
            },
        )

    return {
        "root": str(root),
        "stream_manifest": str(manifest_path),
        "feature_manifest": str(feature_manifest_path),
        "physical_images": manifest["physical_image_count"],
        "referenced_images": manifest["referenced_image_count"],
        "train_images": manifest["train_image_count"],
        "test_images": manifest["test_image_count"],
    }


def _extract_yearbook_features(
    *,
    config: dict[str, Any],
    root: Path,
    image_root: Path,
    records: list[dict[str, Any]],
    device: torch.device,
) -> None:
    from PIL import Image
    from torchvision.models import ResNet18_Weights, resnet18
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import RandomResizedCrop
    from torchvision.transforms import functional as TF

    weights = ResNet18_Weights.IMAGENET1K_V1
    backbone_model = resnet18(weights=weights)
    backbone = torch.nn.Sequential(
        backbone_model.conv1,
        backbone_model.bn1,
        backbone_model.relu,
        backbone_model.maxpool,
        backbone_model.layer1,
        backbone_model.layer2,
    ).to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    dimension = int(config["dimension"])
    generator = torch.Generator(device="cpu").manual_seed(int(config["preprocessing_seed"]))
    raw = torch.randn(128, dimension, generator=generator)
    projection, _ = torch.linalg.qr(raw, mode="reduced")
    projection = projection.to(device)
    pooled_side = int(config["pooled_side"])
    mean = weights.transforms().mean
    std = weights.transforms().std
    extraction_batch_size = int(os.environ.get("FEATURE_BATCH_SIZE", "64"))

    for split in ("train", "test"):
        selected = [record for record in records if record["split"] == split]
        token_path = root / f"{split}_tokens.npy"
        label_path = root / f"{split}_labels.npy"
        year_path = root / f"{split}_years.npy"
        tokens_out = np.lib.format.open_memmap(
            token_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(selected), pooled_side * pooled_side, dimension),
        )
        labels_out = np.lib.format.open_memmap(label_path, mode="w+", dtype=np.int8, shape=(len(selected),))
        years_out = np.lib.format.open_memmap(year_path, mode="w+", dtype=np.int16, shape=(len(selected),))

        for start in range(0, len(selected), extraction_batch_size):
            chunk = selected[start : start + extraction_batch_size]
            images = []
            for record in chunk:
                with Image.open(image_root / record["relative_path"]) as handle:
                    image = handle.convert("RGB")
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(int(config["preprocessing_seed"]) + int(record["event_id"]))
                        top, left, height, width = RandomResizedCrop.get_params(
                            image,
                            scale=tuple(float(x) for x in config["crop_scale"]),
                            ratio=(3.0 / 4.0, 4.0 / 3.0),
                        )
                        image = TF.resized_crop(
                            image,
                            top,
                            left,
                            height,
                            width,
                            [int(config["crop_size"]), int(config["crop_size"])],
                            interpolation=InterpolationMode.BILINEAR,
                            antialias=True,
                        )
                        if torch.rand(()) < 0.5:
                            image = TF.hflip(image)
                    tensor = TF.normalize(TF.to_tensor(image), mean=mean, std=std)
                    images.append(tensor)
            image_batch = torch.stack(images).to(device)
            with torch.inference_mode():
                features = backbone(image_batch)
                features = torch.nn.functional.adaptive_avg_pool2d(features, (pooled_side, pooled_side))
                features = features.permute(0, 2, 3, 1) @ projection
                features = torch.nn.functional.layer_norm(features, (dimension,))
                feature_tokens = features.reshape(len(chunk), pooled_side * pooled_side, dimension)
            stop = start + len(chunk)
            tokens_out[start:stop] = feature_tokens.detach().cpu().numpy().astype(np.float16)
            labels_out[start:stop] = np.asarray([record["label"] for record in chunk], dtype=np.int8)
            years_out[start:stop] = np.asarray([record["year"] for record in chunk], dtype=np.int16)
            if start % (10 * extraction_batch_size) == 0:
                print(f"Yearbook feature cache {split}: {stop}/{len(selected)}", flush=True)
        tokens_out.flush()
        labels_out.flush()
        years_out.flush()


class YearbookStream:
    def __init__(self, root: Path, split: str = "train") -> None:
        self.tokens = np.load(root / f"{split}_tokens.npy", mmap_mode="r")
        self.labels = np.load(root / f"{split}_labels.npy", mmap_mode="r")
        self.years = np.load(root / f"{split}_years.npy", mmap_mode="r")
        self.split = split
        if not (len(self.tokens) == len(self.labels) == len(self.years)):
            raise RuntimeError("Yearbook cache arrays have inconsistent lengths")

    def __len__(self) -> int:
        return len(self.labels)

    def batches(
        self,
        batch_size: int,
        *,
        delay_batches: int,
        max_events: int | None = None,
        device: torch.device,
    ) -> Iterator[StreamBatch]:
        length = min(len(self), max_events) if max_events else len(self)
        for start in range(0, length, batch_size):
            stop = min(start + batch_size, length)
            step = start // batch_size
            rows = np.arange(start, stop, dtype=np.int64)
            tokens = torch.from_numpy(np.asarray(self.tokens[start:stop]).copy()).to(device=device, dtype=torch.float32)
            labels = torch.from_numpy(np.asarray(self.labels[start:stop], dtype=np.float32).copy()).to(device)
            yield StreamBatch(
                tokens=tokens,
                labels=labels,
                event_times=rows,
                reveal_times=rows + delay_batches * batch_size,
                current_key=step,
                due_keys=np.full(stop - start, step + delay_batches, dtype=np.int64),
                strata=[f"delay_batches={delay_batches}"] * (stop - start),
                splits=[self.split] * (stop - start),
                event_ids=rows,
            )


def prepare_criteo(config: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    root = data_dir / "criteo"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_name = config["filename"]
    raw_path = raw_dir / raw_name
    if not raw_path.exists():
        raw_path = Path(
            hf_hub_download(
                repo_id=config["dataset_repo"],
                filename=raw_name,
                repo_type="dataset",
                revision=config["dataset_revision"],
                local_dir=raw_dir,
            )
        )
    actual_sha = sha256_file(raw_path)
    if actual_sha != config["raw_sha256"]:
        raise RuntimeError(f"Criteo SHA-256 mismatch: expected {config['raw_sha256']}, got {actual_sha}")

    safe_path = root / "safe_features.parquet"
    outcome_path = root / "labels_and_schedule.parquet"
    manifest_path = root / "frozen_preprocessing_manifest.json"
    definition = {
        "dataset_revision": config["dataset_revision"],
        "raw_sha256": actual_sha,
        "filter": config["filter"],
        "safe_feature_columns": config["safe_feature_columns"],
        "forbidden_model_columns": config["forbidden_model_columns"],
        "negative_maturity_seconds": config["negative_maturity_seconds"],
        "train_end_seconds": config["train_end_seconds"],
        "validation_end_seconds": config["validation_end_seconds"],
    }
    preprocessing_digest = digest_json(definition)
    if manifest_path.exists() and safe_path.exists() and outcome_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("preprocessing_digest") != preprocessing_digest:
            raise RuntimeError("Existing Criteo preprocessing does not match the frozen protocol")
        return manifest

    safe_columns = list(config["safe_feature_columns"])
    forbidden = set(config["forbidden_model_columns"])
    overlap = forbidden.intersection(safe_columns)
    if overlap:
        raise RuntimeError(f"Forbidden Criteo fields entered feature allowlist: {sorted(overlap)}")

    safe_writer: pq.ParquetWriter | None = None
    outcome_writer: pq.ParquetWriter | None = None
    event_offset = 0
    previous_timestamp = -1
    counts = {"train": 0, "validation": 0, "test": 0, "positive": 0}
    try:
        for chunk_number, frame in enumerate(
            pd.read_csv(raw_path, sep="\t", compression="gzip", chunksize=250_000, low_memory=False)
        ):
            timestamps = frame["timestamp"].to_numpy(dtype=np.int64, copy=False)
            if timestamps.size and (timestamps[0] < previous_timestamp or np.any(timestamps[1:] < timestamps[:-1])):
                raise RuntimeError("Criteo input is not chronological")
            if timestamps.size:
                previous_timestamp = int(timestamps[-1])
            clicked = frame.loc[frame["click"] == 1].copy()
            if clicked.empty:
                continue
            size = len(clicked)
            event_ids = np.arange(event_offset, event_offset + size, dtype=np.int64)
            event_offset += size
            clicked_timestamps = clicked["timestamp"].to_numpy(dtype=np.int64)
            labels = clicked["conversion"].to_numpy(dtype=np.int8)
            conversion_times = clicked["conversion_timestamp"].to_numpy(dtype=np.int64)
            positive = labels == 1
            if np.any(conversion_times[positive] < clicked_timestamps[positive]):
                raise RuntimeError("Criteo contains a conversion before its click")
            reveal = np.where(
                positive,
                conversion_times,
                clicked_timestamps + int(config["negative_maturity_seconds"]),
            ).astype(np.int64)
            delay = reveal - clicked_timestamps
            split_codes = np.where(
                clicked_timestamps < int(config["train_end_seconds"]),
                0,
                np.where(clicked_timestamps < int(config["validation_end_seconds"]), 1, 2),
            ).astype(np.int8)

            safe = clicked[safe_columns].copy()
            safe.insert(0, "event_id", event_ids)
            outcomes = pd.DataFrame(
                {
                    "event_id": event_ids,
                    "label": labels,
                    "reveal_timestamp": reveal,
                    "feedback_delay_seconds": delay,
                    "split_code": split_codes,
                }
            )
            safe_table = pa.Table.from_pandas(safe, preserve_index=False)
            outcome_table = pa.Table.from_pandas(outcomes, preserve_index=False)
            if safe_writer is None:
                safe_writer = pq.ParquetWriter(safe_path, safe_table.schema, compression="zstd")
                outcome_writer = pq.ParquetWriter(outcome_path, outcome_table.schema, compression="zstd")
            safe_writer.write_table(safe_table)
            assert outcome_writer is not None
            outcome_writer.write_table(outcome_table)

            counts["train"] += int(np.sum(split_codes == 0))
            counts["validation"] += int(np.sum(split_codes == 1))
            counts["test"] += int(np.sum(split_codes == 2))
            counts["positive"] += int(np.sum(positive))
            if chunk_number % 10 == 0:
                print(f"Criteo preprocessing: {event_offset} clicked impressions retained", flush=True)
    finally:
        if safe_writer is not None:
            safe_writer.close()
        if outcome_writer is not None:
            outcome_writer.close()

    if event_offset == 0:
        raise RuntimeError("Criteo click filter produced no records")
    safe_schema = set(pq.read_schema(safe_path).names)
    if forbidden.intersection(safe_schema):
        raise RuntimeError("Outcome-derived column found in persisted Criteo feature table")
    manifest = {
        "preprocessing_digest": preprocessing_digest,
        "definition": definition,
        "raw_path": str(raw_path),
        "safe_feature_path": str(safe_path),
        "labels_schedule_path": str(outcome_path),
        "retained_clicks": event_offset,
        "counts": counts,
        "safe_feature_sha256": sha256_file(safe_path),
        "labels_schedule_sha256": sha256_file(outcome_path),
    }
    atomic_json(manifest_path, manifest)
    return manifest


class CriteoTokenEncoder:
    CATEGORICAL = ["uid", "campaign", "cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8", "cat9"]
    NUMERIC = ["timestamp", "cost", "time_since_last_click"]

    def __init__(self, dimension: int, buckets: int, seed: int, device: torch.device) -> None:
        self.dimension = dimension
        self.buckets = buckets
        self.device = device
        generator = torch.Generator(device="cpu").manual_seed(seed + 81_117)
        table = torch.randn(buckets, dimension, generator=generator)
        self.table = torch.nn.functional.normalize(table, dim=1).to(device)
        self.field = torch.randn(len(self.CATEGORICAL) + len(self.NUMERIC), dimension, generator=generator).to(device)
        self.field = torch.nn.functional.normalize(self.field, dim=1)
        self.numeric_direction = torch.randn(len(self.NUMERIC), dimension, generator=generator).to(device)
        self.numeric_direction = torch.nn.functional.normalize(self.numeric_direction, dim=1)
        self.multipliers = torch.tensor(
            [1_000_003 + 2 * index for index in range(len(self.CATEGORICAL))],
            dtype=torch.int64,
            device=device,
        )
        self.offsets = torch.tensor(
            [97_409 * (index + 1) for index in range(len(self.CATEGORICAL))],
            dtype=torch.int64,
            device=device,
        )

    def __call__(self, columns: dict[str, np.ndarray], indices: np.ndarray) -> torch.Tensor:
        categorical = np.column_stack([columns[name][indices] for name in self.CATEGORICAL]).astype(np.int64)
        values = torch.from_numpy(categorical).to(self.device)
        hashed = torch.remainder(values * self.multipliers[None, :] + self.offsets[None, :], self.buckets)
        tokens = self.table[hashed] + self.field[None, : len(self.CATEGORICAL), :]

        timestamp = torch.from_numpy(columns["timestamp"][indices].astype(np.float32)).to(self.device) / 2_592_000.0
        cost = torch.from_numpy(columns["cost"][indices].astype(np.float32)).to(self.device)
        cost = torch.log1p(torch.clamp_min(cost, 0.0) * 1_000_000.0) / math.log1p(60_000.0)
        since_np = columns["time_since_last_click"][indices].astype(np.float32)
        since = torch.from_numpy(np.maximum(since_np, 0.0)).to(self.device)
        since = torch.log1p(since) / math.log1p(2_592_000.0)
        numerics = torch.stack([timestamp, cost, since], dim=1)
        numeric_tokens = (
            self.field[None, len(self.CATEGORICAL) :, :]
            + numerics[:, :, None] * self.numeric_direction[None, :, :]
        )
        all_tokens = torch.cat([tokens, numeric_tokens], dim=1)
        normalized = torch.nn.functional.layer_norm(all_tokens, (self.dimension,))
        # The frozen model-input boundary is float16 on both benchmarks.  Cast
        # back for model arithmetic; replay retains the exact quantized values.
        return normalized.to(torch.float16).to(torch.float32)


class CriteoStream:
    SPLIT_NAMES = np.asarray(["train", "validation", "test"], dtype=object)

    def __init__(self, root: Path, config: dict[str, Any], dimension: int, seed: int, device: torch.device) -> None:
        import pyarrow.parquet as pq

        safe_table = pq.read_table(root / "safe_features.parquet")
        outcome_table = pq.read_table(root / "labels_and_schedule.parquet")
        safe_ids = safe_table["event_id"].to_numpy()
        outcome_ids = outcome_table["event_id"].to_numpy()
        if not np.array_equal(safe_ids, outcome_ids):
            raise RuntimeError("Criteo feature/outcome row identifiers are not aligned")
        forbidden = set(config["forbidden_model_columns"])
        if forbidden.intersection(safe_table.column_names):
            raise RuntimeError("Forbidden Criteo field reached the model feature table")
        self.columns = {
            name: safe_table[name].to_numpy(zero_copy_only=False)
            for name in config["safe_feature_columns"]
        }
        self.event_ids = safe_ids.astype(np.int64)
        self.labels = outcome_table["label"].to_numpy().astype(np.float32)
        self.reveal = outcome_table["reveal_timestamp"].to_numpy().astype(np.int64)
        self.delay = outcome_table["feedback_delay_seconds"].to_numpy().astype(np.int64)
        self.split_codes = outcome_table["split_code"].to_numpy().astype(np.int8)
        self.config = config
        self.encoder = CriteoTokenEncoder(dimension, int(config["hash_buckets"]), seed, device)
        self.device = device

    def with_seed(self, seed: int, dimension: int) -> "CriteoStream":
        clone = object.__new__(CriteoStream)
        clone.columns = self.columns
        clone.event_ids = self.event_ids
        clone.labels = self.labels
        clone.reveal = self.reveal
        clone.delay = self.delay
        clone.split_codes = self.split_codes
        clone.config = self.config
        clone.device = self.device
        clone.encoder = CriteoTokenEncoder(
            dimension,
            int(self.config["hash_buckets"]),
            seed,
            self.device,
        )
        return clone

    def __len__(self) -> int:
        return len(self.labels)

    def indices_for(self, split_names: Sequence[str]) -> np.ndarray:
        codes = [int(np.where(self.SPLIT_NAMES == name)[0][0]) for name in split_names]
        return np.flatnonzero(np.isin(self.split_codes, codes))

    def _delay_strata(self, delays: np.ndarray) -> list[str]:
        bounds = np.asarray(self.config["delay_bins_seconds"], dtype=np.int64)
        bins = np.searchsorted(bounds, delays, side="right") - 1
        output = []
        for bin_index in bins:
            lower = int(bounds[bin_index])
            upper = int(bounds[bin_index + 1])
            upper_text = "inf" if upper == np.iinfo(np.int64).max else str(upper)
            output.append(f"delay_seconds=[{lower},{upper_text})")
        return output

    def batches(
        self,
        batch_size: int,
        *,
        split_names: Sequence[str],
        max_events: int | None,
    ) -> Iterator[StreamBatch]:
        indices = self.indices_for(split_names)
        if max_events:
            indices = indices[:max_events]
        resolution = int(self.config["feedback_resolution_seconds"])
        for offset in range(0, len(indices), batch_size):
            rows = indices[offset : offset + batch_size]
            timestamps = self.columns["timestamp"][rows].astype(np.int64)
            reveal = self.reveal[rows]
            due = np.floor_divide(reveal, resolution).astype(np.int64)
            split_names_batch = self.SPLIT_NAMES[self.split_codes[rows]].tolist()
            yield StreamBatch(
                tokens=self.encoder(self.columns, rows),
                labels=torch.from_numpy(self.labels[rows].copy()).to(self.device),
                event_times=timestamps,
                reveal_times=reveal,
                current_key=int(timestamps[-1] // resolution),
                due_keys=due,
                strata=self._delay_strata(self.delay[rows]),
                splits=split_names_batch,
                event_ids=self.event_ids[rows],
            )


class SyntheticStream:
    def __init__(self, *, seed: int, examples: int, positions: int, dimension: int, variable_delay: bool) -> None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.tokens = torch.randn(examples, positions, dimension, generator=generator)
        direction = torch.randn(dimension, generator=generator)
        logits = torch.einsum("btd,d->b", self.tokens, direction) / math.sqrt(positions * dimension)
        self.labels = torch.bernoulli(torch.sigmoid(logits), generator=generator)
        self.variable_delay = variable_delay
        self.rng = np.random.default_rng(seed + 91)

    def __len__(self) -> int:
        return len(self.labels)

    def batches(
        self,
        batch_size: int,
        *,
        delay_batches: int = 2,
        device: torch.device,
    ) -> Iterator[StreamBatch]:
        for start in range(0, len(self), batch_size):
            stop = min(start + batch_size, len(self))
            step = start // batch_size
            rows = np.arange(start, stop, dtype=np.int64)
            if self.variable_delay:
                delays = self.rng.integers(1, 6, size=len(rows), dtype=np.int64)
                due = step + delays
                strata = [f"delay_batches={value}" for value in delays]
            else:
                due = np.full(len(rows), step + delay_batches, dtype=np.int64)
                strata = [f"delay_batches={delay_batches}"] * len(rows)
            yield StreamBatch(
                tokens=self.tokens[start:stop].to(device),
                labels=self.labels[start:stop].to(device),
                event_times=rows,
                reveal_times=due,
                current_key=step,
                due_keys=due,
                strata=strata,
                splits=["test"] * len(rows),
                event_ids=rows,
            )
