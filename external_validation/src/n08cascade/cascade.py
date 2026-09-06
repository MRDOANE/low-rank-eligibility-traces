from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import torch

from .credit import CreditCodec
from .data import CriteoStream, SyntheticStream, YearbookStream, prepare_criteo, prepare_yearbook
from .engine import make_model, run_trial, warm_start
from .report import build_gate, paired_intervals, select_rank, write_reports
from .util import append_jsonl, atomic_json, digest_json, seed_all


def choose_device() -> torch.device:
    requested = os.environ.get("DEVICE", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE requests CUDA, but torch.cuda.is_available() is false")
    return device


def _trial_path(results_dir: Path, trial: dict[str, Any]) -> Path:
    delay = str(trial["delay"]).replace("/", "-")
    rank = trial.get("rank", "none")
    name = (
        f"{trial['stage']}__{trial['benchmark']}__delay-{delay}__seed-{trial['seed']}"
        f"__{trial['method']}__rank-{rank}.json"
    )
    if "learning_rate" in trial and trial["stage"] == "calibration":
        rate = str(trial["learning_rate"]).replace(".", "p")
        name = name.removesuffix(".json") + f"__lr-{rate}.json"
    return results_dir / "trials" / name


def _run_and_log(
    *,
    trial: dict[str, Any],
    model_config: dict[str, Any],
    batches_factory: Callable[[], Iterator],
    device: torch.device,
    results_dir: Path,
    initial_state: dict[str, torch.Tensor] | None,
    naive_buffer_records: int,
) -> dict[str, Any]:
    progress = results_dir / "progress.jsonl"
    append_jsonl(progress, {"event": "trial_start", "time": time.time(), "trial": trial})
    print(
        f"[{trial['stage']}] {trial['benchmark']} delay={trial['delay']} seed={trial['seed']} "
        f"method={trial['method']} rank={trial.get('rank')}",
        flush=True,
    )
    result = run_trial(
        trial=trial,
        model_config=model_config,
        batches=batches_factory(),
        device=device,
        output_path=_trial_path(results_dir, trial),
        initial_state=initial_state,
        naive_buffer_records=naive_buffer_records,
    )
    append_jsonl(
        progress,
        {
            "event": "trial_complete",
            "time": time.time(),
            "trial": trial,
            "state_peak_bytes": result["state"]["peak_bytes"],
            "throughput": result["throughput_examples_per_second"],
        },
    )
    return result


def _warm_criteo_state(
    *,
    stream: CriteoStream,
    seed: int,
    config_digest: str,
    model_config: dict[str, Any],
    criteo_config: dict[str, Any],
    device: torch.device,
    results_dir: Path,
) -> dict[str, torch.Tensor]:
    cache_path = results_dir / "cache" / f"criteo_warm_seed_{seed}.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload.get("config_digest") == config_digest:
            return payload["state_dict"]
    seed_all(seed)
    model = make_model(model_config, seed, device)

    def batches():
        return stream.batches(
            int(criteo_config["batch_size"]),
            split_names=["train"],
            max_events=None,
        )

    print(f"Criteo chronological warm start, seed={seed}", flush=True)
    warm_start(model, batches, model_config, int(criteo_config["warm_start_epochs"]))
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config_digest": config_digest, "state_dict": state}, cache_path)
    return state


def _calibrate_learning_rates(
    *,
    calibration_config: dict[str, Any],
    model_config: dict[str, Any],
    calibration_seeds: list[int],
    yearbook_stream: YearbookStream,
    criteo_streams: dict[int, CriteoStream],
    config_digest: str,
    yearbook_batch_size: int,
    criteo_batch_size: int,
    device: torch.device,
    results_dir: Path,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Select one optimizer rate per benchmark on disjoint development streams.

    The same selected rate is then used for exact credit, every compressed
    state, and every replay control.  This prevents method-specific tuning from
    becoming an unreported source of advantage.
    """

    rates = [float(value) for value in calibration_config["learning_rates"]]
    rows: list[dict[str, Any]] = []
    selected: dict[str, float] = {}
    for benchmark in ("yearbook", "criteo"):
        rate_scores: dict[float, list[float]] = {rate: [] for rate in rates}
        for rate in rates:
            calibrated_model = {**model_config, "learning_rate": rate}
            for seed in calibration_seeds:
                trial = {
                    "stage": "calibration",
                    "benchmark": benchmark,
                    "delay": 0 if benchmark == "yearbook" else "historical_train",
                    "seed": seed,
                    "method": "immediate_oracle",
                    "rank": None,
                    "learning_rate": rate,
                    "update_batch_size": (
                        yearbook_batch_size if benchmark == "yearbook" else criteo_batch_size
                    ),
                    "config_digest": config_digest,
                }
                if benchmark == "yearbook":
                    def batches_factory(
                        stream: YearbookStream = yearbook_stream,
                    ) -> Iterator:
                        return stream.batches(
                            yearbook_batch_size,
                            delay_batches=0,
                            max_events=int(calibration_config["yearbook_examples"]),
                            device=device,
                        )
                else:
                    stream = criteo_streams[seed]

                    def batches_factory(stream: CriteoStream = stream) -> Iterator:
                        return stream.batches(
                            criteo_batch_size,
                            split_names=["train"],
                            max_events=int(calibration_config["criteo_examples"]),
                        )
                result = _run_and_log(
                    trial=trial,
                    model_config=calibrated_model,
                    batches_factory=batches_factory,
                    device=device,
                    results_dir=results_dir,
                    initial_state=None,
                    naive_buffer_records=524_288,
                )
                score = float(result["predictive"]["overall"]["log_loss"])
                rate_scores[rate].append(score)
                rows.append(
                    {
                        "benchmark": benchmark,
                        "seed": seed,
                        "learning_rate": rate,
                        "prequential_log_loss": score,
                        "balanced_accuracy": result["predictive"]["overall"][
                            "balanced_accuracy"
                        ],
                    }
                )
        means = {rate: sum(values) / len(values) for rate, values in rate_scores.items()}
        selected[benchmark] = min(rates, key=lambda rate: (means[rate], rate))
    summary = {
        "selection_metric": calibration_config["selection_metric"],
        "shared_across_methods": bool(calibration_config["shared_across_methods"]),
        "calibration_seeds": calibration_seeds,
        "selected_learning_rates": selected,
        "rows": rows,
    }
    atomic_json(results_dir / "calibration.json", summary)
    return selected, summary


def run_full(config: dict[str, Any], data_dir: Path, results_dir: Path) -> dict[str, Any]:
    device = choose_device()
    print(f"Compute device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    config_digest = digest_json(config)
    results_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(results_dir / "frozen_protocol.json", config)

    model_config = dict(config["model"])
    yearbook_config = dict(config["yearbook"])
    yearbook_config["dimension"] = model_config["dimension"]
    criteo_config = dict(config["criteo"])
    yearbook_provenance = prepare_yearbook(yearbook_config, data_dir, device)
    criteo_provenance = prepare_criteo(criteo_config, data_dir)
    yearbook_stream = YearbookStream(Path(yearbook_provenance["root"]), split="train")
    yearbook_calibration_stream = YearbookStream(
        Path(yearbook_provenance["root"]), split="test"
    )

    seeds = [int(value) for value in config["seeds"]]
    calibration_seeds = [int(value) for value in config["calibration_seeds"]]
    ranks = [int(value) for value in config["ranks"]]
    audit_results: list[dict[str, Any]] = []
    criteo_streams: dict[int, CriteoStream] = {}
    warm_states: dict[int, dict[str, torch.Tensor]] = {}
    all_stream_seeds = calibration_seeds + seeds
    base_criteo_stream = CriteoStream(
        data_dir / "criteo",
        criteo_config,
        int(model_config["dimension"]),
        all_stream_seeds[0],
        device,
    )
    for seed in all_stream_seeds:
        stream = (
            base_criteo_stream
            if seed == all_stream_seeds[0]
            else base_criteo_stream.with_seed(seed, int(model_config["dimension"]))
        )
        criteo_streams[seed] = stream

    selected_learning_rates, calibration_summary = _calibrate_learning_rates(
        calibration_config=dict(config["calibration"]),
        model_config=model_config,
        calibration_seeds=calibration_seeds,
        yearbook_stream=yearbook_calibration_stream,
        criteo_streams=criteo_streams,
        config_digest=config_digest,
        yearbook_batch_size=int(yearbook_config["batch_size"]),
        criteo_batch_size=int(criteo_config["batch_size"]),
        device=device,
        results_dir=results_dir,
    )
    benchmark_model_configs = {
        benchmark: {**model_config, "learning_rate": rate}
        for benchmark, rate in selected_learning_rates.items()
    }
    warm_digest = digest_json(
        {
            "config_digest": config_digest,
            "selected_criteo_learning_rate": selected_learning_rates["criteo"],
            "model_revision": "trace_adapter_v1.1",
        }
    )
    for seed in seeds:
        stream = criteo_streams[seed]
        warm_states[seed] = _warm_criteo_state(
            stream=stream,
            seed=seed,
            config_digest=warm_digest,
            model_config=benchmark_model_configs["criteo"],
            criteo_config=criteo_config,
            device=device,
            results_dir=results_dir,
        )

    for delay in yearbook_config["delays_in_batches"]:
        for seed in seeds:
            common = {
                "stage": "audit",
                "benchmark": "yearbook",
                "delay": int(delay),
                "seed": seed,
                "config_digest": config_digest,
                "learning_rate": selected_learning_rates["yearbook"],
                "update_batch_size": int(yearbook_config["batch_size"]),
            }
            for method, rank in [("exact_credit", 1)] + [("global_svd", value) for value in ranks]:
                trial = {**common, "method": method, "rank": rank}
                audit_results.append(
                    _run_and_log(
                        trial=trial,
                        model_config=benchmark_model_configs["yearbook"],
                        batches_factory=lambda d=int(delay): yearbook_stream.batches(
                            int(yearbook_config["batch_size"]),
                            delay_batches=d,
                            max_events=(
                                None
                                if config["audit"]["yearbook_examples"] is None
                                else int(config["audit"]["yearbook_examples"])
                            ),
                            device=device,
                        ),
                        device=device,
                        results_dir=results_dir,
                        initial_state=None,
                        naive_buffer_records=524_288,
                    )
                )

    for seed in seeds:
        common = {
            "stage": "audit",
            "benchmark": "criteo",
            "delay": "natural",
            "seed": seed,
            "config_digest": config_digest,
            "learning_rate": selected_learning_rates["criteo"],
            "update_batch_size": int(criteo_config["batch_size"]),
        }
        stream = criteo_streams[seed]
        for method, rank in [("exact_credit", 1)] + [("global_svd", value) for value in ranks]:
            trial = {**common, "method": method, "rank": rank}
            audit_results.append(
                _run_and_log(
                    trial=trial,
                    model_config=benchmark_model_configs["criteo"],
                    batches_factory=lambda s=stream: s.batches(
                        int(criteo_config["batch_size"]),
                        split_names=["validation"],
                        max_events=int(config["audit"]["criteo_examples"]),
                    ),
                    device=device,
                    results_dir=results_dir,
                    initial_state=warm_states[seed],
                    naive_buffer_records=524_288,
                )
            )

    selected_rank, rank_audit = select_rank(audit_results, ranks, config["gate"])
    print(f"Cascade selected global-SVD rank {selected_rank}", flush=True)
    probe_codec = CreditCodec(
        "global_svd",
        layers=int(model_config["layers"]),
        dimension=int(model_config["dimension"]),
        candidate_rank=selected_rank,
        seed=seeds[0],
    )
    target_bytes = probe_codec.target_floats * 4
    methods = [
        "global_svd",
        "full_replay",
        "equal_byte_replay",
        "random_projection",
        "naive_delayed",
        "immediate_oracle",
        "o10_layerwise",
        "o10_shared",
    ]
    full_results: list[dict[str, Any]] = []
    for delay in yearbook_config["delays_in_batches"]:
        for seed in seeds:
            for method in methods:
                trial = {
                    "stage": "full",
                    "benchmark": "yearbook",
                    "delay": int(delay),
                    "seed": seed,
                    "method": method,
                    "rank": selected_rank if method in CreditCodec.METHODS else None,
                    "target_bytes_per_event": target_bytes if method == "equal_byte_replay" else 0,
                    "learning_rate": selected_learning_rates["yearbook"],
                    "update_batch_size": int(yearbook_config["batch_size"]),
                    "config_digest": config_digest,
                }
                full_results.append(
                    _run_and_log(
                        trial=trial,
                        model_config=benchmark_model_configs["yearbook"],
                        batches_factory=lambda d=int(delay): yearbook_stream.batches(
                            int(yearbook_config["batch_size"]),
                            delay_batches=d,
                            max_events=None,
                            device=device,
                        ),
                        device=device,
                        results_dir=results_dir,
                        initial_state=None,
                        naive_buffer_records=524_288,
                    )
                )

    for seed in seeds:
        stream = criteo_streams[seed]
        for method in methods:
            trial = {
                "stage": "full",
                "benchmark": "criteo",
                "delay": "natural",
                "seed": seed,
                "method": method,
                "rank": selected_rank if method in CreditCodec.METHODS else None,
                "target_bytes_per_event": target_bytes if method == "equal_byte_replay" else 0,
                "learning_rate": selected_learning_rates["criteo"],
                "update_batch_size": int(criteo_config["batch_size"]),
                "config_digest": config_digest,
            }
            full_results.append(
                _run_and_log(
                    trial=trial,
                    model_config=benchmark_model_configs["criteo"],
                    batches_factory=lambda s=stream: s.batches(
                        int(criteo_config["batch_size"]),
                        split_names=["validation", "test"],
                        max_events=None,
                    ),
                    device=device,
                    results_dir=results_dir,
                    initial_state=warm_states[seed],
                    naive_buffer_records=524_288,
                )
            )

    intervals = paired_intervals(full_results, float(config["gate"]["paired_ci_level"]))
    gate = build_gate(
        rank_audit=rank_audit,
        intervals=intervals,
        full_results=full_results,
        gate_config=config["gate"],
    )
    provenance = {
        "protocol_version": config["protocol_version"],
        "config_digest": config_digest,
        "project_doi": config["project_doi"],
        "yearbook": yearbook_provenance,
        "criteo": criteo_provenance,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "wandb": "disabled and absent from dependencies",
        "calibration": calibration_summary,
        "criteo_initialization": (
            "Retrospective historical initialization on the chronological training split; "
            "optimizer state is reset, and natural reveal times govern validation and test updates."
        ),
    }
    write_reports(results_dir, audit_results, full_results, rank_audit, intervals, gate, provenance)
    final = {
        "color": gate["color"],
        "mode": "full",
        "claim_evaluated": bool(gate["all_valid"]),
        "continue_toward_iclr_claim": bool(gate["all_pass"]),
        "selected_rank": selected_rank,
        "criteria": gate["criteria"],
        "validity_checks": gate["validity_checks"],
        "report": str(results_dir / "REPORT.md"),
    }
    atomic_json(results_dir / "final_status.json", final)
    return final


def run_smoke(config: dict[str, Any], results_dir: Path) -> dict[str, Any]:
    device = choose_device()
    results_dir.mkdir(parents=True, exist_ok=True)
    smoke_config = json.loads(json.dumps(config))
    smoke_config["seeds"] = [11]
    smoke_config["ranks"] = [2, 4]
    smoke_config["model"]["dimension"] = 8
    smoke_config["model"]["layers"] = 3
    model_config = smoke_config["model"]
    digest = digest_json(smoke_config)
    methods = [
        "exact_credit",
        "global_svd",
        "random_projection",
        "o10_layerwise",
        "o10_shared",
        "full_replay",
        "equal_byte_replay",
        "naive_delayed",
        "immediate_oracle",
    ]
    target_codec = CreditCodec("global_svd", layers=3, dimension=8, candidate_rank=2, seed=11)
    target_bytes = target_codec.target_floats * 4
    completed = []
    for benchmark, variable in (("yearbook", False), ("criteo", True)):
        for method in methods:
            stream = SyntheticStream(seed=101 if not variable else 202, examples=96, positions=12, dimension=8, variable_delay=variable)
            trial = {
                "stage": "smoke",
                "benchmark": benchmark,
                "delay": "natural" if variable else 2,
                "seed": 11,
                "method": method,
                "rank": 2 if method in CreditCodec.METHODS else None,
                "target_bytes_per_event": target_bytes if method == "equal_byte_replay" else 0,
                "update_batch_size": 32,
                "config_digest": digest,
            }
            completed.append(
                _run_and_log(
                    trial=trial,
                    model_config=model_config,
                    batches_factory=lambda s=stream: s.batches(16, delay_batches=2, device=device),
                    device=device,
                    results_dir=results_dir,
                    initial_state=None,
                    naive_buffer_records=64,
                )
            )
    final = {
        "color": "GREEN",
        "mode": "smoke",
        "claim_evaluated": False,
        "continue_toward_iclr_claim": False,
        "completed_trials": len(completed),
        "note": "GREEN in smoke mode certifies the execution path only; it is not a scientific result.",
    }
    atomic_json(results_dir / "final_status.json", final)
    return final


def run_verify(project_root: Path, results_dir: Path) -> dict[str, Any]:
    process = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(project_root / "tests"), "-v"],
        cwd=project_root,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError("Verification tests failed")
    final = {
        "color": "GREEN",
        "mode": "verify",
        "claim_evaluated": False,
        "continue_toward_iclr_claim": False,
        "note": "GREEN in verify mode certifies static/unit checks only.",
    }
    atomic_json(results_dir / "final_status.json", final)
    return final


def run(config_path: Path, mode: str, data_dir: Path, results_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if mode == "full":
        return run_full(config, data_dir, results_dir)
    if mode == "smoke":
        return run_smoke(config, results_dir)
    if mode == "verify":
        return run_verify(config_path.parent.parent, results_dir)
    raise ValueError("mode must be full, smoke, or verify")
