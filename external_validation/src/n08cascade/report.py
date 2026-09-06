from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import t_interval
from .util import atomic_json


def _primary_value(result: dict[str, Any], group: str | None = None) -> float:
    benchmark = result["trial"]["benchmark"]
    predictive = result["predictive"]
    selected = predictive[group or "overall"]
    if benchmark == "yearbook":
        return float(selected["accuracy"])
    return -float(selected["log_loss"])


def select_rank(
    audit_results: list[dict[str, Any]],
    ranks: list[int],
    gate: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    exact_trials = [
        item for item in audit_results if item["trial"]["method"] == "exact_credit"
    ]
    exact = {
        (item["trial"]["benchmark"], item["trial"]["delay"], item["trial"]["seed"]): item
        for item in exact_trials
    }
    if not exact or len(exact) != len(exact_trials):
        raise RuntimeError("Exact-credit audit cells are missing or duplicated")
    rank_rows: list[dict[str, Any]] = []
    for rank in ranks:
        candidates = [
            item
            for item in audit_results
            if item["trial"]["method"] == "global_svd" and int(item["trial"]["rank"]) == rank
        ]
        candidate_keys = [
            (
                item["trial"]["benchmark"],
                item["trial"]["delay"],
                item["trial"]["seed"],
            )
            for item in candidates
        ]
        if len(candidate_keys) != len(set(candidate_keys)) or set(candidate_keys) != set(exact):
            raise RuntimeError(f"Rank-{rank} audit does not match the exact-credit cells")
        by_cell: dict[tuple[str, str], list[dict[str, float | int]]] = defaultdict(list)
        reductions: list[float] = []
        for candidate in candidates:
            key = (
                candidate["trial"]["benchmark"],
                candidate["trial"]["delay"],
                candidate["trial"]["seed"],
            )
            reference = exact[key]
            candidate_metrics = candidate["predictive"]["overall"]
            reference_metrics = reference["predictive"]["overall"]
            accuracy_gap = 100.0 * (float(reference_metrics["accuracy"]) - float(candidate_metrics["accuracy"]))
            reference_loss = float(reference_metrics["log_loss"])
            relative_excess = (float(candidate_metrics["log_loss"]) - reference_loss) / max(reference_loss, 1e-12)
            cosine_value = candidate["credit_quality"].get("gradient_cosine_median")
            feedback = candidate.get("feedback_diagnostics", {})
            by_cell[(key[0], str(key[1]))].append(
                {
                    "accuracy_gap": accuracy_gap,
                    "relative_excess": relative_excess,
                    "gradient_cosine_median": (
                        float(cosine_value) if cosine_value is not None else float("nan")
                    ),
                    "predictions_after_feedback": int(
                        feedback.get("predictions_after_first_feedback", 0)
                    ),
                }
            )
            reductions.append(float(candidate["memory_reduction_vs_exact_per_event"]))
        cell_summaries = []
        for (benchmark, delay), values in sorted(by_cell.items()):
            mean_accuracy_gap = float(np.mean([float(value["accuracy_gap"]) for value in values]))
            mean_relative_excess = float(
                np.mean([float(value["relative_excess"]) for value in values])
            )
            cell_cosines = [
                float(value["gradient_cosine_median"])
                for value in values
                if np.isfinite(float(value["gradient_cosine_median"]))
            ]
            cell_median_cosine = (
                float(np.median(cell_cosines)) if cell_cosines else float("nan")
            )
            minimum_predictions = min(
                int(value["predictions_after_feedback"]) for value in values
            )
            performance_pass = (
                mean_accuracy_gap <= float(gate["max_accuracy_gap_percentage_points"])
                or mean_relative_excess <= float(gate["max_relative_excess_loss"])
            )
            cosine_pass = bool(cell_cosines) and cell_median_cosine >= float(
                gate["min_median_gradient_cosine"]
            )
            feedback_exposure_pass = minimum_predictions >= int(
                gate["minimum_predictions_after_feedback"]
            )
            cell_summaries.append(
                {
                    "benchmark": benchmark,
                    "delay": delay,
                    "seeds": len(values),
                    "mean_accuracy_gap_percentage_points": mean_accuracy_gap,
                    "mean_relative_excess_loss": mean_relative_excess,
                    "median_gradient_cosine": cell_median_cosine,
                    "minimum_predictions_after_feedback": minimum_predictions,
                    "performance_pass": performance_pass,
                    "cosine_pass": cosine_pass,
                    "feedback_exposure_pass": feedback_exposure_pass,
                    "pass": performance_pass and cosine_pass and feedback_exposure_pass,
                }
            )
        finite_cell_cosines = [
            float(cell["median_gradient_cosine"])
            for cell in cell_summaries
            if np.isfinite(float(cell["median_gradient_cosine"]))
        ]
        minimum_cell_cosine = min(finite_cell_cosines) if finite_cell_cosines else float("nan")
        minimum_reduction = min(reductions) if reductions else 0.0
        row = {
            "rank": rank,
            "audit_trials": len(candidates),
            "cells": cell_summaries,
            "max_accuracy_gap_percentage_points": max(
                (cell["mean_accuracy_gap_percentage_points"] for cell in cell_summaries), default=None
            ),
            "max_relative_excess_loss": max(
                (cell["mean_relative_excess_loss"] for cell in cell_summaries), default=None
            ),
            "minimum_cell_median_gradient_cosine": minimum_cell_cosine,
            "minimum_memory_reduction": minimum_reduction,
            "performance_pass": bool(cell_summaries)
            and all(cell["performance_pass"] for cell in cell_summaries),
            "cosine_pass": bool(finite_cell_cosines)
            and all(cell["cosine_pass"] for cell in cell_summaries),
            "feedback_exposure_pass": bool(cell_summaries)
            and all(cell["feedback_exposure_pass"] for cell in cell_summaries),
            "memory_pass": minimum_reduction >= float(gate["min_memory_reduction"]),
        }
        row["rank_pass"] = (
            row["performance_pass"]
            and row["cosine_pass"]
            and row["feedback_exposure_pass"]
            and row["memory_pass"]
        )
        rank_rows.append(row)

    passing = [row for row in rank_rows if row["rank_pass"]]
    if passing:
        selected = min(passing, key=lambda item: int(item["rank"]))
    else:
        eligible = [row for row in rank_rows if row["memory_pass"]]
        pool = eligible or rank_rows
        selected = max(
            pool,
            key=lambda item: (
                float(item["minimum_cell_median_gradient_cosine"]),
                -float(item["max_relative_excess_loss"] or 0.0),
            ),
        )
    return int(selected["rank"]), {"selected_rank": int(selected["rank"]), "candidates": rank_rows}


def paired_intervals(
    full_results: list[dict[str, Any]], level: float
) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for result in full_results:
        trial = result["trial"]
        indexed[(trial["benchmark"], str(trial["delay"]), int(trial["seed"]), trial["method"])] = result

    intervals: list[dict[str, Any]] = []
    comparators = ("equal_byte_replay", "random_projection")
    yearbook_delays = sorted(
        {key[1] for key in indexed if key[0] == "yearbook" and key[3] == "global_svd"}
    )
    seeds = sorted({key[2] for key in indexed})
    for delay in yearbook_delays:
        for comparator in comparators:
            differences = []
            for seed in seeds:
                primary = indexed.get(("yearbook", delay, seed, "global_svd"))
                control = indexed.get(("yearbook", delay, seed, comparator))
                if primary is not None and control is not None:
                    differences.append(_primary_value(primary) - _primary_value(control))
            interval = t_interval(differences, level)
            intervals.append(
                {
                    "condition_id": f"yearbook:delay={delay}",
                    "benchmark": "yearbook",
                    "delay": delay,
                    "metric": "online_accuracy_advantage",
                    "comparator": comparator,
                    "confirmatory": True,
                    **interval,
                    "ci_excludes_zero_positive": interval["lower"] is not None and float(interval["lower"]) > 0.0,
                }
            )

    # Criteo delay bins are descriptive: the 30-day maturity rule makes them
    # strongly outcome-dependent.  The held-out test stream is one independent
    # confirmatory condition, so it is never expanded into several gate votes.
    for comparator in comparators:
        differences = []
        for seed in seeds:
            primary = indexed.get(("criteo", "natural", seed, "global_svd"))
            control = indexed.get(("criteo", "natural", seed, comparator))
            group = "split:test"
            if (
                primary is not None
                and control is not None
                and group in primary["predictive"]
                and group in control["predictive"]
            ):
                differences.append(
                    _primary_value(primary, group) - _primary_value(control, group)
                )
        interval = t_interval(differences, level)
        intervals.append(
            {
                "condition_id": "criteo:test_natural_delay",
                "benchmark": "criteo",
                "delay": "natural_test_stream",
                "metric": "negative_log_loss_advantage",
                "comparator": comparator,
                "confirmatory": True,
                **interval,
                "ci_excludes_zero_positive": interval["lower"] is not None
                and float(interval["lower"]) > 0.0,
            }
        )
    return intervals


def build_gate(
    *,
    rank_audit: dict[str, Any],
    intervals: list[dict[str, Any]],
    full_results: list[dict[str, Any]],
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    selected = next(
        row for row in rank_audit["candidates"] if int(row["rank"]) == int(rank_audit["selected_rank"])
    )
    winning_by_comparator: dict[str, list[str]] = {}
    for comparator in ("equal_byte_replay", "random_projection"):
        winning_by_comparator[comparator] = sorted(
            {
                str(row["condition_id"])
                for row in intervals
                if row.get("confirmatory", False)
                and row["comparator"] == comparator
                and row["ci_excludes_zero_positive"]
            }
        )
    required_conditions = int(gate_config["min_delays_with_positive_ci"])
    successful_comparators = [
        comparator
        for comparator, conditions in winning_by_comparator.items()
        if len(conditions) >= required_conditions
    ]

    yearbook_oracles = [
        item
        for item in full_results
        if item["trial"]["benchmark"] == "yearbook"
        and item["trial"]["method"] == "immediate_oracle"
    ]
    yearbook_by_delay: dict[str, list[float]] = defaultdict(list)
    for item in yearbook_oracles:
        value = item["predictive"]["overall"].get("balanced_accuracy")
        if value is not None:
            yearbook_by_delay[str(item["trial"]["delay"])].append(float(value))
    yearbook_oracle_means = {
        delay: float(np.mean(values)) for delay, values in sorted(yearbook_by_delay.items())
    }
    minimum_yearbook_oracle = (
        min(yearbook_oracle_means.values()) if yearbook_oracle_means else None
    )

    criteo_relative_gains: list[float] = []
    for item in full_results:
        if not (
            item["trial"]["benchmark"] == "criteo"
            and item["trial"]["method"] == "immediate_oracle"
        ):
            continue
        held_out = item["predictive"].get("split:test")
        if not held_out or held_out.get("positive_rate") is None:
            continue
        positive_rate = min(max(float(held_out["positive_rate"]), 1e-12), 1.0 - 1e-12)
        constant_loss = -(
            positive_rate * np.log(positive_rate)
            + (1.0 - positive_rate) * np.log(1.0 - positive_rate)
        )
        oracle_loss = float(held_out["log_loss"])
        criteo_relative_gains.append(
            float((constant_loss - oracle_loss) / max(constant_loss, 1e-12))
        )
    mean_criteo_oracle_gain = (
        float(np.mean(criteo_relative_gains)) if criteo_relative_gains else None
    )

    criteria = {
        "within_exact_performance_tolerance": bool(selected["performance_pass"]),
        "gradient_cosine_at_least_0_95_in_every_audit_cell": bool(
            selected["cosine_pass"]
        ),
        "credit_memory_reduction_at_least_3x": bool(selected["memory_pass"]),
        "paired_advantage_for_one_comparator_at_two_or_more_conditions": bool(
            successful_comparators
        ),
    }
    validity_checks = {
        "audit_has_predictions_after_delayed_feedback_in_every_cell": bool(
            selected["feedback_exposure_pass"]
        ),
        "yearbook_oracle_is_above_floor": minimum_yearbook_oracle is not None
        and minimum_yearbook_oracle
        >= float(gate_config["min_yearbook_oracle_balanced_accuracy"]),
        "criteo_oracle_beats_constant_predictor": mean_criteo_oracle_gain is not None
        and mean_criteo_oracle_gain
        >= float(gate_config["min_criteo_oracle_relative_logloss_gain_over_constant"]),
    }
    all_valid = all(validity_checks.values())
    all_pass = all(criteria.values()) and all_valid
    color = "GREEN" if all_pass else ("YELLOW" if all_valid else "RED")
    return {
        "criteria": criteria,
        "validity_checks": validity_checks,
        "all_valid": all_valid,
        "all_pass": all_pass,
        "color": color,
        "selected_rank": rank_audit["selected_rank"],
        "selected_rank_audit": selected,
        "positive_ci_conditions_by_comparator": winning_by_comparator,
        "successful_comparators": successful_comparators,
        "oracle_diagnostics": {
            "yearbook_mean_balanced_accuracy_by_delay": yearbook_oracle_means,
            "minimum_yearbook_mean_balanced_accuracy": minimum_yearbook_oracle,
            "criteo_relative_logloss_gains_over_constant_by_seed": criteo_relative_gains,
            "mean_criteo_relative_logloss_gain_over_constant": mean_criteo_oracle_gain,
        },
        "interpretation": (
            "Continue toward an ICLR claim only when all four scientific criteria "
            "and all protocol-validity checks are true."
        ),
    }


def _flatten_trial(result: dict[str, Any]) -> dict[str, Any]:
    trial = result["trial"]
    overall = result["predictive"].get("overall", {})
    return {
        "stage": trial["stage"],
        "benchmark": trial["benchmark"],
        "delay": trial["delay"],
        "seed": trial["seed"],
        "method": trial["method"],
        "rank": trial.get("rank"),
        "learning_rate": trial.get("learning_rate"),
        "effective_rank": result.get("codec", {}).get("effective_rank"),
        "examples": result["observed_examples"],
        "log_loss": overall.get("log_loss"),
        "accuracy": overall.get("accuracy"),
        "balanced_accuracy": overall.get("balanced_accuracy"),
        "gradient_cosine_median": result["credit_quality"].get("gradient_cosine_median"),
        "credit_relative_error_median": result["credit_quality"].get("relative_error_median"),
        "state_peak_bytes": result["state"]["peak_bytes"],
        "peak_rss_bytes": result["resources"]["peak_rss_bytes"],
        "peak_vram_allocated_bytes": result["resources"]["peak_vram_allocated_bytes"],
        "update_latency_p50_seconds": result["latency"].get("update_latency_seconds_p50"),
        "update_latency_p95_seconds": result["latency"].get("update_latency_seconds_p95"),
        "throughput_examples_per_second": result["throughput_examples_per_second"],
        "updates_before_flush": result.get("feedback_diagnostics", {}).get(
            "updated_before_flush"
        ),
        "predictions_after_first_feedback": result.get("feedback_diagnostics", {}).get(
            "predictions_after_first_feedback"
        ),
        "online_positive_labels": result.get("feedback_diagnostics", {}).get(
            "online_positive_labels"
        ),
        "online_negative_labels": result.get("feedback_diagnostics", {}).get(
            "online_negative_labels"
        ),
    }


def write_reports(
    results_dir: Path,
    audit_results: list[dict[str, Any]],
    full_results: list[dict[str, Any]],
    rank_audit: dict[str, Any],
    intervals: list[dict[str, Any]],
    gate: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    atomic_json(results_dir / "rank_audit.json", rank_audit)
    atomic_json(results_dir / "paired_intervals.json", intervals)
    atomic_json(results_dir / "gate.json", gate)
    atomic_json(results_dir / "provenance.json", provenance)

    rows = [_flatten_trial(item) for item in audit_results + full_results]
    with (results_dir / "trial_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["stage"])
        writer.writeheader()
        writer.writerows(rows)
    with (results_dir / "paired_intervals.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "condition_id", "benchmark", "delay", "metric", "comparator",
            "confirmatory", "n", "mean", "lower", "upper",
            "ci_excludes_zero_positive",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in intervals])

    def number(value: Any, digits: int = 4) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.{digits}f}"

    color = gate["color"]
    status_meaning = {
        "GREEN": "All scientific criteria and protocol-validity checks passed.",
        "YELLOW": "The protocol was valid, but at least one scientific criterion failed.",
        "RED": "At least one protocol-validity check failed; do not interpret the scientific gate as confirmatory.",
    }[color]
    lines = [
        "# External-validation cascade report",
        "",
        f"Final scientific status: **{color}**",
        "",
        status_meaning,
        "",
        f"Selected global-SVD rank: **{rank_audit['selected_rank']}**",
        "",
        "## ICLR continuation gate",
        "",
        "| Criterion | Pass |",
        "|---|---:|",
    ]
    for name, passed in gate["criteria"].items():
        lines.append(f"| {name.replace('_', ' ')} | {'yes' if passed else 'no'} |")
    lines.extend(
        [
            "",
            "## Protocol-validity checks",
            "",
            "| Check | Pass |",
            "|---|---:|",
        ]
    )
    for name, passed in gate["validity_checks"].items():
        lines.append(f"| {name.replace('_', ' ')} | {'yes' if passed else 'no'} |")

    selected = gate["selected_rank_audit"]
    lines.extend(
        [
            "",
            "## Matched exact-credit audit",
            "",
            "| Benchmark | Delay | Accuracy gap (pp) | Relative excess loss | Median cosine | Predictions after feedback | Cell pass |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in selected["cells"]:
        lines.append(
            "| {benchmark} | {delay} | {accuracy} | {loss} | {cosine} | {predictions} | {passed} |".format(
                benchmark=cell["benchmark"],
                delay=cell["delay"],
                accuracy=number(cell["mean_accuracy_gap_percentage_points"]),
                loss=number(cell["mean_relative_excess_loss"]),
                cosine=number(cell["median_gradient_cosine"]),
                predictions=cell["minimum_predictions_after_feedback"],
                passed="yes" if cell["pass"] else "no",
            )
        )

    oracle = gate["oracle_diagnostics"]
    lines.extend(
        [
            "",
            "## Learner-competence diagnostics",
            "",
            "Yearbook immediate-oracle mean balanced accuracy by delay: "
            + ", ".join(
                f"{delay}={number(value)}"
                for delay, value in oracle[
                    "yearbook_mean_balanced_accuracy_by_delay"
                ].items()
            ),
            "",
            "Criteo immediate-oracle mean relative test-log-loss gain over the constant predictor: "
            + number(oracle["mean_criteo_relative_logloss_gain_over_constant"]),
            "",
            "## Confirmatory paired intervals",
            "",
            "| Condition | Comparator | Metric | Mean advantage | 95% CI | Positive |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in intervals:
        lines.append(
            "| {condition} | {comparator} | {metric} | {mean} | [{lower}, {upper}] | {positive} |".format(
                condition=row["condition_id"],
                comparator=row["comparator"],
                metric=row["metric"],
                mean=number(row["mean"], 6),
                lower=number(row["lower"], 6),
                upper=number(row["upper"], 6),
                positive="yes" if row["ci_excludes_zero_positive"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "Positive-CI confirmatory conditions by comparator: "
            + "; ".join(
                f"{name}={conditions or 'none'}"
                for name, conditions in gate[
                    "positive_ci_conditions_by_comparator"
                ].items()
            ),
            "",
            "The trial-level CSV contains predictive performance, credit error, gradient cosine, actual state bytes, peak RAM/VRAM, update latency, and throughput. Delay-stratified metrics remain in each JSON trial file.",
            "",
        ]
    )
    (results_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
