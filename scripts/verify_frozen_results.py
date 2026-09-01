#!/usr/bin/env python3
"""Verify that committed summaries reproduce the frozen project decisions."""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKS = 0


def load(relative_path: str) -> dict:
    with (ROOT / relative_path).open(encoding="utf-8") as handle:
        return json.load(handle)


def expect(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, message: str) -> None:
    expect(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), message)


def verify_n08() -> None:
    cascade = load("results/n08_v2/cascade_summary.json")
    rescue = load("results/n08_v2/rescue_summary.json")["gate"]

    expect(cascade["gate_passed"] is True, "N08 terminal gate must pass")
    expect(cascade["selected_trace_rank"] == 4, "N08 selected rank must remain four")
    expect(cascade["strong_pareto_signal"] is True, "N08 must retain strong Pareto status")
    expect(
        cascade["decision"] == "n08_v2_strong_pareto_positive_continue_to_o10",
        "Unexpected N08 terminal decision",
    )
    close(rescue["state"]["candidate_fraction_of_full"], 0.3402777777777778, "N08 state")
    close(
        rescue["bootstrap_over_seed_aggregates"]["candidate_gradient_cosine"]["mean"],
        0.9999571055571238,
        "N08 gradient cosine",
    )
    expect(all(rescue["checks"].values()), "Every N08 rescue check must remain true")


def verify_o10() -> None:
    cascade = load("results/o10_v2/cascade_summary.json")
    extension = load("results/o10_v2/extension_summary.json")["analysis"]

    expect(cascade["gate_passed"] is False, "O10 terminal gate was prospectively negative")
    expect(cascade["selected_trace_rank"] == 8, "O10 selected rank must remain eight")
    expect(
        cascade["layerwise_specificity_established"] is False,
        "O10 did not establish layerwise specificity",
    )
    expect(
        cascade["decision"] == "stop_o10_v2_after_decisive_actual_horizon_test_move_to_n10",
        "Unexpected O10 terminal decision",
    )
    close(extension["state"]["candidate_fraction_of_full"], 0.265625, "O10 state")
    close(
        extension["bootstrap_over_paired_replicates"]["candidate_gradient_cosine"]["mean"],
        0.9871400408961556,
        "O10 gradient cosine",
    )


def verify_stacked_confirmation() -> None:
    cascade = load("results/stacked_v1.1/cascade_summary.json")
    confirmation = load("results/stacked_v1.1/confirmation_summary.json")["analysis"]
    bootstrap = confirmation["bootstrap_over_independent_teachers"]

    expect(cascade["confirmation_supportive"] is True, "Stacked confirmation must be supportive")
    expect(cascade["strong_positive"] is False, "Stacked confirmation was not strong")
    expect(cascade["project_viable"] is True, "Stacked project must remain viable")
    expected_decision = (
        "stacked_trace_optimizer_fair_supportive_without_specific_advantage_preserve_claim_boundary"
    )
    expect(cascade["decision"] == expected_decision, "Unexpected stacked terminal decision")
    expect(confirmation["reference_valid"] is True, "Stacked reference must be valid")
    expect(confirmation["supportive"] is True, "Confirmation supportive flag changed")
    expect(confirmation["strong_positive"] is False, "Confirmation strong flag changed")
    expect(all(confirmation["core_checks"].values()), "Every supportive core check must pass")
    expect(
        confirmation["strong_checks"]["candidate_layerwise_gain_is_at_least_0_01_of_initial_error"]
        is False,
        "The frozen strong-effect threshold must remain failed",
    )
    expect(
        confirmation["strong_checks"]["candidate_layerwise_gain_lower_bound_is_positive"] is True,
        "The overall layerwise lower bound must remain positive",
    )
    close(confirmation["state"]["candidate_fraction_of_full"], 0.25390625, "Stacked state")
    close(
        bootstrap["candidate_full_gain_retention"]["mean"],
        1.0003627746712433,
        "Stacked gain retention",
    )
    close(
        bootstrap["candidate_gradient_cosine"]["mean"],
        0.9786455958968763,
        "Stacked gradient cosine",
    )


def verify_stacked_boundary() -> None:
    boundary = load("results/stacked_v1.1/boundary_summary.json")["analysis"]
    bootstrap = boundary["bootstrap_over_independent_teachers"]
    wins = boundary["wins"]

    expect(boundary["reference_valid"] is True, "Boundary reference must be valid")
    expect(boundary["supportive"] is False, "Boundary did not pass the cosine gate")
    expect(wins["candidate_better_than_global_oracle"] == 0, "Global boundary wins changed")
    expect(wins["candidate_better_than_layerwise"] == 12, "Layerwise boundary wins changed")
    expect(wins["candidate_better_than_replay"] == 12, "Replay boundary wins changed")
    close(
        bootstrap["candidate_gradient_cosine"]["mean"],
        0.7804349718601614,
        "Boundary gradient cosine",
    )


def verify_provenance() -> None:
    provenance = load("results/PROVENANCE.json")
    archives = {item["filename"]: item for item in provenance["raw_result_archives"]}

    expect(
        provenance["source_package"]["sha256"]
        == "9a00e7a56a51c14192b5a2332ba8374230175fb81508e86e7f8ba8ce9c5ba28a",
        "Source-package hash changed",
    )
    expect(
        archives["n08_pareto_v20_results.zip"]["sha256"]
        == "76cc56313afd4035e0de404e2368438fb90a8df4b3ea187f314ad17a77eb787d",
        "N08 archive hash changed",
    )
    expect(
        archives["o10_horizon_v20_results.zip"]["sha256"]
        == "588d3697ced7f9124cec00a5f4c7bf00c67f96d94e3e003c06c1ba5c2e5cec77",
        "O10 archive hash changed",
    )
    expect(
        archives["n08_o10_stacked_v11_results.zip"]["sha256"]
        == "17d5ffac380b1de1784d96f841cd77e1109d673bf71920bc6d3805d57573c9a8",
        "Stacked v1.1 archive hash changed",
    )


def main() -> None:
    verify_n08()
    verify_o10()
    verify_stacked_confirmation()
    verify_stacked_boundary()
    verify_provenance()
    print(json.dumps({"status": "verified", "checks": CHECKS, "release": "0.1.0"}))


if __name__ == "__main__":
    main()
