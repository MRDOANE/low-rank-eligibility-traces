#!/usr/bin/env python3
"""Verify the frozen external-validation result set and terminal decision."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "external_validation_v1.1"
CHECKS = 0


def load(name: str):
    with (RESULTS / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def expect(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, message: str) -> None:
    expect(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), message)


def verify_terminal_decision() -> None:
    status = load("final_status.json")
    gate = load("gate.json")

    expect(status["claim_evaluated"] is True, "The scientific claim must be evaluated")
    expect(status["mode"] == "full", "Only the full run can support the frozen decision")
    expect(status["color"] == "YELLOW", "The frozen terminal color must remain YELLOW")
    expect(status["continue_toward_iclr_claim"] is False, "The ICLR continuation gate failed")
    expect(status["selected_rank"] == 4, "The selected external rank must remain four")
    expect(gate["all_valid"] is True, "Every protocol-validity check must pass")
    expect(gate["all_pass"] is False, "The complete confirmatory gate did not pass")
    expect(gate["color"] == "YELLOW", "Gate and status color must agree")
    expect(gate["selected_rank"] == 4, "Gate and status rank must agree")
    expect(all(status["validity_checks"].values()), "Every status validity check must pass")
    expect(all(gate["validity_checks"].values()), "Every gate validity check must pass")

    criteria = gate["criteria"]
    expect(criteria["within_exact_performance_tolerance"] is True, "Exact tolerance changed")
    expect(
        criteria["gradient_cosine_at_least_0_95_in_every_audit_cell"] is True,
        "Gradient-cosine audit changed",
    )
    expect(
        criteria["credit_memory_reduction_at_least_3x"] is True,
        "Memory-reduction audit changed",
    )
    expect(
        criteria["paired_advantage_for_one_comparator_at_two_or_more_conditions"] is False,
        "Competitive-superiority outcome changed",
    )
    expect(gate["successful_comparators"] == [], "No comparator met the paired-win rule")


def verify_rank_audit() -> None:
    audit = load("rank_audit.json")
    selected = load("gate.json")["selected_rank_audit"]

    expect(audit["selected_rank"] == 4, "Rank audit must select rank four")
    expect([candidate["rank"] for candidate in audit["candidates"]] == [2, 4, 8], "Ranks changed")
    expect(
        [candidate["rank_pass"] for candidate in audit["candidates"]] == [False, True, True],
        "Rank outcomes changed",
    )
    expect(selected["audit_trials"] == 20, "Selected rank must contain twenty audit trials")
    expect(len(selected["cells"]) == 4, "Selected rank must contain four benchmark/delay cells")
    expect(
        all(cell["seeds"] == 5 for cell in selected["cells"]),
        "Every audit cell needs five seeds",
    )
    expect(
        all(cell["pass"] for cell in selected["cells"]),
        "Every selected-rank audit cell must pass",
    )
    close(selected["minimum_memory_reduction"], 6.351937984496124, "Memory ratio changed")
    close(
        selected["minimum_cell_median_gradient_cosine"],
        0.997817188501358,
        "Minimum gradient cosine changed",
    )
    close(
        selected["max_accuracy_gap_percentage_points"],
        0.208740234375,
        "Maximum accuracy gap changed",
    )


def verify_paired_intervals() -> None:
    intervals = load("paired_intervals.json")

    expect(len(intervals) == 8, "Expected eight confirmatory intervals")
    expect(all(item["confirmatory"] for item in intervals), "Every interval must be confirmatory")
    expect(
        not any(item["ci_excludes_zero_positive"] for item in intervals),
        "No interval may be recorded as a positive win",
    )
    counts = Counter(item["comparator"] for item in intervals)
    expect(
        counts == {"equal_byte_replay": 4, "random_projection": 4},
        "Comparator cells changed",
    )
    by_condition = {(item["condition_id"], item["comparator"]): item for item in intervals}
    close(
        by_condition[("yearbook:delay=10", "equal_byte_replay")]["mean"],
        -0.1369781451386943,
        "Yearbook delay-10 replay contrast changed",
    )
    close(
        by_condition[("yearbook:delay=100", "random_projection")]["upper"],
        -0.002203150565929327,
        "Yearbook delay-100 sketch interval changed",
    )


def verify_trial_inventory() -> None:
    trial_files = list((RESULTS / "trials").glob("*.json"))
    expect(len(trial_files) == 256, "Expected 256 trial JSON records")

    with (RESULTS / "trial_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expect(len(rows) == 240, "Expected 240 audit/full summary rows")
    expect(
        Counter(row["stage"] for row in rows) == {"audit": 80, "full": 160},
        "Stage counts changed",
    )

    required = {
        "final_status.json",
        "gate.json",
        "rank_audit.json",
        "paired_intervals.json",
        "paired_intervals.csv",
        "trial_summary.csv",
        "frozen_protocol.json",
        "provenance.json",
        "REPORT.md",
    }
    expect(
        required.issubset({path.name for path in RESULTS.iterdir()}),
        "A required artifact is missing",
    )


def verify_archive_provenance() -> None:
    provenance = json.loads(
        (ROOT / "results" / "PROVENANCE.json").read_text(encoding="utf-8")
    )
    source = provenance["external_validation_source_package"]
    archives = {item["filename"]: item for item in provenance["raw_result_archives"]}
    result = archives["n08_external_validation_cascade_v1.1.0_results.zip"]

    expect(source["size_bytes"] == 59270, "External source archive size changed")
    expect(
        source["sha256"] == "51fb9ef099167b5724fe21a85e7fc9a314d6b7023ce3dc282bdd3b99cbb21182",
        "External source archive hash changed",
    )
    expect(result["size_bytes"] == 586250, "External result archive size changed")
    expect(
        result["sha256"] == "868ec7bb45a17b17f95ef26faa229b45cf7330f23f53dc2ac4dff07aff80c37b",
        "External result archive hash changed",
    )


def main() -> None:
    verify_terminal_decision()
    verify_rank_audit()
    verify_paired_intervals()
    verify_trial_inventory()
    verify_archive_provenance()
    print(json.dumps({"status": "verified", "checks": CHECKS, "release": "0.2.0"}))


if __name__ == "__main__":
    main()
