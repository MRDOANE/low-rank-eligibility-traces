# Low-Rank Delayed-Credit States

[![Tests](https://github.com/MRDOANE/low-rank-eligibility-traces/actions/workflows/tests.yml/badge.svg)](https://github.com/MRDOANE/low-rank-eligibility-traces/actions/workflows/tests.yml)
DOI: 10.5281/zenodo.22217984
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository studies what an online learner should retain when a prediction is made now and its label arrives later. The exact delayed-credit state stores the observation-time logit Jacobian for each pending example. Low-rank alternatives compress that state before the outcome is known, then reconstruct it when feedback arrives.

The central result has two parts. Rank-four global SVD reproduces exact delayed credit with high fidelity and substantially less persistent state. That fidelity does not translate into an advantage over input replay or a matched random sketch on the two external streams. The project therefore provides an empirical account of compression fidelity, learning utility, and systems cost rather than a claim of a universally superior delayed learner.

## Results at a glance

### External validation v1.1

The preregistered external study used five main seeds, ranks 2/4/8, the 37,921-image Yearbook archive under fixed delays of 10, 50, and 100 batches, and the chronological Criteo attribution stream under observed conversion delays and a fixed negative-label maturity rule.

| Result | Frozen finding |
|---|---:|
| Selected global-SVD rank | 4 |
| Exact/global per-event state ratio | 6.3519x |
| Median audit gradient cosine | 0.9978 to 0.9999 |
| Largest audit accuracy gap from exact credit | 0.2087 percentage points |
| Full Yearbook advantage over equal-byte replay | -13.70, -7.69, -2.90 points at delays 10, 50, 100 |
| Confirmatory wins with a positive 95% paired interval | 0 |
| Terminal status | YELLOW: valid protocol, competitive criterion failed |

The audit passed the exact-performance, gradient-cosine, feedback-exposure, and threefold-memory criteria in every benchmark/delay cell. The full comparison failed the preregistered requirement of beating the same fixed comparator at two independent conditions. Random projection was statistically tied at Yearbook delays 10 and 50, significantly better at delay 100, and inconclusive on Criteo. Equal-byte replay was substantially better on all three Yearbook delays; the Criteo interval was inconclusive. Global SVD also had unfavorable end-to-end throughput despite short median feedback-update latency.

### Controlled N08/O10 studies

| Experiment | Frozen outcome | Persistent trace state | Interpretation |
|---|---|---:|---|
| N08 v2 | Strong Pareto positive | 34.03% of exact | Rank-four streaming traces retained exact-trace learning and beat matched replay at long synthetic horizons. |
| O10 v2 | Bounded negative | 26.56% of exact | Equal-layer rank-eight traces remained accurate; layer-specific superiority over a global oracle was not established. |
| Stacked v1.1 | Supportive, not strong | 25.39% of exact | Shared-right-subspace traces preserved exact learning with small gains over equal-layer and replay controls; global SVD remained slightly better. |

Together, the controlled and external results identify a useful boundary: low-rank structure can preserve the exact update direction, while downstream value depends on whether exact observation-time credit is preferable to retaining inputs and recomputing after the model has changed.

## What is new here

Low-rank gradient approximation, eligibility traces, RTRL approximations, delayed online learning, and matrix sketching all predate this project. The contribution lies at their intersection:

1. a declared delayed-credit state boundary that retains label-free, observation-time Jacobians until asynchronous labels arrive;
2. exact, SVD, shared/layerwise, random-sketch, and replay representations compared under measured byte budgets;
3. separate tests of gradient fidelity and predictive utility on controlled tasks and two chronological external streams; and
4. evidence that near-exact gradient reconstruction can coexist with worse online prediction and throughput than simpler alternatives.

The dated literature assessment and prohibited novelty claims are in [docs/NOVELTY_AUDIT.md](docs/NOVELTY_AUDIT.md).

## Fast verification

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,external]"

python -m unittest discover -s tests -v
PYTHONPATH=external_validation/src python -m unittest discover -s external_validation/tests -v
python scripts/verify_frozen_results.py
python scripts/verify_external_results.py
```

These checks cover trace construction, finite-difference agreement, state accounting, optimizer selection, gate logic, protocol integrity, and the committed terminal decisions. Full external reproduction downloads the benchmarks rather than redistributing them; see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Repository layout

```text
src/adaptive_memory/                  Controlled N08/O10 implementations
external_validation/                 Two-benchmark v1.1 source and launcher
tests/                               Controlled-study unit tests
scripts/                             Result verifiers and synthetic launchers
docs/                                Results, protocols, claim and novelty audits
results/n08_v2/                      N08 frozen summaries
results/o10_v2/                      O10 frozen summaries
results/stacked_v1.1/                Stacked frozen summaries
results/external_validation_v1.1/    External trial-level results and decision
paper/                               TMLR-oriented manuscript outline
```

## Reproducing the external study

The source under `external_validation/` has one launcher:

```bash
cd external_validation
bash run_external_validation.sh
```

It pins upstream source revisions, downloads the Yearbook and Criteo data, removes any need for Weights & Biases, freezes preprocessing and chronological splits, resumes completed trials, and logs locally. The Criteo source is CC BY-NC-SA 4.0 and is never redistributed. Full details and third-party terms are in [external_validation/README.md](external_validation/README.md) and [external_validation/THIRD_PARTY.md](external_validation/THIRD_PARTY.md).

## Claim boundary

The evidence supports compression and fidelity claims for the tested adapter Jacobians. It does not support superiority over replay or sketching, higher throughput, state-of-the-art benchmark prediction, one-pass operation for every variant, or universal effectiveness across architectures. See [docs/CLAIM_BOUNDARY.md](docs/CLAIM_BOUNDARY.md).

## Citation

Project DOI: [10.5281/zenodo.22217985](https://doi.org/10.5281/zenodo.22217985). Machine-readable metadata are in [CITATION.cff](CITATION.cff).

## License and data

Original project code and documentation are released under the [MIT License](LICENSE). Third-party benchmark source and data retain their original terms. The repository contains derived numerical results and provenance records, not the Yearbook images, Criteo records, or upstream source trees.
