# Low-Rank Eligibility Traces for Delayed Credit Assignment

[![Tests](https://github.com/MRDOANE/low-rank-eligibility-traces/actions/workflows/tests.yml/badge.svg)](https://github.com/MRDOANE/low-rank-eligibility-traces/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the reference implementation, frozen protocols, tests, and compact result summaries for a study of compressed eligibility traces. Eligibility traces store information about earlier events so that a model can learn when feedback arrives much later. The study asks whether low-rank traces can preserve delayed-credit learning while using substantially less persistent state than an exact trace.

## Main result

The project proceeded through three prospectively gated experiments:

| Experiment | Frozen outcome | Persistent trace state | Main interpretation |
|---|---|---:|---|
| N08 v2 | Strong Pareto positive | 34.03% of exact | Rank-four streaming traces retained exact-trace learning and beat matched replay at long horizons. |
| O10 v2 | Bounded negative | 26.56% of exact | Equal-layer rank-eight traces remained accurate, but layer-specific superiority over the global oracle was not established. |
| Stacked v1.1 | Supportive, not strong | 25.39% of exact | The shared-right-subspace trace preserved exact-trace learning and modestly beat equal-layer and replay controls under optimizer-fair evaluation. |

In the final confirmation, the stacked method retained 100.036% of full-trace learning (95% teacher-bootstrap interval 100.005% to 100.093%) with a mean gradient cosine of 0.9786. Its gains were positive over equal-layer and replay controls, but very small in the primary regime, and the batch global-SVD information oracle remained slightly better. The prespecified high-rank boundary showed much larger gains over equal-layer and replay while gradient cosine fell, defining where the approximation stopped preserving the exact update direction.

## Claim boundary

The implementation establishes an information-representation result. It materializes event factors and uses two passes to estimate and apply the shared subspace. The evidence therefore does **not** establish one-pass online operation, lower peak CUDA memory, lower wall time, or superiority to the global-SVD oracle. See [docs/CLAIM_BOUNDARY.md](docs/CLAIM_BOUNDARY.md).

## Quick start

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python -m unittest discover -s tests -v
python scripts/verify_frozen_results.py
```

The unit suite exercises trace construction, state accounting, finite-difference checks, optimizer selection, gate logic, and deterministic fixtures. The result verifier checks that the committed summaries reproduce the frozen terminal decisions.

## Repository layout

```text
src/adaptive_memory/   Reference implementations
tests/                 Focused unit and decision tests
scripts/               GPU launchers, monitors, and result verifier
docs/protocols/         Frozen experimental protocols
docs/runbooks/          Original RunPod commands
results/                Compact frozen summaries; no model checkpoints
paper/                  Manuscript scaffold
```

## Reproducing experiments

The exact commands and stage definitions are documented in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). Full reruns require a CUDA-capable PyTorch environment and substantial GPU time. The final stacked v1.1 run took 121,194 seconds on an NVIDIA RTX 6000 Ada.

## Results and provenance

- [Result interpretation](docs/RESULTS.md)
- [Raw-archive provenance](results/PROVENANCE.json)
- [Frozen N08 protocol](docs/protocols/N08_V20_PROTOCOL.md)
- [Frozen O10 protocol](docs/protocols/O10_V20_PROTOCOL.md)
- [Frozen stacked v1.1 protocol](docs/protocols/N08_O10_STACKED_V11_PROTOCOL.md)

Large checkpoint archives are intentionally excluded from Git history. Their exact filenames, sizes, and SHA-256 hashes are recorded in `results/PROVENANCE.json`; they should be attached to the GitHub release and Zenodo record.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). After the first Zenodo release, add the version DOI to `CITATION.cff` and the DOI badge to this README.

## License

Code and documentation are released under the [MIT License](LICENSE). Frozen numerical result files are included for research transparency and reproducibility.
