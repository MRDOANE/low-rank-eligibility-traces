# Reproducibility Guide

## Scope

The repository contains all source, focused tests, frozen protocols, and compact numerical summaries needed to audit the reported decisions. The tasks are synthetic and generated from deterministic seed namespaces; no external dataset or restricted material is required.

Full `.pt` checkpoints are excluded from Git history. The original result archives and their hashes are recorded in `results/PROVENANCE.json` and should accompany the tagged release as external assets.

## Environment

- Python 3.10 or newer
- NumPy 1.26 or newer
- PyTorch 2.2 or newer
- CUDA-capable PyTorch for full experiments

The final stacked run used PyTorch 2.8.0+cu128 on an NVIDIA RTX 6000 Ada with 47.37 GiB reported device memory.

## Fast audit

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python -m unittest discover -s tests -v
python scripts/verify_frozen_results.py
```

The verifier exits nonzero if a committed summary no longer matches its frozen decision, state budget, or key gate results.

## Full experiment entry points

Run each launcher from the repository root in a CUDA environment. Launchers refuse to overwrite their expected output directories.

| Study | Launcher | Original measured runtime |
|---|---|---:|
| N08 v2 | `bash scripts/run_n08_pareto_cascade.sh` | 53,760 s |
| O10 v2 | `bash scripts/run_o10_horizon_cascade.sh` | 58,183 s |
| Stacked v1.0 diagnosis | `bash scripts/run_n08_o10_stacked_cascade.sh` | See diagnostic protocol |
| Stacked v1.1 confirmation | `bash scripts/run_n08_o10_stacked_optimizer_repair.sh` | 121,194 s |

Original command lines, workloads, and monitoring commands are preserved under `docs/runbooks/`. Stage definitions, seed partitions, gates, bootstrap units, state budgets, and stopping rules are under `docs/protocols/`.

## Frozen result directories

- `results/n08_v2/`: N08 calibration, primary, rank-four rescue, and boundary summaries.
- `results/o10_v2/`: O10 calibration, primary, and long-horizon extension summaries.
- `results/stacked_v1.1/`: all-method calibration, holdout, reference audit, screen, confirmation, and boundary summaries.

The JSON files are the authoritative machine-readable evidence. `docs/RESULTS.md` is a human-readable interpretation of those files.

## Independent checks

Before publishing a modified release:

```bash
python -m unittest discover -s tests -v
python scripts/verify_frozen_results.py
python -m ruff check src tests scripts/verify_frozen_results.py
python -m zipfile -t low-rank-eligibility-traces-v0.1.0-github.zip
```

Do not replace a frozen result file with a rerun under the same version. New experiments should use a new protocol version, output directory, release tag, and Zenodo version.
