# Reproducibility Guide

## Scope

The repository contains all source, focused tests, frozen protocols, and numerical results needed to audit the controlled and external decisions. The controlled tasks are synthetic and generated from deterministic seed namespaces. The external study downloads the Yearbook and Criteo sources at run time; neither dataset is redistributed.

Full `.pt` checkpoints are excluded from Git history. The original result archives and their hashes are recorded in `results/PROVENANCE.json` and should accompany the tagged release as external assets.

## Environment

- Python 3.10 or newer
- NumPy 1.26 or newer
- PyTorch 2.2 or newer
- CUDA-capable PyTorch for full experiments
- The external launcher creates its own environment and installs the packages pinned in `external_validation/requirements.txt`

The final stacked run used PyTorch 2.8.0+cu128 on an NVIDIA RTX 6000 Ada with 47.37 GiB reported device memory.

## Fast audit

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

The verifiers exit nonzero if a committed summary no longer matches its frozen decision, state budget, trial count, or key gate results.

## External-validation launcher

Run from the repository root:

```bash
cd external_validation
bash run_external_validation.sh
```

The default is the full five-seed v1.1 cascade. `RUN_MODE=verify` runs its local integrity checks and `RUN_MODE=smoke` runs a small synthetic end-to-end check. The launcher downloads only missing material, uses pinned upstream revisions, records local logs, and resumes completed trials. It does not use Weights & Biases.

The frozen run used an NVIDIA RTX 6000 Ada. Its exact environment, source revisions, feature allow-list, split hashes, and data object hashes are in `results/external_validation_v1.1/provenance.json` and `external_validation/config/frozen_protocol.json`.

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
- `results/external_validation_v1.1/`: 256 completed trial JSON files, the 240-row analysis table, paired intervals, frozen protocol, provenance, and terminal report. Calibration trials are retained in `trials/` but are intentionally absent from `trial_summary.csv`.

The JSON files are the authoritative machine-readable evidence. `docs/RESULTS.md` is a human-readable interpretation of those files.

## Independent checks

Before publishing a modified release:

```bash
python -m unittest discover -s tests -v
python scripts/verify_frozen_results.py
PYTHONPATH=external_validation/src python -m unittest discover -s external_validation/tests -v
python scripts/verify_external_results.py
python -m ruff check src tests scripts external_validation/src external_validation/tests
python -m zipfile -t low-rank-eligibility-traces-v0.2.0-github.zip
```

Do not replace a frozen result file with a rerun under the same version. New experiments should use a new protocol version, output directory, release tag, and Zenodo version.
