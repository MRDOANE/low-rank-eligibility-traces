# N08+O10 optimizer-fair stacked eligibility repair v1.1

## Decision being run

v1.0 is invalid rather than negative: the exact trace failed its cell-level learning gate in
10/96 confirmation cells, while the candidate alone used a calibrated 0.001 learning rate and all
controls used 0.002. v1.1 performs the single authorized apparatus repair. It calibrates every
method by the same rule, validates those choices twice, and uses new teachers for all performance
inference.

The frozen pre-run probabilities are:

| Event | Probability |
|---|---:|
| Reference repair is valid | 90% |
| Fresh optimizer-fair screen advances | 65% |
| Confirmation is supportive | 45% |
| Confirmation establishes a strong stacked advantage | 20% |

The strong-positive probability is deliberately conservative: at latent rank 4 the equal-layer
sketch had higher premise cosine than the stacked candidate, so much of v1.0's apparent advantage
may disappear when its controls receive fair optimizer selection.

## Run on RunPod

Upload `adaptive-memory-platform-n08-o10-stacked-v1.1.zip` to `/workspace`, then run:

```bash
cd /workspace
mkdir -p n08-o10-stacked-v11
unzip adaptive-memory-platform-n08-o10-stacked-v1.1.zip -d n08-o10-stacked-v11
cd n08-o10-stacked-v11/adaptive-memory-platform

bash scripts/run_n08_o10_stacked_optimizer_repair.sh
```

If the CUDA interpreter is not selected automatically:

```bash
STACKED_PYTHON=/usr/local/bin/python bash scripts/run_n08_o10_stacked_optimizer_repair.sh
```

The launcher runs all relevant tests first and refuses to overwrite an existing v1.1 output.

## Monitor

From a second shell:

```bash
cd /workspace/n08-o10-stacked-v11/adaptive-memory-platform
watch -n 120 bash scripts/watch_n08_o10_stacked_optimizer_repair.sh
```

The reported ETA covers the active stage only. Later stages remain excluded until their frozen
gates authorize them.

## Gated workload

| Stage | Fits | Inferential role |
|---|---:|---|
| All-method calibration | 300 | Optimizer selection only |
| Clean calibration holdout | 30 | Optimizer validation only |
| Known-failure exact-reference audit | 12 | Apparatus validity only |
| Fresh screen | 60 | Gated evidence |
| Fresh confirmation | 480 | Primary evidence |
| Conditional boundary | 60 | Characterization only |

## RTX 6000 Ada wall-time estimate

The estimate is anchored to the returned v1.0 RTX 6000 Ada run, which took 77,579 seconds
(21.55 hours) for 36 short calibration fits, 60 screen fits, and 480 confirmation fits. Scaling
its measured per-horizon throughput on the same GPU gives:

| Frozen branch | RTX 6000 Ada estimate |
|---|---:|
| Invalid stop after holdout or reference audit | 8-11 hours |
| Valid decisive stop after fresh screen | 10-13 hours |
| Confirmation runs; boundary does not | 29-34 hours |
| Supportive result plus boundary | 32-37 hours |

The all-method calibration is now a meaningful part of the runtime: approximately 7-9 hours on
the same RTX 6000 Ada. This is the cost of removing the candidate-only optimizer confound.

GPU cost is the pod's displayed hourly rate multiplied by the applicable range above. RunPod
availability and rates vary, so the checkout rate for the actual pod is authoritative.

## Return file

On a successful terminal decision, the launcher creates:

```text
outputs/n08_o10_stacked_v11_results.zip
```

Attach that single archive. It contains the premise, all calibration and audit summaries, any
authorized screen/confirmation/boundary summaries, progress log, checkpoints, console log, and
measured wall time.
