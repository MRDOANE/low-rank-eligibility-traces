# N08+O10 stacked eligibility sketch v1.0

## Scientific purpose

N08 v2 established that a rank-four streaming low-rank eligibility trace can preserve delayed
learning at 34.03% of full persistent trace state. O10 v2 then showed that a fixed equal-layer
rank allocation retains almost all full-trace learning, but does not reliably beat a same-budget
global information-state control at extreme horizons.

This bridge tests one combined method: an adaptive right-subspace representation of the vertically
stacked layer eligibility matrices. Layer row blocks preserve layer identity, while one shared
right basis uses the state budget globally. At the O10 dimensions, it stores a rank-13 basis and
layer coefficients in 4,160 floats, versus 4,352 allocated floats for the rank-eight equal-layer
sketch and 16,384 floats for the exact trace.

The reference estimator chooses its basis from a left-energy-weighted covariance and then makes a
second pass over the materialized event factors. This is a controlled representation test, not yet
a one-pass streaming implementation. A positive result authorizes building the online subspace
tracker; it does not silently assume that engineering result.

The primary analysis uses a prespecified average of the second half of training, not the final
checkpoint. Independent teacher draws are the inferential units. Confirmation nests two training
replicates inside each of eight teachers so a single noisy endpoint cannot decide the result.

## Run

Upload `adaptive-memory-platform-n08-o10-stacked-v1.0.zip` to `/workspace`, then:

```bash
cd /workspace
mkdir -p n08-o10-stacked-v1
unzip adaptive-memory-platform-n08-o10-stacked-v1.0.zip -d n08-o10-stacked-v1
cd n08-o10-stacked-v1/adaptive-memory-platform

bash scripts/run_n08_o10_stacked_cascade.sh
```

To force a CUDA-enabled interpreter:

```bash
STACKED_PYTHON=/usr/local/bin/python bash scripts/run_n08_o10_stacked_cascade.sh
```

The launcher refuses to overwrite an existing v1.0 output.

## Monitor

From a second shell:

```bash
cd /workspace/n08-o10-stacked-v1/adaptive-memory-platform
watch -n 120 bash scripts/watch_n08_o10_stacked.sh
```

The ETA covers only the active stage. Confirmation and boundary stages are excluded until their
frozen gates select them.

## Gated stages

1. Premise and state-accounting audit.
2. Candidate-only learning-rate calibration on independent teachers.
3. Three-teacher screen at 8x and 32x horizons.
4. Eight-teacher confirmation at 8x, 16x, and 32x, with two nested training replicates.
5. Conditional rank-eight/rank-twelve boundary study at 16x and 32x.

## Estimated wall time

The estimate is anchored to the completed O10 v2 A40 run, which required 58,183 seconds for 990
calibration and learned fits. This bridge runs fewer fits but concentrates them at the expensive
16x and 32x horizons and evaluates both averaged and endpoint weights.

| Frozen branch | A40 estimate | RTX 6000 Ada / RTX 4090 estimate |
|---|---:|---:|
| Decisive stop after the screen | 2-4 hours | 1.5-3 hours |
| Confirmation runs, boundary does not | 23-29 hours | 16-23 hours |
| Supportive or strong result plus boundary | 27-34 hours | 19-27 hours |

The adaptive subspace calculation is vectorized over time. Its largest timing uncertainty is the
batched 64-by-64 eigendecomposition and weighted covariance at the 32x horizon.

## Estimated RunPod cost

RunPod's public pages showed A40 pod rates of approximately $0.35-$0.44 per hour and RTX 6000 Ada
rates of approximately $0.74-$0.84 per hour on 2026-08-29. Community, secure-cloud, storage, and
availability choices can change the checkout rate, so the pod's displayed rate is authoritative.

| Frozen branch | A40 estimate | RTX 6000 Ada estimate |
|---|---:|---:|
| Decisive stop after the screen | $0.70-$1.76 | $1.11-$2.52 |
| Confirmation runs, boundary does not | $8.05-$12.76 | $11.84-$19.32 |
| Supportive or strong result plus boundary | $9.45-$14.96 | $14.06-$22.68 |

These figures cover GPU time only. Sources:
`https://www.runpod.io/gpu-models` and
`https://www.runpod.io/articles/guides/ai-server-cost`.

## Return file

On success the launcher creates:

```text
outputs/n08_o10_stacked_v10_results.zip
```

Attach that single archive. It contains all summaries, the progress log, checkpoints, console log,
and measured wall time.
