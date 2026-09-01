# O10 v2.0 RunPod handoff

## Run

Upload `adaptive-memory-platform-o10-v2.0.zip` to `/workspace`, then:

```bash
cd /workspace
mkdir -p o10-v2
unzip adaptive-memory-platform-o10-v2.0.zip -d o10-v2
cd o10-v2/adaptive-memory-platform

bash scripts/run_o10_horizon_cascade.sh
```

The launcher locates a CUDA-enabled Python automatically. To force one:

```bash
O10_PYTHON=/usr/local/bin/python bash scripts/run_o10_horizon_cascade.sh
```

It refuses to overwrite any existing O10 v2 output.

## Monitor

From a second shell:

```bash
cd /workspace/o10-v2/adaptive-memory-platform
watch -n 120 bash scripts/watch_o10_horizon.sh
```

The ETA covers the current stage. Conditional rescue and 16x/32x stages are deliberately excluded
until their frozen branch is selected.

## Estimated RTX 6000 Ada runtime

The estimate is anchored to O10 v1.0 on the RTX 2000 Ada: standard 300-step, batch-eight,
length-512 runs took about 18 seconds, while the multi-timescale control took about 33 seconds. The
RTX 6000 Ada should accelerate the matrix work, but NumPy teacher-batch construction and many small
factor operations limit scaling.

| Frozen branch | Estimated total wall time |
|---|---:|
| Decisive rank-eight outcome after calibration and five-replicate primary | 3-6 hours |
| Promising rank-eight trend plus actual 16x/32x extension | 7-13 hours |
| Rank-twelve rescue that then stops | 7-12 hours |
| Rank-twelve rescue plus 16x/32x extension | 11-20 hours |

The exact global-SVD control is the largest timing uncertainty. The live monitor becomes more useful
after the first 20 runs in each stage.

## Return file

On success the launcher creates:

```text
outputs/o10_horizon_v20_results.zip
```

Attach that one file. It contains the cascade decision, calibration, primary and conditional
summaries, progress log, checkpoints, console log, and measured wall time.

If packaging is interrupted after the scientific run has finished, attach at minimum:

```text
outputs/o10_horizon_v20/cascade_summary.json
outputs/o10_horizon_v20/run/primary_summary.json
outputs/o10_horizon_v20/run/extension_summary.json   # if present
outputs/o10_horizon_v20/run/rescue_summary.json      # if present
outputs/o10_horizon_v20/wall_runtime_seconds.txt
```
