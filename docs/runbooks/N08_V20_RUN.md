# N08 v2.0 RunPod handoff

## Run

```bash
cd /workspace
unzip -o adaptive-memory-platform-n08-v2.0.zip
cd adaptive-memory-platform
bash scripts/run_n08_pareto_cascade.sh
```

The script finds a CUDA-enabled Python automatically. To force a particular interpreter:

```bash
N08_PYTHON=/usr/local/bin/python bash scripts/run_n08_pareto_cascade.sh
```

It refuses to overwrite `outputs/n08_pareto_v20`.

## Monitor

From a second shell:

```bash
cd /workspace/adaptive-memory-platform
bash scripts/watch_n08_pareto.sh
```

The stage ETA becomes useful after several low-rank runs. Conditional later stages are deliberately
not included until the branch is selected.

## Estimated RTX 2000 Ada runtime

The estimate uses the measured v1.0 throughput: one rank-two, length-24 candidate run took about
141 seconds for 400 steps, while the full and control methods took about 3 to 4 seconds each.

| Branch selected by frozen gates | Estimated total time |
|---|---:|
| Decisive rank-two failure | 14-18 hours |
| Rank-two positive plus frontier and boundary | 22-30 hours |
| Rank-four rescue branch | 28-40 hours |

Rank-four small-matrix SVD throughput is the largest uncertainty. The live monitor reports the ETA
for the current stage from the actual pod.

## Return files

Always attach:

```text
outputs/n08_pareto_v20/cascade_summary.json
outputs/n08_pareto_v20/run/primary_summary.json
outputs/n08_pareto_v20/wall_runtime_seconds.txt
```

Also attach any conditional summaries that exist:

```text
outputs/n08_pareto_v20/run/rescue_summary.json
outputs/n08_pareto_v20/run/frontier_summary.json
outputs/n08_pareto_v20/run/boundary_summary.json
```
