#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"

if [[ -f outputs/o10_horizon_v20.running/progress.jsonl ]]; then
  progress="outputs/o10_horizon_v20.running/progress.jsonl"
elif [[ -f outputs/o10_horizon_v20/run/progress.jsonl ]]; then
  progress="outputs/o10_horizon_v20/run/progress.jsonl"
else
  echo "No O10 v2 progress file exists yet." >&2
  exit 1
fi

python - "${progress}" <<'PY'
import datetime as dt
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
last = rows[-1]
print("Progress file:", path)
print("Current stage:", last.get("stage", last.get("event", "unknown")))
if "completed" in last:
    completed = int(last["completed"])
    total = int(last["total"])
    print(f"Stage progress: {completed}/{total} ({100 * completed / total:.1f}%)")
if "method" in last:
    print("Last completed method:", last["method"])
if "horizon_multiplier" in last:
    print("Last trained horizon:", f"{last['horizon_multiplier']}x")
if "replicate" in last:
    print("Last replicate:", last["replicate"])
eta = last.get("eta_seconds")
if eta is not None:
    finish = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=float(eta))
    print(f"Current-stage ETA: {float(eta) / 3600:.2f} hours")
    print("Current-stage estimated finish:", finish.strftime("%Y-%m-%d %H:%M:%S UTC"))
stages = list(dict.fromkeys(row.get("stage", row.get("event", "unknown")) for row in rows))
print("Stages observed:", ", ".join(stages))
print("Note: ETA covers only the active branch stage; conditional rescue/extension is excluded.")
PY
