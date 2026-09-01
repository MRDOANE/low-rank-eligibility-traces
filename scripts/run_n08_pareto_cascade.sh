#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"
export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"

choose_cuda_python() {
  local candidate
  for candidate in "${N08_PYTHON:-}" /usr/local/bin/python python; do
    [[ -n "${candidate}" ]] || continue
    command -v "${candidate}" >/dev/null 2>&1 || continue
    if "${candidate}" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
    then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

python_bin="$(choose_cuda_python)" || {
  echo "No Python interpreter with working CUDA was found." >&2
  echo "Set N08_PYTHON=/path/to/python if your CUDA environment is elsewhere." >&2
  exit 1
}

"${python_bin}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
PY

output_root="outputs/n08_pareto_v20"
if [[ -e "${output_root}" || -e "${output_root}.running" ]]; then
  echo "Refusing to overwrite ${output_root} or ${output_root}.running" >&2
  exit 1
fi

"${python_bin}" -m unittest \
  tests.test_n08_tasks \
  tests.test_n08_traces \
  tests.test_n08_experiment \
  tests.test_n08_pareto_experiment -v

mkdir -p "${output_root}"
started_at="${SECONDS}"
set +e
"${python_bin}" -m adaptive_memory.n08_pareto_experiment \
  --output-dir "${output_root}.running" \
  --device cuda \
  --calibration-steps 250 \
  --primary-steps 550 \
  --frontier-steps 550 \
  --boundary-steps 550 \
  --batch-size 64
status=$?
set -e

if [[ -d "${output_root}.running" ]]; then
  mv "${output_root}.running" "${output_root}/run"
fi
printf '%s\n' "$((SECONDS - started_at))" > "${output_root}/wall_runtime_seconds.txt"

if [[ "${status}" -ne 0 ]]; then
  echo "N08 v2 cascade failed with status ${status}; partial files remain in ${output_root}." >&2
  exit "${status}"
fi

cp "${output_root}/run/cascade_summary.json" "${output_root}/cascade_summary.json"
for summary in \
  cascade_summary.json \
  run/premise_summary.json \
  run/calibration_summary.json \
  run/primary_summary.json \
  run/rescue_summary.json \
  run/frontier_summary.json \
  run/boundary_summary.json \
  run/progress.jsonl \
  wall_runtime_seconds.txt; do
  if [[ -f "${output_root}/${summary}" ]]; then
    ls -lh "${output_root}/${summary}"
  fi
done

echo
echo "Attach these files when the run finishes:"
echo "  ${output_root}/cascade_summary.json"
echo "  ${output_root}/run/primary_summary.json"
echo "  ${output_root}/run/rescue_summary.json (if present)"
echo "  ${output_root}/run/frontier_summary.json (if present)"
echo "  ${output_root}/run/boundary_summary.json (if present)"
echo "  ${output_root}/wall_runtime_seconds.txt"
