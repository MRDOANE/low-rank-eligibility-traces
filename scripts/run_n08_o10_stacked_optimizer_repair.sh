#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"
export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"

choose_cuda_python() {
  local candidate
  for candidate in "${STACKED_PYTHON:-}" /usr/local/bin/python python; do
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
  echo "Set STACKED_PYTHON=/path/to/python if your CUDA environment is elsewhere." >&2
  exit 1
}

output_root="outputs/n08_o10_stacked_v11"
running_root="${output_root}.running"
result_zip="outputs/n08_o10_stacked_v11_results.zip"
console_log="outputs/n08_o10_stacked_v11_console.log"
if [[ -e "${output_root}" || -e "${running_root}" || -e "${result_zip}" ]]; then
  echo "Refusing to overwrite an existing N08+O10 stacked v1.1 output." >&2
  echo "Preserve and rename only the specific old output before rerunning." >&2
  exit 1
fi
mkdir -p outputs

"${python_bin}" - <<'PY' | tee "${console_log}"
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("GPU memory GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
PY

"${python_bin}" -m unittest \
  tests.test_o10_tasks \
  tests.test_o10_layerwise \
  tests.test_o10_horizon_experiment \
  tests.test_n08_tasks \
  tests.test_n08_traces \
  tests.test_n08_pareto_experiment \
  tests.test_n08_o10_stacked_experiment \
  tests.test_n08_o10_stacked_optimizer_repair -v 2>&1 | tee -a "${console_log}"

started_at="${SECONDS}"
set +e
"${python_bin}" -m adaptive_memory.n08_o10_stacked_optimizer_repair \
  --output-dir "${running_root}" \
  --device cuda \
  --calibration-steps 350 \
  --calibration-holdout-steps 350 \
  --reference-audit-steps 550 \
  --screen-steps 350 \
  --confirmation-steps 550 \
  --boundary-steps 450 \
  --batch-size 8 \
  --primary-eval-examples 256 2>&1 | tee -a "${console_log}"
status=${PIPESTATUS[0]}
set -e

mkdir -p "${output_root}"
if [[ -d "${running_root}" ]]; then
  if [[ "${status}" -eq 0 ]]; then
    mv "${running_root}" "${output_root}/run"
  else
    mv "${running_root}" "${output_root}/run_partial"
  fi
fi
printf '%s\n' "$((SECONDS - started_at))" > "${output_root}/wall_runtime_seconds.txt"
cp "${console_log}" "${output_root}/console.log"

if [[ "${status}" -ne 0 ]]; then
  echo "N08+O10 stacked v1.1 cascade failed with status ${status}." >&2
  echo "Partial files remain in ${output_root}." >&2
  exit "${status}"
fi

cp "${output_root}/run/cascade_summary.json" "${output_root}/cascade_summary.json"
for summary in \
  cascade_summary.json \
  run/premise_summary.json \
  run/optimizer_calibration_summary.json \
  run/calibration_holdout_summary.json \
  run/reference_audit_summary.json \
  run/screen_summary.json \
  run/confirmation_summary.json \
  run/boundary_summary.json \
  run/progress.jsonl \
  wall_runtime_seconds.txt; do
  if [[ -f "${output_root}/${summary}" ]]; then
    ls -lh "${output_root}/${summary}"
  fi
done

zip -q -r "${result_zip}" "${output_root}"
unzip -t "${result_zip}" >/dev/null

echo
echo "N08+O10 stacked v1.1 cascade complete. Attach this one results archive:"
ls -lh "${result_zip}"
