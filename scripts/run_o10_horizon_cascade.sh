#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"
export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"

choose_cuda_python() {
  local candidate
  for candidate in "${O10_PYTHON:-}" /usr/local/bin/python python; do
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
  echo "Set O10_PYTHON=/path/to/python if your CUDA environment is elsewhere." >&2
  exit 1
}

"${python_bin}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("GPU memory GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
PY

output_root="outputs/o10_horizon_v20"
running_root="${output_root}.running"
result_zip="outputs/o10_horizon_v20_results.zip"
console_log="outputs/o10_horizon_v20_console.log"
if [[ -e "${output_root}" || -e "${running_root}" || -e "${result_zip}" ]]; then
  echo "Refusing to overwrite an existing O10 v2 output." >&2
  echo "Remove or rename only the specific old O10 v2 path after preserving it." >&2
  exit 1
fi
mkdir -p outputs

"${python_bin}" -m unittest \
  tests.test_o10_tasks \
  tests.test_o10_layerwise \
  tests.test_o10_experiment \
  tests.test_o10_horizon_experiment -v

started_at="${SECONDS}"
set +e
"${python_bin}" -m adaptive_memory.o10_horizon_experiment \
  --output-dir "${running_root}" \
  --device cuda \
  --calibration-steps 200 \
  --primary-steps 550 \
  --rescue-steps 550 \
  --extension-steps 550 \
  --primary-batch-size 8 2>&1 | tee "${console_log}"
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
  echo "O10 v2 cascade failed with status ${status}; partial files remain in ${output_root}." >&2
  exit "${status}"
fi

cp "${output_root}/run/cascade_summary.json" "${output_root}/cascade_summary.json"
for summary in \
  cascade_summary.json \
  run/premise_summary.json \
  run/primary_calibration_summary.json \
  run/primary_summary.json \
  run/rescue_summary.json \
  run/extension_summary.json \
  run/progress.jsonl \
  wall_runtime_seconds.txt; do
  if [[ -f "${output_root}/${summary}" ]]; then
    ls -lh "${output_root}/${summary}"
  fi
done

zip -q -r "${result_zip}" "${output_root}"
unzip -t "${result_zip}" >/dev/null

echo
echo "O10 v2 cascade complete. Attach this one results archive:"
ls -lh "${result_zip}"
