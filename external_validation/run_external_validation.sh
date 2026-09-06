#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_MODE="${RUN_MODE:-full}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT}/results}"
legacy_root="$(dirname "${ROOT}")/n08_external_validation_cascade_v1.0.0"

if [[ -z "${DATA_DIR+x}" ]]; then
  if [[ -f "${legacy_root}/data/yearbook/frozen_feature_manifest.json" ]] || \
     [[ -f "${legacy_root}/data/criteo/frozen_preprocessing_manifest.json" ]]; then
    DATA_DIR="${legacy_root}/data"
  else
    DATA_DIR="${ROOT}/data"
  fi
fi
if [[ -z "${VENV_DIR+x}" ]]; then
  if [[ -x "${legacy_root}/.venv/bin/python" ]]; then
    VENV_DIR="${legacy_root}/.venv"
  else
    VENV_DIR="${ROOT}/.venv"
  fi
fi
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${LOG_DIR}" "${DATA_DIR}"

if ! command -v git >/dev/null 2>&1; then
  printf 'RED — git is required to fetch the pinned Yearbook protocol source.\n' >&2
  exit 2
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
launcher_log="${LOG_DIR}/launcher_${stamp}.log"
final_color="RED"

finish() {
  code=$?
  if [[ ${code} -ne 0 ]]; then
    printf '\nRED — cascade stopped with exit code %s. See %s\n' "${code}" "${launcher_log}" | tee -a "${launcher_log}"
  fi
}
trap finish EXIT

exec > >(tee -a "${launcher_log}") 2>&1

printf 'N08 external-validation cascade\n'
printf 'Mode: %s\nData: %s\nResults: %s\n' "${RUN_MODE}" "${DATA_DIR}" "${RESULTS_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
if ! "${VENV_PYTHON}" -c \
  'import huggingface_hub, numpy, pandas, PIL, psutil, pyarrow, scipy, tqdm' \
  >/dev/null 2>&1; then
  "${VENV_PYTHON}" -m pip install --disable-pip-version-check -r "${ROOT}/requirements.txt"
fi

if ! "${VENV_PYTHON}" -c 'import torch, torchvision; from packaging.version import Version; assert Version(torch.__version__.split("+")[0]) >= Version("2.2")' >/dev/null 2>&1; then
  if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    torch_index="${TORCH_INDEX_URL}"
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    torch_index="https://download.pytorch.org/whl/cu124"
  else
    torch_index="https://download.pytorch.org/whl/cpu"
  fi
  "${VENV_PYTHON}" -m pip install --disable-pip-version-check \
    --index-url "${torch_index}" 'torch==2.5.1' 'torchvision==0.20.1'
fi

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${VENV_PYTHON}" -m n08cascade.cli \
  --config "${ROOT}/config/frozen_protocol.json" \
  --mode "${RUN_MODE}" \
  --data-dir "${DATA_DIR}" \
  --results-dir "${RESULTS_DIR}"

final_color="$("${VENV_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["color"])' "${RESULTS_DIR}/final_status.json")"
printf '\n%s — completed. Machine-readable decision: %s\n' "${final_color}" "${RESULTS_DIR}/final_status.json"
trap - EXIT
