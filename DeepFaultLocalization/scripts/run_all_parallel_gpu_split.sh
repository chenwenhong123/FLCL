#!/usr/bin/env bash
# 并行总调度：
# - 默认前四个脚本走 GPU0
# - 其余脚本走 GPU1
#
# 用法：
#   bash scripts/run_all_parallel_gpu_split.sh
# 可选覆盖：
#   GPU0=0 GPU1=1 bash scripts/run_all_parallel_gpu_split.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/logs_parallel_${TS}"
mkdir -p "${LOG_DIR}"

SCRIPTS_GPU0=(
  "scripts/run_01_stream_ewc_only.sh"
  "scripts/run_02_replay_der_only.sh"
  "scripts/run_03_replay_der_ewc.sh"
  "scripts/run_04_replay_gem.sh"
)

SCRIPTS_GPU1=(
  "scripts/run_09_replay_der_local_grid.sh"
  "scripts/run_main_opt_l2_ablation.sh"
)

pids=()
names=()

run_one() {
  local gpu="$1"
  local script="$2"
  local base
  base="$(basename "${script}" .sh)"
  local log="${LOG_DIR}/${base}.log"

  if [[ ! -f "${script}" ]]; then
    echo "[ERROR] missing script: ${script}" | tee -a "${LOG_DIR}/missing.log"
    return 1
  fi

  echo "[START] gpu=${gpu} script=${script} log=${log}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export GPU_DEVICE_ORDINAL="${gpu}"
    bash "${script}"
  ) >"${log}" 2>&1 &

  pids+=("$!")
  names+=("${script}")
}

for s in "${SCRIPTS_GPU0[@]}"; do
  run_one "${GPU0}" "${s}"
done

for s in "${SCRIPTS_GPU1[@]}"; do
  run_one "${GPU1}" "${s}"
done

echo "============================================================"
echo "All jobs started."
echo "Logs: ${LOG_DIR}"
echo "============================================================"

exit_code=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  name="${names[$i]}"
  if wait "${pid}"; then
    echo "[DONE] ${name}"
  else
    echo "[FAIL] ${name}"
    exit_code=1
  fi
done

echo "============================================================"
if [[ "${exit_code}" -eq 0 ]]; then
  echo "All scripts finished successfully."
else
  echo "Some scripts failed. Check logs under: ${LOG_DIR}"
fi
echo "============================================================"

exit "${exit_code}"

