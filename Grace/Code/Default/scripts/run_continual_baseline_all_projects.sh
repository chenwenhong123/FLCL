#!/usr/bin/env bash
set -euo pipefail

# 基础增量学习：continual_run.py（无 replay / EWC / GEM）
# Warmup 前 K 个 ID 选基模型，再按时间流对后续缺陷增量训练。
#
# 用法：
#   bash Code/Default/scripts/run_continual_baseline_all_projects.sh
#   bash scripts/run_continual_baseline_all_projects.sh
#
# 常用覆盖：
#   PROJECTS="Lang" bash scripts/run_continual_baseline_all_projects.sh
#   PROJECTS="Lang Chart Closure Math Mockito Time" \
#     LR=0.01 SEED=0 BATCH_SIZE=60 WARMUP_IDS=10 BASE_EPOCHS=15 INC_EPOCHS=2 \
#     bash scripts/run_continual_baseline_all_projects.sh
#
# 环境变量：
#   PROJECTS / LR / SEED / BATCH_SIZE / WARMUP_IDS / BASE_EPOCHS / INC_EPOCHS / PYTHON

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_DIR}"

PROJECTS="${PROJECTS:-Lang Chart Closure Math Mockito Time}"
LR="${LR:-0.01}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-60}"
WARMUP_IDS="${WARMUP_IDS:-10}"
BASE_EPOCHS="${BASE_EPOCHS:-15}"
INC_EPOCHS="${INC_EPOCHS:-2}"
PYTHON="${PYTHON:-python}"
SCRIPT="${SCRIPT:-continual_run.py}"

mkdir -p logs result

echo "==== Continual baseline (continual_run.py) ===="
echo "CODE_DIR=${CODE_DIR}"
echo "SCRIPT=${SCRIPT}"
echo "PROJECTS=${PROJECTS}"
echo "LR=${LR} SEED=${SEED} BATCH_SIZE=${BATCH_SIZE}"
echo "WARMUP_IDS=${WARMUP_IDS} BASE_EPOCHS=${BASE_EPOCHS} INC_EPOCHS=${INC_EPOCHS}"
echo

if [[ ! -f "${SCRIPT}" ]]; then
  echo "Error: cannot find ${SCRIPT} in ${CODE_DIR}"
  exit 1
fi

pkl_path() {
  local project="$1"
  if [[ -f "${project}.pkl" ]]; then
    echo "${project}.pkl"
  else
    echo "GraceDate/${project}.pkl"
  fi
}

for project in ${PROJECTS}; do
  path="$(pkl_path "${project}")"
  if [[ ! -f "${path}" ]]; then
    echo "Error: data file not found: ${path}"
    exit 1
  fi
done

for project in ${PROJECTS}; do
  tag="${project}_continual_base${BASE_EPOCHS}_inc${INC_EPOCHS}_lr${LR}_bs${BATCH_SIZE}_warm${WARMUP_IDS}"
  log="logs/${tag}.log"
  echo ">>> [continual][${project}] log=${log}"
  "${PYTHON}" "${SCRIPT}" \
    "${project}" \
    "${LR}" \
    "${SEED}" \
    "${BATCH_SIZE}" \
    "${WARMUP_IDS}" \
    "${BASE_EPOCHS}" \
    "${INC_EPOCHS}" \
    | tee "${log}"
  echo ">>> [continual][${project}] done"
  echo
done

echo "All continual baseline runs completed."
echo "Outputs: result/<Project>/continual_base${BASE_EPOCHS}_inc${INC_EPOCHS}_lr${LR}_bs${BATCH_SIZE}_warm${WARMUP_IDS}_ts*/"
echo "  metrics.csv  model.pt  summary.txt"
echo "Logs: logs/<Project>_continual_*.log"
