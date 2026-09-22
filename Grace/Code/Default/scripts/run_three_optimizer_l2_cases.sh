#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_DIR}"

# 三种全量实验：
# 1) 不用Adam + 不用L2  -> SGD + wd=0
# 2) 不用Adam + 使用L2  -> SGD + wd=SGD_WEIGHT_DECAY
# 3) 使用Adam + 使用L2  -> Adam + wd=ADAM_WEIGHT_DECAY
#
# 用法示例：
#   bash scripts/run_three_optimizer_l2_cases.sh
#   LR=0.005 SEED=0 BATCH_SIZE=60 bash scripts/run_three_optimizer_l2_cases.sh
#   PROJECTS="Lang Closure" SGD_WEIGHT_DECAY=1e-4 ADAM_WEIGHT_DECAY=1e-4 bash scripts/run_three_optimizer_l2_cases.sh

LR="${LR:-0.01}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-60}"
WARMUP_IDS="${WARMUP_IDS:-10}"
BASE_EPOCHS="${BASE_EPOCHS:-15}"
INC_EPOCHS="${INC_EPOCHS:-2}"

# SGD 相关
SGD_MOMENTUM="${SGD_MOMENTUM:-0.9}"
SGD_WEIGHT_DECAY="${SGD_WEIGHT_DECAY:-1e-4}"

# Adam 相关
ADAM_WEIGHT_DECAY="${ADAM_WEIGHT_DECAY:-1e-4}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"

PROJECTS="${PROJECTS:-Lang Chart Closure Math Mockito Time}"

SGD_SCRIPT="${SGD_SCRIPT:-continual_run_sgd_l2.py}"
ADAM_SCRIPT="${ADAM_SCRIPT:-continual_run_adam_l2.py}"

echo "==== Three optimizer/L2 cases ===="
echo "CODE_DIR=${CODE_DIR}"
echo "PROJECTS=${PROJECTS}"
echo "LR=${LR} SEED=${SEED} BATCH_SIZE=${BATCH_SIZE}"
echo "WARMUP_IDS=${WARMUP_IDS} BASE_EPOCHS=${BASE_EPOCHS} INC_EPOCHS=${INC_EPOCHS}"
echo "SGD_MOMENTUM=${SGD_MOMENTUM} SGD_WEIGHT_DECAY=${SGD_WEIGHT_DECAY}"
echo "ADAM_WEIGHT_DECAY=${ADAM_WEIGHT_DECAY} ADAM_BETA1=${ADAM_BETA1} ADAM_BETA2=${ADAM_BETA2}"
echo

if [[ ! -f "${SGD_SCRIPT}" ]]; then
  echo "Error: cannot find ${SGD_SCRIPT} in current directory: $(pwd)"
  exit 1
fi
if [[ ! -f "${ADAM_SCRIPT}" ]]; then
  echo "Error: cannot find ${ADAM_SCRIPT} in current directory: $(pwd)"
  exit 1
fi

run_case_1() {
  local project="$1"
  echo ">>> [Case1][${project}] SGD + no L2"
  python "${SGD_SCRIPT}" \
    "${project}" "${LR}" "${SEED}" "${BATCH_SIZE}" \
    "${WARMUP_IDS}" "${BASE_EPOCHS}" "${INC_EPOCHS}" \
    "0.0" "${SGD_MOMENTUM}"
}

run_case_2() {
  local project="$1"
  echo ">>> [Case2][${project}] SGD + L2"
  python "${SGD_SCRIPT}" \
    "${project}" "${LR}" "${SEED}" "${BATCH_SIZE}" \
    "${WARMUP_IDS}" "${BASE_EPOCHS}" "${INC_EPOCHS}" \
    "${SGD_WEIGHT_DECAY}" "${SGD_MOMENTUM}"
}

run_case_3() {
  local project="$1"
  echo ">>> [Case3][${project}] Adam + L2"
  python "${ADAM_SCRIPT}" \
    "${project}" "${LR}" "${SEED}" "${BATCH_SIZE}" \
    "${WARMUP_IDS}" "${BASE_EPOCHS}" "${INC_EPOCHS}" \
    "${ADAM_WEIGHT_DECAY}" "${ADAM_BETA1}" "${ADAM_BETA2}"
}

for project in ${PROJECTS}; do
  run_case_1 "${project}"
  run_case_2 "${project}"
  run_case_3 "${project}"
done

echo
echo "All three experiment cases completed."
