#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_DIR}"

# EWC-only 全量实验（无 replay）
# 通过将 replay 参数固定为 0 来禁用 replay 分支。
#
# 用法：
#   bash scripts/run_ewc_only_all_projects.sh
#   LR=0.01 SEED=0 BATCH_SIZE=60 bash scripts/run_ewc_only_all_projects.sh
#   PROJECTS="Lang Closure" EWC_LAMBDA=20.0 bash scripts/run_ewc_only_all_projects.sh

PROJECTS="${PROJECTS:-Lang Chart Closure Math Mockito Time}"

LR="${LR:-0.01}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-60}"
WARMUP_IDS="${WARMUP_IDS:-10}"
BASE_EPOCHS="${BASE_EPOCHS:-15}"
INC_EPOCHS="${INC_EPOCHS:-2}"

# Replay disabled (keep fixed)
REPLAY_SIZE=0
REPLAY_PER_STEP=0
REPLAY_BETA=0.0
DISTILL_ALPHA=0.0
TEMPERATURE="${TEMPERATURE:-2.0}"

# EWC config
EWC_LAMBDA="${EWC_LAMBDA:-10.0}"
EWC_GAMMA="${EWC_GAMMA:-0.9}"
FISHER_IDS="${FISHER_IDS:-4}"

SCRIPT="${SCRIPT:-continual_run_ewc.py}"

echo "==== EWC-only batch run (no replay) ===="
echo "CODE_DIR=${CODE_DIR}"
echo "SCRIPT=${SCRIPT}"
echo "PROJECTS=${PROJECTS}"
echo "LR=${LR} SEED=${SEED} BATCH_SIZE=${BATCH_SIZE}"
echo "WARMUP_IDS=${WARMUP_IDS} BASE_EPOCHS=${BASE_EPOCHS} INC_EPOCHS=${INC_EPOCHS}"
echo "REPLAY_SIZE=${REPLAY_SIZE} REPLAY_PER_STEP=${REPLAY_PER_STEP} REPLAY_BETA=${REPLAY_BETA} DISTILL_ALPHA=${DISTILL_ALPHA}"
echo "EWC_LAMBDA=${EWC_LAMBDA} EWC_GAMMA=${EWC_GAMMA} FISHER_IDS=${FISHER_IDS}"
echo

if [[ ! -f "${SCRIPT}" ]]; then
  echo "Error: cannot find ${SCRIPT} in current directory: $(pwd)"
  exit 1
fi

for project in ${PROJECTS}; do
  echo ">>> [${project}] EWC-only"
  python "${SCRIPT}" \
    "${project}" \
    "${LR}" \
    "${SEED}" \
    "${BATCH_SIZE}" \
    "${WARMUP_IDS}" \
    "${BASE_EPOCHS}" \
    "${INC_EPOCHS}" \
    "${REPLAY_SIZE}" \
    "${REPLAY_PER_STEP}" \
    "${REPLAY_BETA}" \
    "${DISTILL_ALPHA}" \
    "${TEMPERATURE}" \
    "${EWC_LAMBDA}" \
    "${EWC_GAMMA}" \
    "${FISHER_IDS}"
done

echo
echo "All EWC-only runs completed."
