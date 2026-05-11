#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_continual_dropout_all_cases.sh
#   LR=3e-5 SEED=7 BATCH_SIZE=1 bash run_continual_dropout_all_cases.sh
#   PROJECTS="Lang Closure" INC_EPOCHS=3 bash run_continual_dropout_all_cases.sh

# ------------------------
# Tunable hyper-parameters
# ------------------------
LR="${LR:-0.01}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-60}"
WARMUP_IDS="${WARMUP_IDS:-10}"
BASE_EPOCHS="${BASE_EPOCHS:-15}"
INC_EPOCHS="${INC_EPOCHS:-2}"
REPLAY_SIZE="${REPLAY_SIZE:-20}"
REPLAY_PER_STEP="${REPLAY_PER_STEP:-2}"
REPLAY_BETA="${REPLAY_BETA:-1.0}"
DISTILL_ALPHA="${DISTILL_ALPHA:-0.5}"
TEMPERATURE="${TEMPERATURE:-2.0}"
EWC_LAMBDA="${EWC_LAMBDA:-10.0}"
EWC_GAMMA="${EWC_GAMMA:-0.9}"
FISHER_IDS="${FISHER_IDS:-4}"
DROPOUT_ALPHA="${DROPOUT_ALPHA:-0.1}"
DROPOUT_TEMPERATURE="${DROPOUT_TEMPERATURE:-2.0}"

# Projects to run (space-separated)
PROJECTS="${PROJECTS:-Lang Chart Closure Math Mockito Time}"

# Path to runner script (relative to current folder)
SCRIPT="${SCRIPT:-continual_dropout.py}"

echo "==== continual_dropout batch run ===="
echo "SCRIPT=${SCRIPT}"
echo "PROJECTS=${PROJECTS}"
echo "LR=${LR} SEED=${SEED} BATCH_SIZE=${BATCH_SIZE}"
echo "WARMUP_IDS=${WARMUP_IDS} BASE_EPOCHS=${BASE_EPOCHS} INC_EPOCHS=${INC_EPOCHS}"
echo "REPLAY_SIZE=${REPLAY_SIZE} REPLAY_PER_STEP=${REPLAY_PER_STEP}"
echo "REPLAY_BETA=${REPLAY_BETA} DISTILL_ALPHA=${DISTILL_ALPHA} TEMPERATURE=${TEMPERATURE}"
echo "EWC_LAMBDA=${EWC_LAMBDA} EWC_GAMMA=${EWC_GAMMA} FISHER_IDS=${FISHER_IDS}"
echo "DROPOUT_ALPHA=${DROPOUT_ALPHA} DROPOUT_TEMPERATURE=${DROPOUT_TEMPERATURE}"
echo

if [[ ! -f "${SCRIPT}" ]]; then
  echo "Error: cannot find ${SCRIPT} in current directory: $(pwd)"
  exit 1
fi

run_one_case() {
  local project="$1"
  local use_ewc="$2"
  local use_gem="$3"

  echo ">>> [${project}] use_ewc=${use_ewc} use_gem=${use_gem}"
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
    "${FISHER_IDS}" \
    "${DROPOUT_ALPHA}" \
    "${DROPOUT_TEMPERATURE}" \
    "${use_ewc}" \
    "${use_gem}"
}

for project in ${PROJECTS}; do
  # 1) 全开（默认）
  run_one_case "${project}" 1 1
  # 2) 关 EWC，开 GEM
  run_one_case "${project}" 0 1
  # 3) 开 EWC，关 GEM
  run_one_case "${project}" 1 0
  # 4) 都关（Replay + DER++ + Dropout）
  run_one_case "${project}" 0 0
done

echo
echo "All runs completed."
