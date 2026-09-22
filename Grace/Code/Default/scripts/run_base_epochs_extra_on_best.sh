#!/usr/bin/env bash
set -euo pipefail

# 补扫 base_epochs={30,35,40,45,50}。策略与主扫描相同：Replay-only（da=0）。
# 其余钉死：warm=10, inc=2, lr=0.01, bs=60, seed=0。
#
#   bash scripts/run_base_epochs_extra_on_best.sh
#   PROJECTS="Lang" CUDA_VISIBLE_DEVICES=1 bash scripts/run_base_epochs_extra_on_best.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_DIR}"

PROJECTS="${PROJECTS:-Lang Chart Closure Math Mockito Time}"
PYTHON="${PYTHON:-python}"
SCRIPT="${SCRIPT:-continual_run_with_replay.py}"

SEED="${SEED:-0}"
LR="${LR:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-60}"
WARMUP_IDS="${WARMUP_IDS:-10}"
INC_EPOCHS="${INC_EPOCHS:-2}"
BASE_EPOCHS_LIST="${BASE_EPOCHS_LIST:-30 35 40 45 50}"

REPLAY_SIZE="${REPLAY_SIZE:-20}"
REPLAY_PER_STEP="${REPLAY_PER_STEP:-2}"
REPLAY_BETA="${REPLAY_BETA:-1.0}"
DISTILL_ALPHA="${DISTILL_ALPHA:-0.0}"
TEMPERATURE="${TEMPERATURE:-2.0}"

mkdir -p logs result

if [[ ! -f "${SCRIPT}" ]]; then
  echo "Error: cannot find ${SCRIPT} in ${CODE_DIR}"
  exit 1
fi

echo "==== Extra base_epochs on best strategy (replay-only) ===="
echo "CODE_DIR=${CODE_DIR}"
echo "PROJECTS=${PROJECTS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "fixed: lr=${LR} bs=${BATCH_SIZE} seed=${SEED} warm=${WARMUP_IDS} inc=${INC_EPOCHS}"
echo "base_epochs: ${BASE_EPOCHS_LIST}"
echo

for project in ${PROJECTS}; do
  echo "===== ${project} ====="
  for base in ${BASE_EPOCHS_LIST}; do
    tag="${project}_best_replay_sweep_base_warm${WARMUP_IDS}_base${base}_inc${INC_EPOCHS}"
    log="logs/${tag}.log"
    echo ">>> ${tag}"
    echo "    log=${log}"
    "${PYTHON}" "${SCRIPT}" \
      "${project}" "${LR}" "${SEED}" "${BATCH_SIZE}" \
      "${WARMUP_IDS}" "${base}" "${INC_EPOCHS}" \
      "${REPLAY_SIZE}" "${REPLAY_PER_STEP}" "${REPLAY_BETA}" "${DISTILL_ALPHA}" "${TEMPERATURE}" \
      | tee "${log}"
    echo ">>> ${tag} done"
    echo
  done
done

echo "==== Extra base_epochs finished ===="
echo "Logs: logs/<Project>_best_replay_sweep_base_warm${WARMUP_IDS}_base{30,35,40,45,50}_inc${INC_EPOCHS}.log"
