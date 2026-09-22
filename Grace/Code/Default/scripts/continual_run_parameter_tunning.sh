#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_DIR}"

# 用法（在 Code/Default 下）：
#   bash scripts/continual_run_parameter_tunning.sh

PROJECTS=(Lang Math Time Chart Mockito Closure)

LR=0.005
SEED=0
BS=60
WARM=10
BASE=15
INC=2

REPLAY_SIZE=20
REPLAY_PER_STEP=2
REPLAY_BETA=1.0
DISTILL_ALPHA=0.5
TEMP=2.0

EWC_LAMBDA=10.0
EWC_GAMMA=0.9
FISHER_IDS=4

for P in "${PROJECTS[@]}"; do
  echo "===== ${P} : continual ====="
  python continual_run.py "$P" "$LR" "$SEED" "$BS" \
    "$WARM" "$BASE" "$INC"

  echo "===== ${P} : replay ====="
  python continual_run_with_replay.py "$P" "$LR" "$SEED" "$BS" \
    "$WARM" "$BASE" "$INC" \
    "$REPLAY_SIZE" "$REPLAY_PER_STEP" "$REPLAY_BETA" "$DISTILL_ALPHA" "$TEMP"

  echo "===== ${P} : replay + ewc ====="
  python continual_run_ewc.py "$P" "$LR" "$SEED" "$BS" \
    "$WARM" "$BASE" "$INC" \
    "$REPLAY_SIZE" "$REPLAY_PER_STEP" "$REPLAY_BETA" "$DISTILL_ALPHA" "$TEMP" \
    "$EWC_LAMBDA" "$EWC_GAMMA" "$FISHER_IDS"

  echo "===== ${P} : replay + gem ====="
  python continual_run_gem.py "$P" "$LR" "$SEED" "$BS" \
    "$WARM" "$BASE" "$INC" \
    "$REPLAY_SIZE" "$REPLAY_PER_STEP" "$REPLAY_BETA" "$DISTILL_ALPHA" "$TEMP"

  echo "===== ${P} : replay + ewc + gem ====="
  python continual_run_all.py "$P" "$LR" "$SEED" "$BS" \
    "$WARM" "$BASE" "$INC" \
    "$REPLAY_SIZE" "$REPLAY_PER_STEP" "$REPLAY_BETA" "$DISTILL_ALPHA" "$TEMP" \
    "$EWC_LAMBDA" "$EWC_GAMMA" "$FISHER_IDS"

  echo "===== ${P} : mask (ewc=0 gem=0) ====="
  python continual_run_mask.py "$P" "$LR" "$SEED" "$BS" \
    "$WARM" "$BASE" "$INC" \
    "$REPLAY_SIZE" "$REPLAY_PER_STEP" "$REPLAY_BETA" "$DISTILL_ALPHA" "$TEMP" \
    "$EWC_LAMBDA" "$EWC_GAMMA" "$FISHER_IDS" \
    0 0

  echo "===== ${P} : mask (ewc=1 gem=0) ====="
  python continual_run_mask.py "$P" "$LR" "$SEED" "$BS" \
    "$WARM" "$BASE" "$INC" \
    "$REPLAY_SIZE" "$REPLAY_PER_STEP" "$REPLAY_BETA" "$DISTILL_ALPHA" "$TEMP" \
    "$EWC_LAMBDA" "$EWC_GAMMA" "$FISHER_IDS" \
    1 0

  echo "===== ${P} : mask (ewc=1 gem=1) ====="
  python continual_run_mask.py "$P" "$LR" "$SEED" "$BS" \
    "$WARM" "$BASE" "$INC" \
    "$REPLAY_SIZE" "$REPLAY_PER_STEP" "$REPLAY_BETA" "$DISTILL_ALPHA" "$TEMP" \
    "$EWC_LAMBDA" "$EWC_GAMMA" "$FISHER_IDS" \
    1 1
done

echo "All experiments finished."
