#!/usr/bin/env bash
set -euo pipefail

# 热力图补实验：在 run_basic_param_sweep_on_best.sh 同一套 Replay-only 与 5 点网格上，
# 每次固定一个协议参数为默认值，另外两个做 5×5 全组合。
# 不画图，只跑实验。不去重重复格子时每项目 75 次；默认去重后 61 次，
# 再跳过已有一维扫描格子后每项目 48 次新跑。
#
# 三张热力图：
#   H1 固定 warmup=10，网格 base_epochs × inc_epochs
#   H2 固定 base_epochs=15，网格 warmup × inc_epochs
#   H3 固定 inc_epochs=2，网格 warmup × base_epochs
#
# 用法：
#   bash scripts/run_basic_param_heatmap_grids_on_best.sh
#   PROJECTS="Lang" CUDA_VISIBLE_DEVICES=1 bash scripts/run_basic_param_heatmap_grids_on_best.sh
#   HEATMAPS="fix_warmup" SKIP_ONE_D=0 bash scripts/run_basic_param_heatmap_grids_on_best.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_DIR}"

PROJECTS="${PROJECTS:-Lang Chart Closure Math Mockito Time}"
PYTHON="${PYTHON:-python}"
SCRIPT="${SCRIPT:-continual_run_with_replay.py}"

SEED="${SEED:-0}"
LR="${LR:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-60}"
CENTER_WARM="${CENTER_WARM:-10}"
CENTER_BASE="${CENTER_BASE:-15}"
CENTER_INC="${CENTER_INC:-2}"

REPLAY_SIZE="${REPLAY_SIZE:-20}"
REPLAY_PER_STEP="${REPLAY_PER_STEP:-2}"
REPLAY_BETA="${REPLAY_BETA:-1.0}"
DISTILL_ALPHA="${DISTILL_ALPHA:-0.0}"
TEMPERATURE="${TEMPERATURE:-2.0}"

WARMUP_LIST="${WARMUP_LIST:-6 8 10 12 14}"
BASE_EPOCHS_LIST="${BASE_EPOCHS_LIST:-5 10 15 20 25}"
INC_EPOCHS_LIST="${INC_EPOCHS_LIST:-1 2 3 4 5}"

# 空格分隔：fix_warmup fix_base fix_inc
HEATMAPS="${HEATMAPS:-fix_warmup fix_base fix_inc}"
# 1：跳过一维扫描已覆盖的格子（某两维钉在默认、只动一维）
SKIP_ONE_D="${SKIP_ONE_D:-1}"

mkdir -p logs result

if [[ ! -f "${SCRIPT}" ]]; then
  echo "Error: cannot find ${SCRIPT} in ${CODE_DIR}"
  exit 1
fi

heatmap_enabled() {
  local name="$1"
  for h in ${HEATMAPS}; do
    if [[ "${h}" == "${name}" ]]; then
      return 0
    fi
  done
  return 1
}

combo_in() {
  local needle="$1"
  shift
  local x
  for x in "$@"; do
    if [[ "${x}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

is_one_d_point() {
  local warm="$1" base="$2" inc="$3"
  if [[ "${base}" == "${CENTER_BASE}" && "${inc}" == "${CENTER_INC}" ]]; then
    return 0
  fi
  if [[ "${warm}" == "${CENTER_WARM}" && "${inc}" == "${CENTER_INC}" ]]; then
    return 0
  fi
  if [[ "${warm}" == "${CENTER_WARM}" && "${base}" == "${CENTER_BASE}" ]]; then
    return 0
  fi
  return 1
}

COMBOS=()
add_combo() {
  local warm="$1" base="$2" inc="$3"
  local key="${warm}|${base}|${inc}"
  if combo_in "${key}" "${COMBOS[@]+"${COMBOS[@]}"}"; then
    return 0
  fi
  COMBOS+=("${key}")
}

if heatmap_enabled fix_warmup; then
  for base in ${BASE_EPOCHS_LIST}; do
    for inc in ${INC_EPOCHS_LIST}; do
      add_combo "${CENTER_WARM}" "${base}" "${inc}"
    done
  done
fi
if heatmap_enabled fix_base; then
  for warm in ${WARMUP_LIST}; do
    for inc in ${INC_EPOCHS_LIST}; do
      add_combo "${warm}" "${CENTER_BASE}" "${inc}"
    done
  done
fi
if heatmap_enabled fix_inc; then
  for warm in ${WARMUP_LIST}; do
    for base in ${BASE_EPOCHS_LIST}; do
      add_combo "${warm}" "${base}" "${CENTER_INC}"
    done
  done
fi

echo "==== Heatmap grids on best strategy (replay-only) ===="
echo "CODE_DIR=${CODE_DIR}"
echo "PROJECTS=${PROJECTS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "strategy: ${SCRIPT}  rs=${REPLAY_SIZE} rps=${REPLAY_PER_STEP} rb=${REPLAY_BETA} da=${DISTILL_ALPHA} T=${TEMPERATURE}"
echo "fixed (source): lr=${LR} bs=${BATCH_SIZE} seed=${SEED}"
echo "defaults: warm=${CENTER_WARM} base=${CENTER_BASE} inc=${CENTER_INC}"
echo "warmup grid:      ${WARMUP_LIST}"
echo "base_epochs grid: ${BASE_EPOCHS_LIST}"
echo "inc_epochs grid:  ${INC_EPOCHS_LIST}"
echo "HEATMAPS=${HEATMAPS}  SKIP_ONE_D=${SKIP_ONE_D}"
echo "unique combos (after heatmap union): ${#COMBOS[@]}"
echo
echo "---- unique (warmup, base_epochs, inc_epochs) ----"
n_skip=0
n_run=0
PLAN_RUN=()
PLAN_SKIP=()
for key in "${COMBOS[@]}"; do
  IFS='|' read -r warm base inc <<<"${key}"
  if [[ "${SKIP_ONE_D}" == "1" ]] && is_one_d_point "${warm}" "${base}" "${inc}"; then
    PLAN_SKIP+=("${key}")
    n_skip=$((n_skip + 1))
    echo "  skip-1d  warm=${warm}  base=${base}  inc=${inc}"
  else
    PLAN_RUN+=("${key}")
    n_run=$((n_run + 1))
    echo "  run      warm=${warm}  base=${base}  inc=${inc}"
  fi
done
echo "---- plan: run=${n_run}  skip_1d=${n_skip}  per project ----"
echo "---- total new jobs if all projects: $((n_run * $(set -- ${PROJECTS}; echo $#))) ----"
echo

run_one() {
  local project="$1" warm="$2" base="$3" inc="$4"
  local tag="${project}_heatmap_warm${warm}_base${base}_inc${inc}"
  local log="logs/${tag}.log"
  echo ">>> ${tag}"
  echo "    log=${log}"
  "${PYTHON}" "${SCRIPT}" \
    "${project}" "${LR}" "${SEED}" "${BATCH_SIZE}" \
    "${warm}" "${base}" "${inc}" \
    "${REPLAY_SIZE}" "${REPLAY_PER_STEP}" "${REPLAY_BETA}" "${DISTILL_ALPHA}" "${TEMPERATURE}" \
    | tee "${log}"
  echo ">>> ${tag} done"
  echo
}

for project in ${PROJECTS}; do
  echo "===== ${project} ====="
  for key in "${PLAN_RUN[@]+"${PLAN_RUN[@]}"}"; do
    IFS='|' read -r warm base inc <<<"${key}"
    run_one "${project}" "${warm}" "${base}" "${inc}"
  done
done

echo "==== Heatmap grid runs finished ===="
echo "Logs: logs/<Project>_heatmap_warm*_base*_inc*.log"
echo "Outputs: result/<Project>/replay_base*_inc*_lr${LR}_bs${BATCH_SIZE}_warm*_rs${REPLAY_SIZE}_rps${REPLAY_PER_STEP}_rb${REPLAY_BETA}_da${DISTILL_ALPHA}_ts*/"
