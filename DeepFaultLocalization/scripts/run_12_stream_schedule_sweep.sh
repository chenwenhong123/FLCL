#!/usr/bin/env bash
# 基础增量日程参数扫描（单因素 5 档，中心点只跑一次）
#
# 固定组合（来自 result_ablation/09162208 带策略实验，incr_pre_top1 六项目合计最高）：
#   入口     continual_stream_train.py
#   策略     EWC（--use_ewc true；不扫 replay，因其不是基础增量参数）
#   模型     mlp_dfl_1
#   优化     adam + L2
#
# 扫描参数（钉住其余参数时，base_epochs 与 run_10/11 默认一致为 50）：
#   warmup          默认 10    {6, 8, 10, 12, 14}     → --warmup
#   base_epochs     默认 50    {5, 10, 15, 20, 25}    → --training_epochs（warmup 合训；高档 30–45 见 run_13，50 即中心点）
#   inc_epochs      默认 2     {1, 2, 3, 4, 5}        → --incr_epochs
#
# AXIS=all 时 14 组配置 × 6 项目 = 84 次任务（中心点不重复）：
#   center          w=10 e=50 i=2
#   warmup          w∈{6,8,12,14}     e=50 i=2
#   base_epochs     e∈{5,10,15,20,25}  w=10 i=2
#   inc_epochs      i∈{1,3,4,5}       w=10 e=50
#
# 服务器：
#   cd /path/to/DeepFaultLocalization
#   export PYTHON=python3 DATA_ROOT=. CUDA_VISIBLE_DEVICES=0
#   nohup bash scripts/run_12_stream_schedule_sweep.sh > logs/run_12_$(date +%m%d%H%M).log 2>&1 &
#
# 只跑某一轴（含该轴上的中心取值，各 5 档 × 6 项目）：
#   AXIS=warmup bash scripts/run_12_stream_schedule_sweep.sh
#   AXIS=base_epochs bash scripts/run_12_stream_schedule_sweep.sh
#   AXIS=inc_epochs bash scripts/run_12_stream_schedule_sweep.sh
#   AXIS=center bash scripts/run_12_stream_schedule_sweep.sh
#
# 覆盖模型 / 关掉 EWC（改为基础无策略增量）：
#   MODELS="mlp_dfl_1 birnn" bash scripts/run_12_stream_schedule_sweep.sh
#   USE_EWC=false bash scripts/run_12_stream_schedule_sweep.sh

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-.}"
TECH="${TECH:-DeepFL_CL}"
LOSS="${LOSS:-softmax}"
DUMP_STEP="${DUMP_STEP:-10}"
GPU_MEM="${GPU_MEM:-0.2}"
USE_EWC="${USE_EWC:-true}"
OPTIMIZER="${OPTIMIZER:-adam}"
USE_L2="${USE_L2:-true}"
EWC_LAMBDA="${EWC_LAMBDA:-10.0}"
EWC_GAMMA="${EWC_GAMMA:-0.9}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
AXIS="${AXIS:-all}"

CENTER_WARMUP="${CENTER_WARMUP:-10}"
CENTER_EPOCHS="${CENTER_EPOCHS:-50}"
CENTER_INCR="${CENTER_INCR:-2}"

WARMUP_GRID=(6 8 10 12 14)
EPOCH_GRID=(5 10 15 20 25)
INCR_GRID=(1 2 3 4 5)

SUBJECTS=(Chart Lang Math Time Closure Mockito)
if [[ -n "${MODELS:-}" ]]; then
  read -r -a MODELS <<< "${MODELS}"
else
  MODELS=(mlp_dfl_1)
fi

STAMP="$(date +%m%d%H%M)"
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/${STAMP}_stream_schedule_sweep.log"
FAIL_LOG="${LOG_DIR}/${STAMP}_stream_schedule_sweep_fail.txt"
: > "${FAIL_LOG}"
OUT_ROOT="${ROOT}/result_stream_schedule/${STAMP}"
mkdir -p "${OUT_ROOT}"

n_fail=0
n_ok=0
n_skip=0

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${MASTER_LOG}"
}

has_metrics() {
  local run_dir="$1"
  local f
  shopt -s nullglob
  for f in "${run_dir}"/*_continual_metrics.txt; do
    shopt -u nullglob
    return 0
  done
  shopt -u nullglob
  return 1
}

run_one() {
  local tag="$1" warmup="$2" epochs="$3" incr="$4" sub="$5" model="$6"
  local run_dir="${OUT_ROOT}/${tag}/${model}/${sub}"
  mkdir -p "${run_dir}"
  if [[ "${SKIP_EXISTING}" == "1" ]] && has_metrics "${run_dir}"; then
    n_skip=$((n_skip + 1))
    log "SKIP  exists ${tag} sub=${sub} model=${model}"
    return 0
  fi
  log "START ${tag} sub=${sub} model=${model} warmup=${warmup} base_epochs=${epochs} inc_epochs=${incr}"
  if "${PYTHON}" -u continual_stream_train.py "${sub}" "${model}" \
      --data_root "${DATA_ROOT}" \
      --out_dir "${run_dir}" \
      --tech "${TECH}" \
      --loss "${LOSS}" \
      --warmup "${warmup}" \
      --training_epochs "${epochs}" \
      --incr_epochs "${incr}" \
      --dump_step "${DUMP_STEP}" \
      --gpu_mem "${GPU_MEM}" \
      --use_ewc "${USE_EWC}" \
      --ewc_lambda "${EWC_LAMBDA}" \
      --ewc_gamma "${EWC_GAMMA}" \
      --optimizer "${OPTIMIZER}" \
      --use_l2 "${USE_L2}" \
      >>"${MASTER_LOG}" 2>&1; then
    n_ok=$((n_ok + 1))
    log "OK    ${tag} sub=${sub} model=${model}"
    return 0
  fi
  n_fail=$((n_fail + 1))
  log "FAIL  ${tag} sub=${sub} model=${model}"
  echo "${tag} sub=${sub} model=${model} warmup=${warmup} base_epochs=${epochs} inc_epochs=${incr}" >> "${FAIL_LOG}"
  return 0
}

run_combo() {
  local tag="$1" warmup="$2" epochs="$3" incr="$4"
  local sub model
  for sub in "${SUBJECTS[@]}"; do
    for model in "${MODELS[@]}"; do
      run_one "${tag}" "${warmup}" "${epochs}" "${incr}" "${sub}" "${model}"
    done
  done
}

run_center() {
  log "======== center: warmup=${CENTER_WARMUP} base_epochs=${CENTER_EPOCHS} inc_epochs=${CENTER_INCR} ========"
  run_combo "center_w${CENTER_WARMUP}_e${CENTER_EPOCHS}_i${CENTER_INCR}" \
    "${CENTER_WARMUP}" "${CENTER_EPOCHS}" "${CENTER_INCR}"
}

run_warmup_axis() {
  local w include_center="${1:-0}"
  log "======== axis warmup  grid={${WARMUP_GRID[*]}}  base_epochs=${CENTER_EPOCHS} inc_epochs=${CENTER_INCR} ========"
  for w in "${WARMUP_GRID[@]}"; do
    if [[ "${include_center}" != "1" && "${w}" -eq "${CENTER_WARMUP}" ]]; then
      log "skip warmup=${w} (covered by center)"
      continue
    fi
    run_combo "warmup_w${w}" "${w}" "${CENTER_EPOCHS}" "${CENTER_INCR}"
  done
}

run_epochs_axis() {
  local e include_center="${1:-0}"
  log "======== axis base_epochs  grid={${EPOCH_GRID[*]}}  warmup=${CENTER_WARMUP} inc_epochs=${CENTER_INCR} ========"
  for e in "${EPOCH_GRID[@]}"; do
    if [[ "${include_center}" != "1" && "${e}" -eq "${CENTER_EPOCHS}" ]]; then
      log "skip base_epochs=${e} (covered by center)"
      continue
    fi
    run_combo "base_epochs_e${e}" "${CENTER_WARMUP}" "${e}" "${CENTER_INCR}"
  done
}

run_incr_axis() {
  local i include_center="${1:-0}"
  log "======== axis inc_epochs  grid={${INCR_GRID[*]}}  warmup=${CENTER_WARMUP} base_epochs=${CENTER_EPOCHS} ========"
  for i in "${INCR_GRID[@]}"; do
    if [[ "${include_center}" != "1" && "${i}" -eq "${CENTER_INCR}" ]]; then
      log "skip inc_epochs=${i} (covered by center)"
      continue
    fi
    run_combo "inc_epochs_i${i}" "${CENTER_WARMUP}" "${CENTER_EPOCHS}" "${i}"
  done
}

log "ROOT=${ROOT}"
log "PYTHON=${PYTHON} ($(command -v "${PYTHON}" || true))"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
log "DATA_ROOT=${DATA_ROOT} TECH=${TECH} AXIS=${AXIS}"
log "MODELS=${MODELS[*]} SUBJECTS=${SUBJECTS[*]}"
log "fixed: use_ewc=${USE_EWC} optimizer=${OPTIMIZER} use_l2=${USE_L2} ewc_lambda=${EWC_LAMBDA} ewc_gamma=${EWC_GAMMA}"
log "center: warmup=${CENTER_WARMUP} base_epochs=${CENTER_EPOCHS} inc_epochs=${CENTER_INCR}"
log "OUT_ROOT=${OUT_ROOT}"

case "${AXIS}" in
  all)
    run_center
    run_warmup_axis 0
    run_epochs_axis 0
    run_incr_axis 0
    ;;
  center)
    run_center
    ;;
  warmup)
    run_warmup_axis 1
    ;;
  base_epochs)
    run_epochs_axis 1
    ;;
  inc_epochs)
    run_incr_axis 1
    ;;
  *)
    echo "AXIS must be all|center|warmup|base_epochs|inc_epochs" >&2
    exit 1
    ;;
esac

log "======== DONE ok=${n_ok} skip=${n_skip} fail=${n_fail} ========"
log "master log: ${MASTER_LOG}"
log "fail list:  ${FAIL_LOG}"
log "out:        ${OUT_ROOT}"

if [[ "${n_fail}" -gt 0 ]]; then
  exit 1
fi
exit 0
