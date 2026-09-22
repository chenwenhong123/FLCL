#!/usr/bin/env bash
# 二维日程参数网格（热力图用）。策略与 run_11 最优组合一致：
#   continual_stream_train.py，EWC + mlp_dfl_1，adam+L2
#   热力图固定默认：warmup=10，base_epochs=50（与 run_10/11 的 training_epochs 默认相同），inc_epochs=2
#
# 一维网格（已跑，见 result_stream_schedule/；注意 run_12 的 warmup/inc 轴是在 base_epochs=15 上扫的）：
#   warmup       {6, 8, 10, 12, 14}                 默认 10
#   base_epochs  {5,10,15,20,25} ∪ {30,35,40,45,50} 一维中心曾为 15，热力图固定改为 50
#   inc_epochs   {1, 2, 3, 4, 5}                    默认 2
#
# 三张热力图（固定一维默认，另两维全组合）：
#
#   HEATMAP=fix_warmup        固定 warmup=10
#     行 base_epochs × 列 inc_epochs = 10×5 = 50
#     一维已有：(w=10, 全部 e, i=2)；(w=10, e=15, 全部 i)
#     需新跑：其余 36 格（含 e=50 且 i≠2 的一行）
#
#   HEATMAP=fix_base_epochs   固定 base_epochs=50
#     行 warmup × 列 inc_epochs = 5×5 = 25
#     一维已有：仅 (w=10, e=50, i=2)
#     需新跑：其余 24 格
#
#   HEATMAP=fix_inc_epochs    固定 inc_epochs=2
#     行 warmup × 列 base_epochs = 5×10 = 50
#     一维已有：(全部 w, e=15, i=2)；(w=10, 全部 e, i=2)
#     需新跑：其余 36 格（含 w≠10 且 e=50 的一列）
#
# 默认 SKIP_1D=1：跳过上面「一维已有」格子（96 组 × 6 项目 = 576 次）。
# SKIP_1D=0：每张图跑满格（50+25+50=125 组 × 6 = 750 次），目录自洽。
#
# 服务器：
#   cd /path/to/DeepFaultLocalization
#   export PYTHON=python3 DATA_ROOT=. CUDA_VISIBLE_DEVICES=0
#   nohup bash scripts/run_14_stream_heatmap_grids.sh > logs/run_14_$(date +%m%d%H%M).log 2>&1 &
#
# 只跑一张图：
#   HEATMAP=fix_warmup bash scripts/run_14_stream_heatmap_grids.sh
#   HEATMAP=fix_base_epochs bash scripts/run_14_stream_heatmap_grids.sh
#   HEATMAP=fix_inc_epochs bash scripts/run_14_stream_heatmap_grids.sh
#
# 补跑一维十字线到本目录：
#   SKIP_1D=0 HEATMAP=fix_warmup bash scripts/run_14_stream_heatmap_grids.sh

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
SKIP_1D="${SKIP_1D:-1}"
HEATMAP="${HEATMAP:-all}"

CENTER_WARMUP="${CENTER_WARMUP:-10}"
CENTER_EPOCHS="${CENTER_EPOCHS:-50}"
CENTER_INCR="${CENTER_INCR:-2}"
# run_12 一维的 warmup / inc_epochs 轴是在 base_epochs=15 上扫的，SKIP_1D 按实际已跑格子判断
SCHEDULE_1D_EPOCHS="${SCHEDULE_1D_EPOCHS:-15}"

WARMUP_GRID=(6 8 10 12 14)
EPOCH_GRID=(5 10 15 20 25 30 35 40 45 50)
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
MASTER_LOG="${LOG_DIR}/${STAMP}_stream_heatmap_grids.log"
FAIL_LOG="${LOG_DIR}/${STAMP}_stream_heatmap_grids_fail.txt"
: > "${FAIL_LOG}"
OUT_ROOT="${ROOT}/result_stream_heatmap/${STAMP}"
mkdir -p "${OUT_ROOT}"

n_fail=0
n_ok=0
n_skip=0

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${MASTER_LOG}"
}

is_1d_cell() {
  local _heatmap="$1" w="$2" e="$3" i="$4"
  # (全部 w, e=15, i=2)：run_12 warmup 轴
  if [[ "${e}" -eq "${SCHEDULE_1D_EPOCHS}" && "${i}" -eq "${CENTER_INCR}" ]]; then
    return 0
  fi
  # (w=10, 全部 e, i=2)：run_12/13 base_epochs 轴（含 e=50）
  if [[ "${w}" -eq "${CENTER_WARMUP}" && "${i}" -eq "${CENTER_INCR}" ]]; then
    return 0
  fi
  # (w=10, e=15, 全部 i)：run_12 inc_epochs 轴
  if [[ "${w}" -eq "${CENTER_WARMUP}" && "${e}" -eq "${SCHEDULE_1D_EPOCHS}" ]]; then
    return 0
  fi
  return 1
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
  local heatmap="$1" tag="$2" warmup="$3" epochs="$4" incr="$5" sub="$6" model="$7"
  local run_dir="${OUT_ROOT}/${heatmap}/${tag}/${model}/${sub}"
  mkdir -p "${run_dir}"
  if [[ "${SKIP_EXISTING}" == "1" ]] && has_metrics "${run_dir}"; then
    n_skip=$((n_skip + 1))
    log "SKIP  exists ${heatmap}/${tag} sub=${sub} model=${model}"
    return 0
  fi
  log "START ${heatmap}/${tag} sub=${sub} model=${model} warmup=${warmup} base_epochs=${epochs} inc_epochs=${incr}"
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
    log "OK    ${heatmap}/${tag} sub=${sub} model=${model}"
    return 0
  fi
  n_fail=$((n_fail + 1))
  log "FAIL  ${heatmap}/${tag} sub=${sub} model=${model}"
  echo "${heatmap}/${tag} sub=${sub} model=${model} w=${warmup} e=${epochs} i=${incr}" >> "${FAIL_LOG}"
  return 0
}

run_combo() {
  local heatmap="$1" tag="$2" warmup="$3" epochs="$4" incr="$5"
  if [[ "${SKIP_1D}" == "1" ]] && is_1d_cell "${heatmap}" "${warmup}" "${epochs}" "${incr}"; then
    n_skip=$((n_skip + 1))
    log "SKIP  1D ${heatmap}/${tag} warmup=${warmup} base_epochs=${epochs} inc_epochs=${incr}"
    return 0
  fi
  local sub model
  for sub in "${SUBJECTS[@]}"; do
    for model in "${MODELS[@]}"; do
      run_one "${heatmap}" "${tag}" "${warmup}" "${epochs}" "${incr}" "${sub}" "${model}"
    done
  done
}

run_fix_warmup() {
  local e i
  log "======== HEATMAP fix_warmup  warmup=${CENTER_WARMUP}  e×i = {${EPOCH_GRID[*]}} × {${INCR_GRID[*]}} ========"
  for e in "${EPOCH_GRID[@]}"; do
    for i in "${INCR_GRID[@]}"; do
      run_combo "fix_warmup_w${CENTER_WARMUP}" "e${e}_i${i}" "${CENTER_WARMUP}" "${e}" "${i}"
    done
  done
}

run_fix_base_epochs() {
  local w i
  log "======== HEATMAP fix_base_epochs  base_epochs=${CENTER_EPOCHS}  w×i = {${WARMUP_GRID[*]}} × {${INCR_GRID[*]}} ========"
  for w in "${WARMUP_GRID[@]}"; do
    for i in "${INCR_GRID[@]}"; do
      run_combo "fix_base_epochs_e${CENTER_EPOCHS}" "w${w}_i${i}" "${w}" "${CENTER_EPOCHS}" "${i}"
    done
  done
}

run_fix_inc_epochs() {
  local w e
  log "======== HEATMAP fix_inc_epochs  inc_epochs=${CENTER_INCR}  w×e = {${WARMUP_GRID[*]}} × {${EPOCH_GRID[*]}} ========"
  for w in "${WARMUP_GRID[@]}"; do
    for e in "${EPOCH_GRID[@]}"; do
      run_combo "fix_inc_epochs_i${CENTER_INCR}" "w${w}_e${e}" "${w}" "${e}" "${CENTER_INCR}"
    done
  done
}

log "ROOT=${ROOT}"
log "PYTHON=${PYTHON} ($(command -v "${PYTHON}" || true))"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
log "HEATMAP=${HEATMAP} SKIP_1D=${SKIP_1D} SKIP_EXISTING=${SKIP_EXISTING}"
log "MODELS=${MODELS[*]} SUBJECTS=${SUBJECTS[*]}"
log "fixed: use_ewc=${USE_EWC} optimizer=${OPTIMIZER} use_l2=${USE_L2}"
log "center: warmup=${CENTER_WARMUP} base_epochs=${CENTER_EPOCHS} inc_epochs=${CENTER_INCR} (1D skip epochs=${SCHEDULE_1D_EPOCHS})"
log "OUT_ROOT=${OUT_ROOT}"

case "${HEATMAP}" in
  all)
    run_fix_warmup
    run_fix_base_epochs
    run_fix_inc_epochs
    ;;
  fix_warmup)
    run_fix_warmup
    ;;
  fix_base_epochs)
    run_fix_base_epochs
    ;;
  fix_inc_epochs)
    run_fix_inc_epochs
    ;;
  *)
    echo "HEATMAP must be all|fix_warmup|fix_base_epochs|fix_inc_epochs" >&2
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
