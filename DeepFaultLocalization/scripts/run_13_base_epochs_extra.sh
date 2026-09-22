#!/usr/bin/env bash
# 补跑 base_epochs 高档：{30, 35, 40, 45}
# 固定组合与 run_12 一致：EWC + mlp_dfl_1，adam+L2，warmup=10，inc_epochs=2
# 钉住的 base_epochs 默认 50 由 run_12 中心点覆盖，本脚本不再跑 e=50。
#
# 规模：4 档 × 6 项目 = 24 次
#
# 服务器：
#   cd /path/to/DeepFaultLocalization
#   export PYTHON=python3 DATA_ROOT=. CUDA_VISIBLE_DEVICES=0
#   nohup bash scripts/run_13_base_epochs_extra.sh > logs/run_13_$(date +%m%d%H%M).log 2>&1 &

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

WARMUP="${WARMUP:-10}"
INCR_EPOCHS="${INCR_EPOCHS:-2}"
EPOCH_GRID=(30 35 40 45)

SUBJECTS=(Chart Lang Math Time Closure Mockito)
if [[ -n "${MODELS:-}" ]]; then
  read -r -a MODELS <<< "${MODELS}"
else
  MODELS=(mlp_dfl_1)
fi

STAMP="$(date +%m%d%H%M)"
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/${STAMP}_base_epochs_extra.log"
FAIL_LOG="${LOG_DIR}/${STAMP}_base_epochs_extra_fail.txt"
: > "${FAIL_LOG}"
OUT_ROOT="${ROOT}/result_stream_schedule/${STAMP}"
mkdir -p "${OUT_ROOT}"

n_fail=0
n_ok=0

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${MASTER_LOG}"
}

run_one() {
  local epochs="$1" sub="$2" model="$3"
  local tag="base_epochs_e${epochs}"
  local run_dir="${OUT_ROOT}/${tag}/${model}/${sub}"
  mkdir -p "${run_dir}"
  log "START ${tag} sub=${sub} model=${model} warmup=${WARMUP} base_epochs=${epochs} inc_epochs=${INCR_EPOCHS}"
  if "${PYTHON}" -u continual_stream_train.py "${sub}" "${model}" \
      --data_root "${DATA_ROOT}" \
      --out_dir "${run_dir}" \
      --tech "${TECH}" \
      --loss "${LOSS}" \
      --warmup "${WARMUP}" \
      --training_epochs "${epochs}" \
      --incr_epochs "${INCR_EPOCHS}" \
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
  echo "${tag} sub=${sub} model=${model} base_epochs=${epochs}" >> "${FAIL_LOG}"
  return 0
}

log "ROOT=${ROOT}"
log "PYTHON=${PYTHON} ($(command -v "${PYTHON}" || true))"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
log "OUT_ROOT=${OUT_ROOT}"
log "fixed: use_ewc=${USE_EWC} optimizer=${OPTIMIZER} use_l2=${USE_L2} warmup=${WARMUP} inc_epochs=${INCR_EPOCHS}"
log "base_epochs grid={${EPOCH_GRID[*]}} MODELS=${MODELS[*]}"

for e in "${EPOCH_GRID[@]}"; do
  log "======== base_epochs=${e} ========"
  for sub in "${SUBJECTS[@]}"; do
    for model in "${MODELS[@]}"; do
      run_one "${e}" "${sub}" "${model}"
    done
  done
done

log "======== DONE ok=${n_ok} fail=${n_fail} ========"
log "master log: ${MASTER_LOG}"
log "fail list:  ${FAIL_LOG}"
log "out:        ${OUT_ROOT}"

if [[ "${n_fail}" -gt 0 ]]; then
  exit 1
fi
exit 0
