#!/usr/bin/env bash
# One-shot（main.py，无持续学习）+ 基础流式增量（continual_stream_train.py，无 EWC/Replay/GEM）
#
# 模型（3 个有代表性骨干）：
#   mlp       — 单隐层全连接（最简基线）
#   mlp_dfl_1 — DeepFL 定制分块全连接
#   birnn     — 双向序列模型
#
# 项目：Chart Lang Math Time Closure Mockito（全部版本）
#
# 服务器挂跑示例：
#   cd /path/to/DeepFaultLocalization
#   nohup bash scripts/run_10_oneshot_and_stream_baseline.sh > logs/run_10_$(date +%m%d%H%M).log 2>&1 &
#
# 常用环境变量：
#   PYTHON=python3
#   DATA_ROOT=.                 # 其下需有 DeepFL/（one-shot）与 DeepFL_CL/（增量）
#   CUDA_VISIBLE_DEVICES=0
#   PHASE=all|stream|oneshot    # 默认 all：先增量后 one-shot
#   SKIP_EXISTING=1             # one-shot 已有结果文件则跳过该版本
#
# 预计规模（3 模型 × 6 项目）：
#   增量：18 次流式任务
#   one-shot：3 × (26+65+106+27+133+38) = 1185 次 main.py

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-.}"
PHASE="${PHASE:-all}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
LOSS="${LOSS:-softmax}"
EPOCHS="${EPOCHS:-50}"
DUMP_STEP="${DUMP_STEP:-10}"
WARMUP="${WARMUP:-10}"
INCR_EPOCHS="${INCR_EPOCHS:-1}"
GPU_MEM="${GPU_MEM:-0.2}"
ONESHOT_TECH="${ONESHOT_TECH:-DeepFL}"
STREAM_TECH="${STREAM_TECH:-DeepFL_CL}"

STAMP="$(date +%m%d%H%M)"
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/${STAMP}_oneshot_stream_baseline.log"
FAIL_LOG="${LOG_DIR}/${STAMP}_fail.txt"
: > "${FAIL_LOG}"

ONESHOT_OUT="${ROOT}/result_oneshot/${STAMP}"
STREAM_OUT="${ROOT}/result_stream_baseline/${STAMP}"
mkdir -p "${ONESHOT_OUT}" "${STREAM_OUT}"

MODELS=(mlp mlp_dfl_1 birnn)
SUBJECTS=(Chart Lang Math Time Closure Mockito)
MAX_VERSIONS=(26 65 106 27 133 38)

n_fail=0
n_ok=0

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${MASTER_LOG}"
}

run_or_record() {
  local desc="$1"
  shift
  log "START ${desc}"
  if "${PYTHON}" -u "$@" >>"${MASTER_LOG}" 2>&1; then
    n_ok=$((n_ok + 1))
    log "OK    ${desc}"
    return 0
  fi
  n_fail=$((n_fail + 1))
  log "FAIL  ${desc}"
  echo "${desc}" >> "${FAIL_LOG}"
  return 0
}

run_stream() {
  log "======== PHASE stream: 基础增量（use_ewc=false，无 replay/gem）========"
  log "STREAM_OUT=${STREAM_OUT} STREAM_TECH=${STREAM_TECH} DATA_ROOT=${DATA_ROOT}"
  local sub model run_dir
  for sub in "${SUBJECTS[@]}"; do
    for model in "${MODELS[@]}"; do
      run_dir="${STREAM_OUT}/${model}/${sub}"
      mkdir -p "${run_dir}"
      run_or_record "stream sub=${sub} model=${model}" \
        continual_stream_train.py "${sub}" "${model}" \
          --data_root "${DATA_ROOT}" \
          --out_dir "${run_dir}" \
          --tech "${STREAM_TECH}" \
          --loss "${LOSS}" \
          --warmup "${WARMUP}" \
          --training_epochs "${EPOCHS}" \
          --incr_epochs "${INCR_EPOCHS}" \
          --dump_step "${DUMP_STEP}" \
          --gpu_mem "${GPU_MEM}" \
          --use_ewc false
    done
  done
}

oneshot_done() {
  local out_root="$1" sub="$2" ver="$3" model="$4"
  local p
  p="${out_root}/${sub}/${ver}/${ONESHOT_TECH}/${model}-${LOSS}-${EPOCHS}"
  [[ -f "${p}" ]]
}

run_oneshot() {
  log "======== PHASE oneshot: main.py 逐版本独立训练（无增量策略）========"
  log "ONESHOT_OUT=${ONESHOT_OUT} ONESHOT_TECH=${ONESHOT_TECH} DATA_ROOT=${DATA_ROOT}"
  local i sub vmax v model train_dir
  for model in "${MODELS[@]}"; do
    for i in "${!SUBJECTS[@]}"; do
      sub="${SUBJECTS[$i]}"
      vmax="${MAX_VERSIONS[$i]}"
      for ((v = 1; v <= vmax; v++)); do
        train_dir="${DATA_ROOT}/${ONESHOT_TECH}/${sub}/${v}"
        if [[ ! -f "${train_dir}/Train.csv" ]]; then
          log "SKIP  missing data sub=${sub} v=${v}"
          continue
        fi
        if [[ "${SKIP_EXISTING}" == "1" ]] && oneshot_done "${ONESHOT_OUT}" "${sub}" "${v}" "${model}"; then
          log "SKIP  exists sub=${sub} v=${v} model=${model}"
          continue
        fi
        run_or_record "oneshot sub=${sub} v=${v} model=${model}" \
          main.py \
            "${DATA_ROOT}" \
            "${ONESHOT_OUT}" \
            "${sub}" \
            "${v}" \
            "${model}" \
            "${ONESHOT_TECH}" \
            "${LOSS}" \
            "${EPOCHS}" \
            "${DUMP_STEP}"
      done
    done
    log "rank_parser model=${model}"
    run_or_record "rank_parser model=${model}" \
      rank_parser.py \
        "${DATA_ROOT}" \
        "${ONESHOT_OUT}" \
        "${ONESHOT_TECH}" \
        "${model}" \
        "${LOSS}" \
        "${EPOCHS}" \
        all
  done
}

log "ROOT=${ROOT}"
log "PYTHON=${PYTHON} ($(command -v "${PYTHON}" || true))"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
log "PHASE=${PHASE} STAMP=${STAMP}"
log "MODELS=${MODELS[*]}"

case "${PHASE}" in
  stream) run_stream ;;
  oneshot) run_oneshot ;;
  all)
    run_stream
    run_oneshot
    ;;
  *)
    echo "PHASE must be all|stream|oneshot" >&2
    exit 1
    ;;
esac

log "======== DONE ok=${n_ok} fail=${n_fail} ========"
log "master log: ${MASTER_LOG}"
log "fail list:  ${FAIL_LOG}"
log "stream out: ${STREAM_OUT}"
log "oneshot out:${ONESHOT_OUT}"

if [[ "${n_fail}" -gt 0 ]]; then
  exit 1
fi
exit 0
