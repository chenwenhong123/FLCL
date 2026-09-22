#!/usr/bin/env bash
# 组合消融实验（相对基础增量：无策略、adam + L2）
#
# 编号  名称            Replay监督  DER++蒸馏  EWC  A-GEM  入口
# 1     replay          开          关         关   关     continual_replay.py  (beta=1, da=0)
# 2     derpp           开          开         关   关     continual_replay.py  (beta=1, da=0.5) 主配置
# 3     ewc             关          关         开   关     continual_stream_train.py --use_ewc true
# 4     replay_ewc      开          开         开   关     continual_replay.py  --use_ewc true
# 5     replay_gem      开          开         关   开     continual_gem.py     --use_ewc false
# 6     replay_ewc_gem  开          开         开   开     continual_gem.py     --use_ewc true
# 7     sgd_nol2        关          关         关   关     stream  optimizer=sgd  use_l2=false
# 8     adam_nol2       关          关         关   关     stream  optimizer=adam use_l2=false
# 9     sgd_l2          关          关         关   关     stream  optimizer=sgd  use_l2=true
#
# 7–9 是「相对基础增量（adam+L2、无策略）」的优化器/L2 对照，不重复 adam+L2。
#
# 默认模型与 run_10 一致：mlp / mlp_dfl_1 / birnn；6 个项目全版本。
#
# 服务器：
#   cd /path/to/DeepFaultLocalization
#   export PYTHON=python3 DATA_ROOT=. CUDA_VISIBLE_DEVICES=0
#   nohup bash scripts/run_11_module_ablation.sh > logs/run_11_$(date +%m%d%H%M).log 2>&1 &
#
# 只跑部分编号：
#   ONLY="1 2 3" bash scripts/run_11_module_ablation.sh
#   ONLY="derpp replay_gem" bash scripts/run_11_module_ablation.sh
#
# 覆盖模型：
#   MODELS="mlp_dfl_1 birnn" bash scripts/run_11_module_ablation.sh

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-.}"
TECH="${TECH:-DeepFL_CL}"
LOSS="${LOSS:-softmax}"
EPOCHS="${EPOCHS:-50}"
DUMP_STEP="${DUMP_STEP:-10}"
WARMUP="${WARMUP:-10}"
INCR_EPOCHS="${INCR_EPOCHS:-1}"
GPU_MEM="${GPU_MEM:-0.2}"

REPLAY_SIZE="${REPLAY_SIZE:-20}"
REPLAY_PER_STEP="${REPLAY_PER_STEP:-2}"
REPLAY_BETA="${REPLAY_BETA:-1.0}"
DISTILL_ALPHA="${DISTILL_ALPHA:-0.5}"
TEMPERATURE="${TEMPERATURE:-2.0}"
EWC_LAMBDA="${EWC_LAMBDA:-10.0}"
EWC_GAMMA="${EWC_GAMMA:-0.9}"

STAMP="$(date +%m%d%H%M)"
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/${STAMP}_module_ablation.log"
FAIL_LOG="${LOG_DIR}/${STAMP}_module_ablation_fail.txt"
: > "${FAIL_LOG}"
OUT_ROOT="${ROOT}/result_ablation/${STAMP}"
mkdir -p "${OUT_ROOT}"

SUBJECTS=(Chart Lang Math Time Closure Mockito)
if [[ -n "${MODELS:-}" ]]; then
  read -r -a MODELS <<< "${MODELS}"
else
  MODELS=(mlp mlp_dfl_1 birnn)
fi

ONLY_RAW="${ONLY:-}"
n_fail=0
n_ok=0

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${MASTER_LOG}"
}

should_run() {
  local id="$1"
  local name="$2"
  if [[ -z "${ONLY_RAW}" ]]; then
    return 0
  fi
  local tok
  for tok in ${ONLY_RAW}; do
    if [[ "${tok}" == "${id}" || "${tok}" == "${name}" ]]; then
      return 0
    fi
  done
  return 1
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
  local exp_id="$1" name="$2" use_ewc="$3" optimizer="$4" use_l2="$5"
  local sub model run_dir
  for sub in "${SUBJECTS[@]}"; do
    for model in "${MODELS[@]}"; do
      run_dir="${OUT_ROOT}/${exp_id}_${name}/${model}/${sub}"
      mkdir -p "${run_dir}"
      run_or_record "${exp_id}_${name} stream sub=${sub} model=${model}" \
        continual_stream_train.py "${sub}" "${model}" \
          --data_root "${DATA_ROOT}" \
          --out_dir "${run_dir}" \
          --tech "${TECH}" \
          --loss "${LOSS}" \
          --warmup "${WARMUP}" \
          --training_epochs "${EPOCHS}" \
          --incr_epochs "${INCR_EPOCHS}" \
          --dump_step "${DUMP_STEP}" \
          --gpu_mem "${GPU_MEM}" \
          --use_ewc "${use_ewc}" \
          --ewc_lambda "${EWC_LAMBDA}" \
          --ewc_gamma "${EWC_GAMMA}" \
          --optimizer "${optimizer}" \
          --use_l2 "${use_l2}"
    done
  done
}

run_replay() {
  local exp_id="$1" name="$2" use_ewc="$3" replay_beta="$4" distill_alpha="$5"
  local sub model run_dir
  for sub in "${SUBJECTS[@]}"; do
    for model in "${MODELS[@]}"; do
      run_dir="${OUT_ROOT}/${exp_id}_${name}/${model}/${sub}"
      mkdir -p "${run_dir}"
      run_or_record "${exp_id}_${name} replay sub=${sub} model=${model}" \
        continual_replay.py "${sub}" "${model}" \
          --data_root "${DATA_ROOT}" \
          --out_dir "${run_dir}" \
          --tech "${TECH}" \
          --loss "${LOSS}" \
          --warmup "${WARMUP}" \
          --training_epochs "${EPOCHS}" \
          --incr_epochs "${INCR_EPOCHS}" \
          --dump_step "${DUMP_STEP}" \
          --gpu_mem "${GPU_MEM}" \
          --use_replay true \
          --use_ewc "${use_ewc}" \
          --replay_size "${REPLAY_SIZE}" \
          --replay_per_step "${REPLAY_PER_STEP}" \
          --replay_beta "${replay_beta}" \
          --distill_alpha "${distill_alpha}" \
          --temperature "${TEMPERATURE}" \
          --ewc_lambda "${EWC_LAMBDA}" \
          --ewc_gamma "${EWC_GAMMA}"
    done
  done
}

run_gem() {
  local exp_id="$1" name="$2" use_ewc="$3"
  local sub model run_dir
  for sub in "${SUBJECTS[@]}"; do
    for model in "${MODELS[@]}"; do
      run_dir="${OUT_ROOT}/${exp_id}_${name}/${model}/${sub}"
      mkdir -p "${run_dir}"
      run_or_record "${exp_id}_${name} gem sub=${sub} model=${model}" \
        continual_gem.py "${sub}" "${model}" \
          --data_root "${DATA_ROOT}" \
          --out_dir "${run_dir}" \
          --tech "${TECH}" \
          --loss "${LOSS}" \
          --warmup "${WARMUP}" \
          --training_epochs "${EPOCHS}" \
          --incr_epochs "${INCR_EPOCHS}" \
          --dump_step "${DUMP_STEP}" \
          --gpu_mem "${GPU_MEM}" \
          --use_replay true \
          --use_ewc "${use_ewc}" \
          --replay_size "${REPLAY_SIZE}" \
          --replay_per_step "${REPLAY_PER_STEP}" \
          --replay_beta "${REPLAY_BETA}" \
          --distill_alpha "${DISTILL_ALPHA}" \
          --temperature "${TEMPERATURE}" \
          --ewc_lambda "${EWC_LAMBDA}" \
          --ewc_gamma "${EWC_GAMMA}"
    done
  done
}

log "ROOT=${ROOT}"
log "PYTHON=${PYTHON} DATA_ROOT=${DATA_ROOT} TECH=${TECH}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
log "STAMP=${STAMP} ONLY=${ONLY_RAW:-all}"
log "MODELS=${MODELS[*]}"
log "OUT_ROOT=${OUT_ROOT}"

if should_run 1 replay; then
  log "======== 1 replay: 监督回放，无蒸馏/EWC/GEM (da=0) ========"
  run_replay 1 replay false "${REPLAY_BETA}" 0.0
fi
if should_run 2 derpp; then
  log "======== 2 derpp: 监督回放+蒸馏 主配置 (da=${DISTILL_ALPHA}) ========"
  run_replay 2 derpp false "${REPLAY_BETA}" "${DISTILL_ALPHA}"
fi
if should_run 3 ewc; then
  log "======== 3 ewc: 仅 EWC ========"
  run_stream 3 ewc true adam true
fi
if should_run 4 replay_ewc; then
  log "======== 4 replay_ewc: DER++ + EWC ========"
  run_replay 4 replay_ewc true "${REPLAY_BETA}" "${DISTILL_ALPHA}"
fi
if should_run 5 replay_gem; then
  log "======== 5 replay_gem: DER++ + A-GEM ========"
  run_gem 5 replay_gem false
fi
if should_run 6 replay_ewc_gem; then
  log "======== 6 replay_ewc_gem: DER++ + EWC + A-GEM ========"
  run_gem 6 replay_ewc_gem true
fi
if should_run 7 sgd_nol2; then
  log "======== 7 sgd_nol2: 无策略，sgd + 无 L2 ========"
  run_stream 7 sgd_nol2 false sgd false
fi
if should_run 8 adam_nol2; then
  log "======== 8 adam_nol2: 无策略，adam + 无 L2 ========"
  run_stream 8 adam_nol2 false adam false
fi
if should_run 9 sgd_l2; then
  log "======== 9 sgd_l2: 无策略，sgd + L2 ========"
  run_stream 9 sgd_l2 false sgd true
fi

log "======== DONE ok=${n_ok} fail=${n_fail} ========"
log "master log: ${MASTER_LOG}"
log "fail list:  ${FAIL_LOG}"
log "out:        ${OUT_ROOT}"

if [[ "${n_fail}" -gt 0 ]]; then
  exit 1
fi
exit 0
