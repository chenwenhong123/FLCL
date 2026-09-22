#!/usr/bin/env bash
set -euo pipefail

# 组合消融：在「已完成的 one-shot / 基础增量」之上，拆开现有执行脚本里的策略开关。
# 脚本会自动 cd 到 Code/Default。
#
# 用法：
#   bash scripts/run_ablation_all_projects.sh
#   PROJECTS="Lang" CUDA_VISIBLE_DEVICES=1 bash scripts/run_ablation_all_projects.sh
#   RUN_CL=1 RUN_OPT=0 bash scripts/run_ablation_all_projects.sh
#   CASES="replay derpp ewc replay_ewc replay_gem replay_ewc_gem" bash scripts/run_ablation_all_projects.sh
#
# 默认跳过：
#   one-shot、naive 基础增量（continual_run.py）——你已经跑完。
#   设 RUN_NAIVE=1 才会再跑 naive。
#
# 主配置与已完成实验对齐：lr=0.01 seed=0 bs=60 warm=10 base=15 inc=2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_DIR}"

PROJECTS="${PROJECTS:-Lang Chart Closure Math Mockito Time}"
LR="${LR:-0.01}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-60}"
WARMUP_IDS="${WARMUP_IDS:-10}"
BASE_EPOCHS="${BASE_EPOCHS:-15}"
INC_EPOCHS="${INC_EPOCHS:-2}"
PYTHON="${PYTHON:-python}"

REPLAY_SIZE="${REPLAY_SIZE:-20}"
REPLAY_PER_STEP="${REPLAY_PER_STEP:-2}"
REPLAY_BETA="${REPLAY_BETA:-1.0}"
DISTILL_ALPHA="${DISTILL_ALPHA:-0.5}"
TEMPERATURE="${TEMPERATURE:-2.0}"
EWC_LAMBDA="${EWC_LAMBDA:-10.0}"
EWC_GAMMA="${EWC_GAMMA:-0.9}"
FISHER_IDS="${FISHER_IDS:-4}"

SGD_MOMENTUM="${SGD_MOMENTUM:-0.9}"
SGD_WEIGHT_DECAY="${SGD_WEIGHT_DECAY:-1e-4}"
ADAM_WEIGHT_DECAY="${ADAM_WEIGHT_DECAY:-1e-4}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"

RUN_NAIVE="${RUN_NAIVE:-0}"
RUN_CL="${RUN_CL:-1}"
RUN_OPT="${RUN_OPT:-1}"
# 空格分隔；默认不含 naive（由 RUN_NAIVE 控制）
CASES="${CASES:-replay derpp ewc replay_ewc replay_gem replay_ewc_gem}"

mkdir -p logs result

echo "==== Strategy ablation ===="
echo "CODE_DIR=${CODE_DIR}"
echo "PROJECTS=${PROJECTS}"
echo "LR=${LR} SEED=${SEED} BATCH_SIZE=${BATCH_SIZE}"
echo "WARMUP_IDS=${WARMUP_IDS} BASE_EPOCHS=${BASE_EPOCHS} INC_EPOCHS=${INC_EPOCHS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset, all GPUs visible>}"
echo "RUN_NAIVE=${RUN_NAIVE} RUN_CL=${RUN_CL} RUN_OPT=${RUN_OPT}"
echo "CASES=${CASES}"
echo "replay: size=${REPLAY_SIZE} per_step=${REPLAY_PER_STEP} beta=${REPLAY_BETA} da=${DISTILL_ALPHA} T=${TEMPERATURE}"
echo "ewc: lambda=${EWC_LAMBDA} gamma=${EWC_GAMMA} fisher_ids=${FISHER_IDS}"
echo

need() {
  if [[ ! -f "$1" ]]; then
    echo "Error: missing $1 in ${CODE_DIR}"
    exit 1
  fi
}

need continual_run.py
need continual_run_with_replay.py
need continual_run_ewc.py
need continual_run_gem.py
need continual_run_all.py
need continual_run_sgd_l2.py
need continual_run_adam_l2.py

pkl_path() {
  local project="$1"
  if [[ -f "${project}.pkl" ]]; then
    echo "${project}.pkl"
  else
    echo "GraceDate/${project}.pkl"
  fi
}

for project in ${PROJECTS}; do
  path="$(pkl_path "${project}")"
  if [[ ! -f "${path}" ]]; then
    echo "Error: data file not found: ${path}"
    exit 1
  fi
done

run_logged() {
  local tag="$1"
  shift
  local log="logs/${tag}.log"
  echo ">>> ${tag}"
  echo "    log=${log}"
  echo "    cmd: ${PYTHON} $*"
  "${PYTHON}" "$@" | tee "${log}"
  echo ">>> ${tag} done"
  echo
}

case_enabled() {
  local name="$1"
  for c in ${CASES}; do
    if [[ "${c}" == "${name}" ]]; then
      return 0
    fi
  done
  return 1
}

# ---------- 抗遗忘策略组合 ----------
# Replay / DER++ / EWC / A-GEM 的合法组合（蒸馏依赖 replay 样本，GEM 依赖 replay 参考梯度）

run_naive() {
  local p="$1"
  run_logged "${p}_ablate_naive_base${BASE_EPOCHS}_inc${INC_EPOCHS}_lr${LR}_bs${BATCH_SIZE}_warm${WARMUP_IDS}" \
    continual_run.py "${p}" "${LR}" "${SEED}" "${BATCH_SIZE}" \
    "${WARMUP_IDS}" "${BASE_EPOCHS}" "${INC_EPOCHS}"
}

run_replay() {
  # 仅经验回放监督，关闭 DER++ 蒸馏
  local p="$1"
  run_logged "${p}_ablate_replay_rs${REPLAY_SIZE}_rps${REPLAY_PER_STEP}_rb${REPLAY_BETA}" \
    continual_run_with_replay.py "${p}" "${LR}" "${SEED}" "${BATCH_SIZE}" \
    "${WARMUP_IDS}" "${BASE_EPOCHS}" "${INC_EPOCHS}" \
    "${REPLAY_SIZE}" "${REPLAY_PER_STEP}" "${REPLAY_BETA}" "0.0" "${TEMPERATURE}"
}

run_derpp() {
  # Replay + DER++（回放监督 + 蒸馏），主配置
  local p="$1"
  run_logged "${p}_ablate_derpp_rs${REPLAY_SIZE}_rps${REPLAY_PER_STEP}_rb${REPLAY_BETA}_da${DISTILL_ALPHA}" \
    continual_run_with_replay.py "${p}" "${LR}" "${SEED}" "${BATCH_SIZE}" \
    "${WARMUP_IDS}" "${BASE_EPOCHS}" "${INC_EPOCHS}" \
    "${REPLAY_SIZE}" "${REPLAY_PER_STEP}" "${REPLAY_BETA}" "${DISTILL_ALPHA}" "${TEMPERATURE}"
}

run_ewc() {
  # 仅 EWC，关闭 replay / 蒸馏
  local p="$1"
  run_logged "${p}_ablate_ewc_el${EWC_LAMBDA}_eg${EWC_GAMMA}_fid${FISHER_IDS}" \
    continual_run_ewc.py "${p}" "${LR}" "${SEED}" "${BATCH_SIZE}" \
    "${WARMUP_IDS}" "${BASE_EPOCHS}" "${INC_EPOCHS}" \
    "0" "0" "0.0" "0.0" "${TEMPERATURE}" \
    "${EWC_LAMBDA}" "${EWC_GAMMA}" "${FISHER_IDS}"
}

run_replay_ewc() {
  # Replay + DER++ + EWC
  local p="$1"
  run_logged "${p}_ablate_replay_ewc_el${EWC_LAMBDA}_rps${REPLAY_PER_STEP}_rb${REPLAY_BETA}" \
    continual_run_ewc.py "${p}" "${LR}" "${SEED}" "${BATCH_SIZE}" \
    "${WARMUP_IDS}" "${BASE_EPOCHS}" "${INC_EPOCHS}" \
    "${REPLAY_SIZE}" "${REPLAY_PER_STEP}" "${REPLAY_BETA}" "${DISTILL_ALPHA}" "${TEMPERATURE}" \
    "${EWC_LAMBDA}" "${EWC_GAMMA}" "${FISHER_IDS}"
}

run_replay_gem() {
  # Replay + DER++ + A-GEM
  local p="$1"
  run_logged "${p}_ablate_replay_gem_rps${REPLAY_PER_STEP}_rb${REPLAY_BETA}_da${DISTILL_ALPHA}" \
    continual_run_gem.py "${p}" "${LR}" "${SEED}" "${BATCH_SIZE}" \
    "${WARMUP_IDS}" "${BASE_EPOCHS}" "${INC_EPOCHS}" \
    "${REPLAY_SIZE}" "${REPLAY_PER_STEP}" "${REPLAY_BETA}" "${DISTILL_ALPHA}" "${TEMPERATURE}"
}

run_replay_ewc_gem() {
  # Replay + DER++ + EWC + A-GEM
  local p="$1"
  run_logged "${p}_ablate_replay_ewc_gem_el${EWC_LAMBDA}_rps${REPLAY_PER_STEP}" \
    continual_run_all.py "${p}" "${LR}" "${SEED}" "${BATCH_SIZE}" \
    "${WARMUP_IDS}" "${BASE_EPOCHS}" "${INC_EPOCHS}" \
    "${REPLAY_SIZE}" "${REPLAY_PER_STEP}" "${REPLAY_BETA}" "${DISTILL_ALPHA}" "${TEMPERATURE}" \
    "${EWC_LAMBDA}" "${EWC_GAMMA}" "${FISHER_IDS}"
}

run_cl_cases() {
  local p="$1"
  if case_enabled replay; then run_replay "${p}"; fi
  if case_enabled derpp; then run_derpp "${p}"; fi
  if case_enabled ewc; then run_ewc "${p}"; fi
  if case_enabled replay_ewc; then run_replay_ewc "${p}"; fi
  if case_enabled replay_gem; then run_replay_gem "${p}"; fi
  if case_enabled replay_ewc_gem; then run_replay_ewc_gem "${p}"; fi
}

# ---------- 优化器 / L2（作用在 naive 增量骨干上，与抗遗忘模块正交） ----------

run_opt_cases() {
  local p="$1"
  run_logged "${p}_ablate_sgd_nol2_mom${SGD_MOMENTUM}" \
    continual_run_sgd_l2.py "${p}" "${LR}" "${SEED}" "${BATCH_SIZE}" \
    "${WARMUP_IDS}" "${BASE_EPOCHS}" "${INC_EPOCHS}" \
    "0.0" "${SGD_MOMENTUM}"
  run_logged "${p}_ablate_sgd_l2_wd${SGD_WEIGHT_DECAY}_mom${SGD_MOMENTUM}" \
    continual_run_sgd_l2.py "${p}" "${LR}" "${SEED}" "${BATCH_SIZE}" \
    "${WARMUP_IDS}" "${BASE_EPOCHS}" "${INC_EPOCHS}" \
    "${SGD_WEIGHT_DECAY}" "${SGD_MOMENTUM}"
  run_logged "${p}_ablate_adam_l2_wd${ADAM_WEIGHT_DECAY}" \
    continual_run_adam_l2.py "${p}" "${LR}" "${SEED}" "${BATCH_SIZE}" \
    "${WARMUP_IDS}" "${BASE_EPOCHS}" "${INC_EPOCHS}" \
    "${ADAM_WEIGHT_DECAY}" "${ADAM_BETA1}" "${ADAM_BETA2}"
}

for project in ${PROJECTS}; do
  echo "===== project ${project} ====="
  if [[ "${RUN_CL}" == "1" ]]; then
    if [[ "${RUN_NAIVE}" == "1" ]]; then
      run_naive "${project}"
    fi
    run_cl_cases "${project}"
  elif [[ "${RUN_NAIVE}" == "1" ]]; then
    run_naive "${project}"
  fi
  if [[ "${RUN_OPT}" == "1" ]]; then
    run_opt_cases "${project}"
  fi
done

echo "==== Ablation finished ===="
echo "结果目录: result/<Project>/<策略前缀>_..._ts*/{metrics.csv,model.pt,summary.txt}"
echo "日志: logs/<Project>_ablate_*.log"
echo
echo "对照已完成实验时："
echo "  one-shot  -> result/<Project>/*result_final_* 与各 bug pkl"
echo "  naive 增量 -> result/<Project>/continual_base${BASE_EPOCHS}_inc${INC_EPOCHS}_*"
