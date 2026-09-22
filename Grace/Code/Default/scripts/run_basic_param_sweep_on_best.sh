#!/usr/bin/env bash
set -euo pipefail

# 在「指标最好的策略组合」上，对基础增量协议参数做一维 5 点扫描。
#
# 选定策略（六项目 macro Top-1 最高）：
#   Replay 监督，关闭 DER++ 蒸馏
#   python continual_run_with_replay.py ... replay_size=20 rps=2 beta=1.0 distill_alpha=0.0 T=2.0
#   不扫描 replay_size / replay_per_step（非基础增量参数）。
#   lr / batch_size 固定为源码配置 0.01 / 60，不搜索。
#
# 扫描方式：每次只改一个参数，其余钉在主配置。
# 中心点 (warm=10, base=15, inc=2) 已有结果时默认跳过。
#
# 用法：
#   bash scripts/run_basic_param_sweep_on_best.sh
#   PROJECTS="Lang" CUDA_VISIBLE_DEVICES=1 bash scripts/run_basic_param_sweep_on_best.sh
#   SWEEPS="warmup base_epochs" bash scripts/run_basic_param_sweep_on_best.sh
#   SKIP_CENTER=0 bash scripts/run_basic_param_sweep_on_best.sh   # 连中心点也重跑

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_DIR}"

PROJECTS="${PROJECTS:-Lang Chart Closure Math Mockito Time}"
PYTHON="${PYTHON:-python}"
SCRIPT="${SCRIPT:-continual_run_with_replay.py}"

SEED="${SEED:-0}"
# 与源码 / 已完成实验一致，不搜索
LR="${LR:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-60}"
CENTER_WARM="${CENTER_WARM:-10}"
CENTER_BASE="${CENTER_BASE:-15}"
CENTER_INC="${CENTER_INC:-2}"

# 策略侧固定（最好组合：replay-only）
REPLAY_SIZE="${REPLAY_SIZE:-20}"
REPLAY_PER_STEP="${REPLAY_PER_STEP:-2}"
REPLAY_BETA="${REPLAY_BETA:-1.0}"
DISTILL_ALPHA="${DISTILL_ALPHA:-0.0}"
TEMPERATURE="${TEMPERATURE:-2.0}"

# 5 点网格（可覆盖）；只扫协议参数
WARMUP_LIST="${WARMUP_LIST:-6 8 10 12 14}"
BASE_EPOCHS_LIST="${BASE_EPOCHS_LIST:-5 10 15 20 25}"
INC_EPOCHS_LIST="${INC_EPOCHS_LIST:-1 2 3 4 5}"

# 空格分隔：warmup base_epochs inc_epochs
SWEEPS="${SWEEPS:-warmup base_epochs inc_epochs}"
SKIP_CENTER="${SKIP_CENTER:-1}"

mkdir -p logs result

if [[ ! -f "${SCRIPT}" ]]; then
  echo "Error: cannot find ${SCRIPT} in ${CODE_DIR}"
  exit 1
fi

echo "==== Basic-param 5-point sweep on best strategy (replay-only) ===="
echo "CODE_DIR=${CODE_DIR}"
echo "PROJECTS=${PROJECTS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "strategy: ${SCRIPT}  rs=${REPLAY_SIZE} rps=${REPLAY_PER_STEP} rb=${REPLAY_BETA} da=${DISTILL_ALPHA} T=${TEMPERATURE}"
echo "fixed (source): lr=${LR} bs=${BATCH_SIZE} seed=${SEED}"
echo "center: warm=${CENTER_WARM} base=${CENTER_BASE} inc=${CENTER_INC}"
echo "SWEEPS=${SWEEPS}  SKIP_CENTER=${SKIP_CENTER}"
echo "warmup:        ${WARMUP_LIST}"
echo "base_epochs:   ${BASE_EPOCHS_LIST}"
echo "inc_epochs:    ${INC_EPOCHS_LIST}"
echo

is_center() {
  local warm="$1" base="$2" inc="$3"
  [[ "${warm}" == "${CENTER_WARM}" && "${base}" == "${CENTER_BASE}" && "${inc}" == "${CENTER_INC}" ]]
}

run_one() {
  local project="$1" warm="$2" base="$3" inc="$4" axis="$5"
  if [[ "${SKIP_CENTER}" == "1" ]] && is_center "${warm}" "${base}" "${inc}"; then
    echo "    skip center (${project} ${axis} warm=${warm} base=${base} inc=${inc})"
    return 0
  fi
  local tag
  tag="${project}_best_replay_sweep_${axis}_warm${warm}_base${base}_inc${inc}"
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

sweep_enabled() {
  local name="$1"
  for s in ${SWEEPS}; do
    if [[ "${s}" == "${name}" ]]; then
      return 0
    fi
  done
  return 1
}

n_runs=0
for project in ${PROJECTS}; do
  echo "===== ${project} ====="
  if sweep_enabled warmup; then
    for warm in ${WARMUP_LIST}; do
      run_one "${project}" "${warm}" "${CENTER_BASE}" "${CENTER_INC}" "warmup"
      n_runs=$((n_runs + 1))
    done
  fi
  if sweep_enabled base_epochs; then
    for base in ${BASE_EPOCHS_LIST}; do
      run_one "${project}" "${CENTER_WARM}" "${base}" "${CENTER_INC}" "base"
      n_runs=$((n_runs + 1))
    done
  fi
  if sweep_enabled inc_epochs; then
    for inc in ${INC_EPOCHS_LIST}; do
      run_one "${project}" "${CENTER_WARM}" "${CENTER_BASE}" "${inc}" "inc"
      n_runs=$((n_runs + 1))
    done
  fi
done

echo "==== Sweep finished (looped ${n_runs} slots; center skips do not train) ===="
echo "Outputs: result/<Project>/replay_base*_inc*_lr*_bs*_warm*_rs${REPLAY_SIZE}_rps${REPLAY_PER_STEP}_rb${REPLAY_BETA}_da${DISTILL_ALPHA}_ts*/"
echo "Logs: logs/<Project>_best_replay_sweep_*.log"
