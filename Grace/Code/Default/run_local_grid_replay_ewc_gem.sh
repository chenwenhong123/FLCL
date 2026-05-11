#!/usr/bin/env bash
set -euo pipefail

# 局部网格搜索（主配置固定，全项目）：
# 1) Replay + EWC （continual_run_ewc.py）
# 2) Replay + GEM （continual_run_gem.py）
#
# 说明：
# - 每种策略最多搜索 3 个关键参数（当前即 3 个）
# - 默认跑全项目：Lang Chart Closure Math Mockito Time
# - 默认采用你当前主配置：lr=0.01, seed=0, bs=60, warm=10, base=15, inc=2
#
# 用法示例：
#   bash run_local_grid_replay_ewc_gem.sh
#   LR=0.005 PROJECTS="Lang Closure" bash run_local_grid_replay_ewc_gem.sh
#   AVG_MIN_PER_RUN=8 bash run_local_grid_replay_ewc_gem.sh

# ------------------------
# 主配置（可调）
# ------------------------
PROJECTS="${PROJECTS:-Lang Chart Closure Math Mockito Time}"
LR="${LR:-0.01}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-60}"
WARMUP_IDS="${WARMUP_IDS:-10}"
BASE_EPOCHS="${BASE_EPOCHS:-15}"
INC_EPOCHS="${INC_EPOCHS:-2}"

# 公共参数（可调）
REPLAY_SIZE="${REPLAY_SIZE:-20}"
TEMPERATURE="${TEMPERATURE:-2.0}"
EWC_GAMMA="${EWC_GAMMA:-0.9}"
FISHER_IDS="${FISHER_IDS:-4}"

# Replay+EWC 局部网格（3个关键参数）
EWC_REPLAY_PER_STEP_LIST="${EWC_REPLAY_PER_STEP_LIST:-1 2 3}"
EWC_LAMBDA_LIST="${EWC_LAMBDA_LIST:-5 10 20}"
EWC_REPLAY_BETA_LIST="${EWC_REPLAY_BETA_LIST:-0.5 1.0}"
# 其余固定
EWC_DISTILL_ALPHA="${EWC_DISTILL_ALPHA:-0.5}"

# Replay+GEM 局部网格（3个关键参数）
GEM_REPLAY_PER_STEP_LIST="${GEM_REPLAY_PER_STEP_LIST:-1 2 3}"
GEM_REPLAY_BETA_LIST="${GEM_REPLAY_BETA_LIST:-0.5 1.0 1.5}"
GEM_DISTILL_ALPHA_LIST="${GEM_DISTILL_ALPHA_LIST:-0.3 0.5}"

EWC_SCRIPT="${EWC_SCRIPT:-continual_run_ewc.py}"
GEM_SCRIPT="${GEM_SCRIPT:-continual_run_gem.py}"

# 仅用于总时长估计，可按机器实测调整
AVG_MIN_PER_RUN="${AVG_MIN_PER_RUN:-10}"

if [[ ! -f "${EWC_SCRIPT}" ]]; then
  echo "Error: cannot find ${EWC_SCRIPT} in $(pwd)"
  exit 1
fi
if [[ ! -f "${GEM_SCRIPT}" ]]; then
  echo "Error: cannot find ${GEM_SCRIPT} in $(pwd)"
  exit 1
fi

mkdir -p logs
mkdir -p grid_outputs/ewc grid_outputs/gem

count_words() {
  # shellcheck disable=SC2086
  set -- $1
  echo $#
}

num_projects=$(count_words "${PROJECTS}")
n_ewc_rps=$(count_words "${EWC_REPLAY_PER_STEP_LIST}")
n_ewc_lambda=$(count_words "${EWC_LAMBDA_LIST}")
n_ewc_beta=$(count_words "${EWC_REPLAY_BETA_LIST}")
n_gem_rps=$(count_words "${GEM_REPLAY_PER_STEP_LIST}")
n_gem_beta=$(count_words "${GEM_REPLAY_BETA_LIST}")
n_gem_da=$(count_words "${GEM_DISTILL_ALPHA_LIST}")

ewc_combos=$((n_ewc_rps * n_ewc_lambda * n_ewc_beta))
gem_combos=$((n_gem_rps * n_gem_beta * n_gem_da))
total_runs=$((num_projects * (ewc_combos + gem_combos)))
est_total_min=$((total_runs * AVG_MIN_PER_RUN))
est_h=$((est_total_min / 60))
est_m=$((est_total_min % 60))

echo "==== Local Grid: Replay+EWC / Replay+GEM ===="
echo "PROJECTS=${PROJECTS}"
echo "MainConfig: lr=${LR} seed=${SEED} bs=${BATCH_SIZE} warm=${WARMUP_IDS} base=${BASE_EPOCHS} inc=${INC_EPOCHS}"
echo
echo "[Replay+EWC] key params:"
echo "  replay_per_step: ${EWC_REPLAY_PER_STEP_LIST}"
echo "  ewc_lambda:      ${EWC_LAMBDA_LIST}"
echo "  replay_beta:     ${EWC_REPLAY_BETA_LIST}"
echo "  fixed: distill_alpha=${EWC_DISTILL_ALPHA}, replay_size=${REPLAY_SIZE}, temperature=${TEMPERATURE}, ewc_gamma=${EWC_GAMMA}, fisher_ids=${FISHER_IDS}"
echo
echo "[Replay+GEM] key params:"
echo "  replay_per_step: ${GEM_REPLAY_PER_STEP_LIST}"
echo "  replay_beta:     ${GEM_REPLAY_BETA_LIST}"
echo "  distill_alpha:   ${GEM_DISTILL_ALPHA_LIST}"
echo "  fixed: replay_size=${REPLAY_SIZE}, temperature=${TEMPERATURE}"
echo
echo "EWC combos/project: ${ewc_combos}"
echo "GEM combos/project: ${gem_combos}"
echo "Total runs: ${total_runs} (${num_projects} projects)"
echo "Estimated time: ~${est_h}h ${est_m}m (AVG_MIN_PER_RUN=${AVG_MIN_PER_RUN})"
echo "Logs dir: logs/"
echo

run_ewc() {
  local p="$1" rps="$2" el="$3" rb="$4"
  local tag="${p}_ewc_rps${rps}_el${el}_rb${rb}"
  echo ">>> [EWC] ${tag}"
  python "${EWC_SCRIPT}" \
    "${p}" "${LR}" "${SEED}" "${BATCH_SIZE}" \
    "${WARMUP_IDS}" "${BASE_EPOCHS}" "${INC_EPOCHS}" \
    "${REPLAY_SIZE}" "${rps}" "${rb}" "${EWC_DISTILL_ALPHA}" "${TEMPERATURE}" \
    "${el}" "${EWC_GAMMA}" "${FISHER_IDS}" \
    | tee "logs/${tag}.log"

  # 防覆盖归档：ewc.py原始输出文件名不含网格参数，需按tag复制
  local out_dir="result/${p}"
  local base_name="${p}_continual_with_replay_ewc"
  local suffix="base${BASE_EPOCHS}_inc${INC_EPOCHS}_lr${LR}_bs${BATCH_SIZE}_warm${WARMUP_IDS}"
  cp "${out_dir}/${base_name}_summary_${suffix}.txt" "grid_outputs/ewc/${tag}_summary.txt"
  cp "${out_dir}/${base_name}_metrics_${suffix}.csv" "grid_outputs/ewc/${tag}_metrics.csv"
  cp "${out_dir}/${base_name}_model_${suffix}.pt" "grid_outputs/ewc/${tag}_model.pt"
}

run_gem() {
  local p="$1" rps="$2" rb="$3" da="$4"
  local tag="${p}_gem_rps${rps}_rb${rb}_da${da}"
  echo ">>> [GEM] ${tag}"
  python "${GEM_SCRIPT}" \
    "${p}" "${LR}" "${SEED}" "${BATCH_SIZE}" \
    "${WARMUP_IDS}" "${BASE_EPOCHS}" "${INC_EPOCHS}" \
    "${REPLAY_SIZE}" "${rps}" "${rb}" "${da}" "${TEMPERATURE}" \
    | tee "logs/${tag}.log"

  # 防覆盖归档：gem.py原始输出文件名不含网格参数，需按tag复制
  local out_dir="result/${p}"
  local base_name="${p}_continual_with_replay_gem"
  local suffix="base${BASE_EPOCHS}_inc${INC_EPOCHS}_lr${LR}_bs${BATCH_SIZE}_warm${WARMUP_IDS}"
  cp "${out_dir}/${base_name}_summary_${suffix}.txt" "grid_outputs/gem/${tag}_summary.txt"
  cp "${out_dir}/${base_name}_metrics_${suffix}.csv" "grid_outputs/gem/${tag}_metrics.csv"
  cp "${out_dir}/${base_name}_model_${suffix}.pt" "grid_outputs/gem/${tag}_model.pt"
}

for p in ${PROJECTS}; do
  # Replay + EWC local grid
  for rps in ${EWC_REPLAY_PER_STEP_LIST}; do
    for el in ${EWC_LAMBDA_LIST}; do
      for rb in ${EWC_REPLAY_BETA_LIST}; do
        run_ewc "${p}" "${rps}" "${el}" "${rb}"
      done
    done
  done

  # Replay + GEM local grid
  for rps in ${GEM_REPLAY_PER_STEP_LIST}; do
    for rb in ${GEM_REPLAY_BETA_LIST}; do
      for da in ${GEM_DISTILL_ALPHA_LIST}; do
        run_gem "${p}" "${rps}" "${rb}" "${da}"
      done
    done
  done
done

echo
echo "All local grid runs completed."
