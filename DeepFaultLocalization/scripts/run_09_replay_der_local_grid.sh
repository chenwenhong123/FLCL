#!/usr/bin/env bash
# 仅 mlp_dfl_1 + 全项目 的局部网格搜索（两种策略）：
#   1) EWC-only（continual_stream_train.py）
#   2) Replay+GEM（continual_gem.py）
#
# 每种策略最多扫描 3 个关键参数，使用单因子局部搜索（OAT）：
#   baseline + 每个参数各自波动（其余参数固定 baseline）
#
# 一键运行：
#   bash scripts/run_09_replay_der_local_grid.sh
#
# 仅打印计划不执行：
#   DRY_RUN=1 bash scripts/run_09_replay_der_local_grid.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-.}"
TECH="${TECH:-DeepFL_CL}"
LOSS="${LOSS:-softmax}"
MODEL="${MODEL:-mlp_dfl_1}"
WARMUP="${WARMUP:-10}"
EPOCHS="${EPOCHS:-50}"
INCR_EPOCHS="${INCR_EPOCHS:-1}"
DUMP_STEP="${DUMP_STEP:-10}"
GPU_MEM="${GPU_MEM:-0.2}"
DRY_RUN="${DRY_RUN:-0}"

SUBJECTS=(Chart Lang Math Time Closure Mockito)

# ========= EWC-only: 参数波动范围（2个关键参数） =========
# ewc_lambda: 5.0 ~ 20.0
# ewc_gamma : 0.80 ~ 0.95
EWC_BASE_LAMBDA="${EWC_BASE_LAMBDA:-10.0}"
EWC_BASE_GAMMA="${EWC_BASE_GAMMA:-0.90}"
EWC_LAMBDA_GRID=(${EWC_LAMBDA_GRID:-5.0 10.0 20.0})
EWC_GAMMA_GRID=(${EWC_GAMMA_GRID:-0.80 0.90 0.95})

# ========= Replay+GEM: 参数波动范围（3个关键参数） =========
# replay_per_step: 1 ~ 4
# replay_beta    : 0.5 ~ 2.0
# distill_alpha  : 0.3 ~ 0.7
# （temperature 固定为 2.0；replay_size 固定为 20）
RG_BASE_RPS="${RG_BASE_RPS:-2}"
RG_BASE_BETA="${RG_BASE_BETA:-1.0}"
RG_BASE_DA="${RG_BASE_DA:-0.5}"
RG_FIXED_TEMP="${RG_FIXED_TEMP:-2.0}"
RG_FIXED_REPLAY_SIZE="${RG_FIXED_REPLAY_SIZE:-20}"
RG_RPS_GRID=(${RG_RPS_GRID:-1 2 4})
RG_BETA_GRID=(${RG_BETA_GRID:-0.5 1.0 2.0})
RG_DA_GRID=(${RG_DA_GRID:-0.3 0.5 0.7})

OUT_ROOT="${ROOT}/result_grid_local_mlpdfl1"
mkdir -p "${OUT_ROOT}"
PLAN_FILE="${OUT_ROOT}/grid_plan.tsv"
echo -e "grid_id\tstrategy\tvariant\tewc_lambda\tewc_gamma\treplay_size\treplay_per_step\treplay_beta\tdistill_alpha\ttemperature" > "${PLAN_FILE}"

grid_id=0

run_ewc_cfg() {
  local variant="$1"
  local ewc_lambda="$2"
  local ewc_gamma="$3"
  grid_id=$((grid_id + 1))
  local cfg_id
  cfg_id="$(printf "%03d" "${grid_id}")"
  local out_cfg="${OUT_ROOT}/ewc_only_${cfg_id}"
  mkdir -p "${out_cfg}"
  echo -e "${cfg_id}\tewc_only\t${variant}\t${ewc_lambda}\t${ewc_gamma}\t-\t-\t-\t-\t-" >> "${PLAN_FILE}"

  echo "========== GRID ${cfg_id} [EWC-only/${variant}] =========="
  echo "model=${MODEL} ewc_lambda=${ewc_lambda} ewc_gamma=${ewc_gamma} out=${out_cfg}"
  [[ "${DRY_RUN}" == "1" ]] && return 0

  for sub in "${SUBJECTS[@]}"; do
    local run_dir="${out_cfg}/${sub}"
    mkdir -p "${run_dir}"
    "${PYTHON}" continual_stream_train.py "${sub}" "${MODEL}" \
      --data_root "${DATA_ROOT}" \
      --out_dir "${run_dir}" \
      --tech "${TECH}" \
      --loss "${LOSS}" \
      --warmup "${WARMUP}" \
      --training_epochs "${EPOCHS}" \
      --incr_epochs "${INCR_EPOCHS}" \
      --dump_step "${DUMP_STEP}" \
      --gpu_mem "${GPU_MEM}" \
      --use_ewc true \
      --ewc_lambda "${ewc_lambda}" \
      --ewc_gamma "${ewc_gamma}"
  done
}

run_rg_cfg() {
  local variant="$1"
  local replay_per_step="$2"
  local replay_beta="$3"
  local distill_alpha="$4"
  grid_id=$((grid_id + 1))
  local cfg_id
  cfg_id="$(printf "%03d" "${grid_id}")"
  local out_cfg="${OUT_ROOT}/replay_gem_${cfg_id}"
  mkdir -p "${out_cfg}"
  echo -e "${cfg_id}\treplay_gem\t${variant}\t-\t-\t${RG_FIXED_REPLAY_SIZE}\t${replay_per_step}\t${replay_beta}\t${distill_alpha}\t${RG_FIXED_TEMP}" >> "${PLAN_FILE}"

  echo "========== GRID ${cfg_id} [Replay+GEM/${variant}] =========="
  echo "model=${MODEL} replay_per_step=${replay_per_step} replay_beta=${replay_beta} distill_alpha=${distill_alpha} out=${out_cfg}"
  [[ "${DRY_RUN}" == "1" ]] && return 0

  for sub in "${SUBJECTS[@]}"; do
    local run_dir="${out_cfg}/${sub}"
    mkdir -p "${run_dir}"
    "${PYTHON}" continual_gem.py "${sub}" "${MODEL}" \
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
      --use_ewc false \
      --replay_size "${RG_FIXED_REPLAY_SIZE}" \
      --replay_per_step "${replay_per_step}" \
      --replay_beta "${replay_beta}" \
      --distill_alpha "${distill_alpha}" \
      --temperature "${RG_FIXED_TEMP}"
  done
}

# ---------- EWC-only: baseline + OAT ----------
run_ewc_cfg "baseline" "${EWC_BASE_LAMBDA}" "${EWC_BASE_GAMMA}"
for v in "${EWC_LAMBDA_GRID[@]}"; do
  [[ "${v}" == "${EWC_BASE_LAMBDA}" ]] && continue
  run_ewc_cfg "oat_ewc_lambda" "${v}" "${EWC_BASE_GAMMA}"
done
for v in "${EWC_GAMMA_GRID[@]}"; do
  [[ "${v}" == "${EWC_BASE_GAMMA}" ]] && continue
  run_ewc_cfg "oat_ewc_gamma" "${EWC_BASE_LAMBDA}" "${v}"
done

# ---------- Replay+GEM: baseline + OAT ----------
run_rg_cfg "baseline" "${RG_BASE_RPS}" "${RG_BASE_BETA}" "${RG_BASE_DA}"
for v in "${RG_RPS_GRID[@]}"; do
  [[ "${v}" == "${RG_BASE_RPS}" ]] && continue
  run_rg_cfg "oat_replay_per_step" "${v}" "${RG_BASE_BETA}" "${RG_BASE_DA}"
done
for v in "${RG_BETA_GRID[@]}"; do
  [[ "${v}" == "${RG_BASE_BETA}" ]] && continue
  run_rg_cfg "oat_replay_beta" "${RG_BASE_RPS}" "${v}" "${RG_BASE_DA}"
done
for v in "${RG_DA_GRID[@]}"; do
  [[ "${v}" == "${RG_BASE_DA}" ]] && continue
  run_rg_cfg "oat_distill_alpha" "${RG_BASE_RPS}" "${RG_BASE_BETA}" "${v}"
done

echo "All runs planned/executed. plan=${PLAN_FILE}"
