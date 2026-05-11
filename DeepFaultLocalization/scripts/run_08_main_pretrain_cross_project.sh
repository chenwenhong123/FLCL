#!/usr/bin/env bash
# 跨项目版本（LOPO）：每次留一个项目作为目标项目，其余项目用于全局预训练池。
#
# 对每个目标项目 target_sub：
#   1) 用其它 5 个项目在 ver=1..PRETRAIN_VMAX 预训练（每个 target_sub 单独 ckpt_dir）
#   2) 在 target_sub 的各版本做微调评估
#
# 默认目标版本范围：
# - 从 TARGET_START_VER 到该项目最大版本（默认 1）
#
# 用法示例：
#   bash scripts/run_08_main_pretrain_cross_project.sh
#   TARGET_START_VER=16 PRETRAIN_VMAX=15 bash scripts/run_08_main_pretrain_cross_project.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-.}"
OUT_DIR="${OUT_DIR:-${ROOT}/result_pretrain_cross_project}"
CKPT_ROOT="${CKPT_ROOT:-${ROOT}/pretrain_ckpts_cross_project}"
TECH="${TECH:-DeepFL_CL}"

PRETRAIN_VMAX="${PRETRAIN_VMAX:-15}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-50}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"
DUMP_STEP="${DUMP_STEP:-10}"
DISPLAY_STEP="${DISPLAY_STEP:-2}"
GPU_MEM="${GPU_MEM:-0.8}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-2048}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-2048}"
FORCE_PRETRAIN="${FORCE_PRETRAIN:-0}"
TARGET_START_VER="${TARGET_START_VER:-1}"

SUBJECTS=(Chart Lang Math Time Closure Mockito)
MAX_VERSIONS=(26 65 106 27 133 38)

for i in "${!SUBJECTS[@]}"; do
  target_sub="${SUBJECTS[$i]}"
  target_vmax="${MAX_VERSIONS[$i]}"

  # 构造预训练项目列表（排除当前 target_sub）
  train_subs=()
  for s in "${SUBJECTS[@]}"; do
    if [[ "${s}" != "${target_sub}" ]]; then
      train_subs+=("${s}")
    fi
  done
  train_subs_csv="$(IFS=,; echo "${train_subs[*]}")"

  ckpt_dir="${CKPT_ROOT}/${target_sub}"
  mkdir -p "${ckpt_dir}"

  start_ver="${TARGET_START_VER}"
  if (( start_ver < 1 )); then
    start_ver=1
  fi
  if (( start_ver > target_vmax )); then
    echo "skip target_sub=${target_sub}: start_ver=${start_ver} > vmax=${target_vmax}"
    continue
  fi

  echo "==== [cross-project] target=${target_sub}, pretrain_subjects=${train_subs_csv} ===="

  first_ver="${start_ver}"
  "${PYTHON}" main_pretrain.py \
    --data_root "${DATA_ROOT}" \
    --out_dir "${OUT_DIR}" \
    --ckpt_dir "${ckpt_dir}" \
    --tech "${TECH}" \
    --subjects "${train_subs_csv}" \
    --v_max "${PRETRAIN_VMAX}" \
    --pretrain_epochs "${PRETRAIN_EPOCHS}" \
    --finetune_epochs "${FINETUNE_EPOCHS}" \
    --dump_step "${DUMP_STEP}" \
    --display_step "${DISPLAY_STEP}" \
    --gpu_mem "${GPU_MEM}" \
    --pretrain_batch_size "${PRETRAIN_BATCH_SIZE}" \
    --finetune_batch_size "${FINETUNE_BATCH_SIZE}" \
    --target_sub "${target_sub}" \
    --target_ver "${first_ver}" \
    $( [[ "${FORCE_PRETRAIN}" == "1" ]] && echo "--force_pretrain" )

  for ((v=start_ver; v<=target_vmax; v++)); do
    # 第一条已跑过（用于触发预训练），避免重复。
    if (( v == first_ver )); then
      continue
    fi
    echo "---- [cross-project] target=${target_sub} ver=${v} ----"
    "${PYTHON}" main_pretrain.py \
      --data_root "${DATA_ROOT}" \
      --out_dir "${OUT_DIR}" \
      --ckpt_dir "${ckpt_dir}" \
      --tech "${TECH}" \
      --subjects "${train_subs_csv}" \
      --v_max "${PRETRAIN_VMAX}" \
      --pretrain_epochs "${PRETRAIN_EPOCHS}" \
      --finetune_epochs "${FINETUNE_EPOCHS}" \
      --dump_step "${DUMP_STEP}" \
      --display_step "${DISPLAY_STEP}" \
      --gpu_mem "${GPU_MEM}" \
      --pretrain_batch_size "${PRETRAIN_BATCH_SIZE}" \
      --finetune_batch_size "${FINETUNE_BATCH_SIZE}" \
      --target_sub "${target_sub}" \
      --target_ver "${v}"
  done
done

echo "==== [cross-project] per-target rank summary ===="
for i in "${!SUBJECTS[@]}"; do
  target_sub="${SUBJECTS[$i]}"
  if [[ ! -d "${OUT_DIR}/${target_sub}" ]]; then
    echo "skip rank summary for ${target_sub}: no output dir ${OUT_DIR}/${target_sub}"
    continue
  fi
  echo "---- rank summary target=${target_sub} ----"
  "${PYTHON}" rank_parser.py \
    "${DATA_ROOT}" \
    "${OUT_DIR}" \
    "${TECH}" \
    "main_pretrain" \
    "softmax" \
    "${FINETUNE_EPOCHS}" \
    "${target_sub}"
done

echo "run_08_main_pretrain_cross_project：全部任务结束。"
