#!/usr/bin/env bash
# 全项目：先构建/复用一次全局预训练 ckpt，再对各项目各版本做微调评估。
#
# 默认策略：
# - 预训练池：6 项目 × ver=1..PRETRAIN_VMAX（默认 15）
# - 微调评估：每项目从 TARGET_START_VER 到该项目最大版本
#   （默认 TARGET_START_VER=PRETRAIN_VMAX+1，避免与预训练池版本重叠）
#
# 用法示例：
#   bash scripts/run_07_main_pretrain_fullstack.sh
#   TARGET_START_VER=1 bash scripts/run_07_main_pretrain_fullstack.sh
#   FORCE_PRETRAIN=1 PRETRAIN_EPOCHS=80 bash scripts/run_07_main_pretrain_fullstack.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-.}"
OUT_DIR="${OUT_DIR:-${ROOT}/result_pretrain_fullstack}"
CKPT_DIR="${CKPT_DIR:-${ROOT}/pretrain_ckpts_fullstack}"
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
RANK_SUB="${RANK_SUB:-all}"

TARGET_START_VER="${TARGET_START_VER:-$((PRETRAIN_VMAX + 1))}"

SUBJECTS=(Chart Lang Math Time Closure Mockito)
MAX_VERSIONS=(26 65 106 27 133 38)
SUBJECTS_CSV="$(IFS=,; echo "${SUBJECTS[*]}")"

echo "==== [main_pretrain fullstack] build/reuse global pretrain ckpt ===="
first_sub="${SUBJECTS[0]}"
"${PYTHON}" main_pretrain.py \
  --data_root "${DATA_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --ckpt_dir "${CKPT_DIR}" \
  --tech "${TECH}" \
  --subjects "${SUBJECTS_CSV}" \
  --v_max "${PRETRAIN_VMAX}" \
  --pretrain_epochs "${PRETRAIN_EPOCHS}" \
  --finetune_epochs "${FINETUNE_EPOCHS}" \
  --dump_step "${DUMP_STEP}" \
  --display_step "${DISPLAY_STEP}" \
  --gpu_mem "${GPU_MEM}" \
  --pretrain_batch_size "${PRETRAIN_BATCH_SIZE}" \
  --finetune_batch_size "${FINETUNE_BATCH_SIZE}" \
  --target_sub "${first_sub}" \
  --target_ver "${TARGET_START_VER}" \
  $( [[ "${FORCE_PRETRAIN}" == "1" ]] && echo "--force_pretrain" )

echo "==== [main_pretrain fullstack] run all subjects/versions ===="
for i in "${!SUBJECTS[@]}"; do
  sub="${SUBJECTS[$i]}"
  vmax="${MAX_VERSIONS[$i]}"

  start_ver="${TARGET_START_VER}"
  if (( start_ver < 1 )); then
    start_ver=1
  fi
  if (( start_ver > vmax )); then
    echo "skip subject=${sub}: start_ver=${start_ver} > vmax=${vmax}"
    continue
  fi

  for ((v=start_ver; v<=vmax; v++)); do
    # 第一条已在上面跑过（用于触发预训练），避免重复。
    if [[ "${sub}" == "${first_sub}" && "${v}" -eq "${TARGET_START_VER}" ]]; then
      continue
    fi

    echo "---- finetune subject=${sub} ver=${v} ----"
    "${PYTHON}" main_pretrain.py \
      --data_root "${DATA_ROOT}" \
      --out_dir "${OUT_DIR}" \
      --ckpt_dir "${CKPT_DIR}" \
      --tech "${TECH}" \
      --subjects "${SUBJECTS_CSV}" \
      --v_max "${PRETRAIN_VMAX}" \
      --pretrain_epochs "${PRETRAIN_EPOCHS}" \
      --finetune_epochs "${FINETUNE_EPOCHS}" \
      --dump_step "${DUMP_STEP}" \
      --display_step "${DISPLAY_STEP}" \
      --gpu_mem "${GPU_MEM}" \
      --pretrain_batch_size "${PRETRAIN_BATCH_SIZE}" \
      --finetune_batch_size "${FINETUNE_BATCH_SIZE}" \
      --target_sub "${sub}" \
      --target_ver "${v}"
  done
done

echo "==== [main_pretrain fullstack] rank summary ===="
"${PYTHON}" rank_parser.py \
  "${DATA_ROOT}" \
  "${OUT_DIR}" \
  "${TECH}" \
  "main_pretrain" \
  "softmax" \
  "${FINETUNE_EPOCHS}" \
  "${RANK_SUB}"

echo "run_07_main_pretrain_fullstack：全部任务结束。"
