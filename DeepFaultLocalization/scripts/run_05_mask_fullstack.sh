#!/usr/bin/env bash
# Mask 全开栈：Replay + DER++ + EWC + A-GEM + Mask（continual_mask.py）
# 全量：全部 Defects4J 项目 × 三个模型（mlp_dfl_1 / mlp2 / birnn）。
# 不传 --v_end，由各项目默认最大版本。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-.}"
TECH="${TECH:-DeepFL_CL}"

SUBJECTS=(Chart Lang Math Time Closure Mockito)
MODELS=(mlp_dfl_1 mlp2 birnn)

for sub in "${SUBJECTS[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "==== continual_mask FULLSTACK subject=${sub} model=${model} ===="
    "${PYTHON}" continual_mask.py "${sub}" "${model}" \
      --data_root "${DATA_ROOT}" \
      --tech "${TECH}" \
      --loss softmax \
      --warmup 10 \
      --training_epochs 50 \
      --incr_epochs 1 \
      --use_replay true \
      --use_ewc true \
      --use_gem true \
      --use_mask true \
      --replay_size 20 \
      --replay_per_step 2 \
      --replay_beta 1.0 \
      --distill_alpha 0.5 \
      --temperature 2.0 \
      --ewc_lambda 10.0 \
      --ewc_gamma 0.9 \
      --mask_keep_ratio 0.6 \
      --mask_ema 0.9
  done
done

echo "run_05_mask_fullstack：全部任务结束。"
