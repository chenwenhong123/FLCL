#!/usr/bin/env bash
# EWC + 重放 + DER++：continual_replay.py（--use_replay true --use_ewc true）
# 全量：全部项目 × 6 模型。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-.}"
TECH="${TECH:-DeepFL_CL}"

SUBJECTS=(Chart Lang Math Time Closure Mockito)
MODELS=(mlp mlp2 mlp_dfl_1 mlp_dfl_2 rnn birnn)

for sub in "${SUBJECTS[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "==== continual_replay REPLAY+DER++ + EWC subject=${sub} model=${model} ===="
    "${PYTHON}" continual_replay.py "${sub}" "${model}" \
      --data_root "${DATA_ROOT}" \
      --tech "${TECH}" \
      --loss softmax \
      --warmup 10 \
      --training_epochs 50 \
      --incr_epochs 1 \
      --use_replay true \
      --use_ewc true \
      --replay_size 20 \
      --replay_per_step 2 \
      --replay_beta 1.0 \
      --distill_alpha 0.5 \
      --temperature 2.0 \
      --ewc_lambda 10.0 \
      --ewc_gamma 0.9
  done
done

echo "run_03_replay_der_ewc：全部任务结束。"
