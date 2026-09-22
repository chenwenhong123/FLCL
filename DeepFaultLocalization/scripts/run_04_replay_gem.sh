#!/usr/bin/env bash
# 重放 + DER++ + A-GEM（硬启用）：continual_gem.py
# 全量：全部项目 × 6 模型（与 continual_gem 支持列表一致）。

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
    echo "==== continual_gem REPLAY+DER++ + GEM subject=${sub} model=${model} ===="
    "${PYTHON}" continual_gem.py "${sub}" "${model}" \
      --data_root "${DATA_ROOT}" \
      --tech "${TECH}" \
      --loss softmax \
      --warmup 10 \
      --training_epochs 50 \
      --incr_epochs 1 \
      --use_replay true \
      --use_ewc false \
      --replay_size 20 \
      --replay_per_step 2 \
      --replay_beta 1.0 \
      --distill_alpha 0.5 \
      --temperature 2.0
  done
done

echo "run_04_replay_gem：全部任务结束。"
