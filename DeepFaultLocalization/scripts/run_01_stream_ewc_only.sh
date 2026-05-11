#!/usr/bin/env bash
# 仅 EWC：continual_stream_train.py（--use_ewc true）
# 全量：全部 Defects4J 项目 × continual_stream_train 支持的 6 个模型。
# 不传 --v_end，由各项目默认最大版本（Lang 等跑满）。

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
    echo "==== continual_stream_train EWC-only subject=${sub} model=${model} ===="
    "${PYTHON}" continual_stream_train.py "${sub}" "${model}" \
      --data_root "${DATA_ROOT}" \
      --tech "${TECH}" \
      --loss softmax \
      --warmup 10 \
      --training_epochs 50 \
      --incr_epochs 1 \
      --use_ewc true \
      --ewc_lambda 10.0 \
      --ewc_gamma 0.9
  done
done

echo "run_01_stream_ewc_only：全部任务结束。"
