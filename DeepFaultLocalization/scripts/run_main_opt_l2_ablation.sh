#!/usr/bin/env bash
# 四组 optimizer×L2 消融：走 continual_stream_train.py 普通流式（非 main 逐 bug 全量训练，显著更快）。
# 1) sgd + 无 L2   2) adam + 无 L2   3) sgd + L2   4) adam + L2
#
# 输出根：result_stream_opt_l2_ablation/<setting>/<model>/<subject>/
# 每组 run 的指标见 {mmddHHMM}_continual_metrics.txt。
# 注意：rank_parser 依赖 main 式路径 <out>/<sub>/<v>/<tech>/<model>-<loss>-<epoch>，与流式目录不兼容，此处不调用。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-.}"
LOSS="${LOSS:-softmax}"
EPOCHS="${EPOCHS:-50}"
DUMP_STEP="${DUMP_STEP:-10}"
WARMUP="${WARMUP:-10}"
INCR_EPOCHS="${INCR_EPOCHS:-1}"
TECH="${TECH:-DeepFL_CL}"

MODELS=(mlp mlp2 mlp_dfl_1 mlp_dfl_2 rnn birnn)
SUBJECTS=(Chart Lang Math Time Closure Mockito)

SET_NAMES=(noadam_nol2 adam_nol2 noadam_l2 adam_l2)
SET_OPTIMIZERS=(sgd adam sgd adam)
SET_USE_L2=(false false true true)

run_one_setting() {
  local setting_name="$1"
  local optimizer_name="$2"
  local use_l2="$3"
  local out_root="${ROOT}/result_stream_opt_l2_ablation/${setting_name}"

  mkdir -p "${out_root}"
  echo "============================================================"
  echo "Start setting=${setting_name} optimizer=${optimizer_name} use_l2=${use_l2} (continual_stream_train)"
  echo "Output root: ${out_root}"
  echo "============================================================"

  for model in "${MODELS[@]}"; do
    for sub in "${SUBJECTS[@]}"; do
      local run_dir="${out_root}/${model}/${sub}"
      mkdir -p "${run_dir}"
      echo "---- [${setting_name}] sub=${sub} model=${model} -> ${run_dir}"
      "${PYTHON}" continual_stream_train.py "${sub}" "${model}" \
        --data_root "${DATA_ROOT}" \
        --out_dir "${run_dir}" \
        --tech "${TECH}" \
        --loss "${LOSS}" \
        --warmup "${WARMUP}" \
        --training_epochs "${EPOCHS}" \
        --incr_epochs "${INCR_EPOCHS}" \
        --dump_step "${DUMP_STEP}" \
        --use_ewc false \
        --optimizer "${optimizer_name}" \
        --use_l2 "${use_l2}"
    done
  done

  echo "Finished setting=${setting_name}"
}

for idx in "${!SET_NAMES[@]}"; do
  run_one_setting "${SET_NAMES[$idx]}" "${SET_OPTIMIZERS[$idx]}" "${SET_USE_L2[$idx]}"
done

echo "All settings finished."
