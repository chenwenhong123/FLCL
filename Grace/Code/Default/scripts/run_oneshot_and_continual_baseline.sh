#!/usr/bin/env bash
set -euo pipefail

# 依次跑完整 one-shot 与基础增量实验（六个 Defects4J 项目，参数可覆盖）。
#
# 用法：
#   bash Code/Default/scripts/run_oneshot_and_continual_baseline.sh
#
# 只跑其中一种：
#   RUN_ONESHOT=1 RUN_CONTINUAL=0 bash scripts/run_oneshot_and_continual_baseline.sh
#   RUN_ONESHOT=0 RUN_CONTINUAL=1 bash scripts/run_oneshot_and_continual_baseline.sh
#
# 先小项目试跑：
#   PROJECTS="Lang" bash scripts/run_oneshot_and_continual_baseline.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ONESHOT="${RUN_ONESHOT:-1}"
RUN_CONTINUAL="${RUN_CONTINUAL:-1}"

echo "==== One-shot + continual baseline ===="
echo "RUN_ONESHOT=${RUN_ONESHOT} RUN_CONTINUAL=${RUN_CONTINUAL}"
echo "PROJECTS=${PROJECTS:-Lang Chart Closure Math Mockito Time}"
echo

if [[ "${RUN_ONESHOT}" == "1" ]]; then
  bash "${SCRIPT_DIR}/run_oneshot_all_projects.sh"
fi

if [[ "${RUN_CONTINUAL}" == "1" ]]; then
  bash "${SCRIPT_DIR}/run_continual_baseline_all_projects.sh"
fi

echo "==== All requested experiments completed ===="
