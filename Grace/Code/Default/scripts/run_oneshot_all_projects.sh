#!/usr/bin/env bash
set -euo pipefail

# 原始 Grace one-shot：对每个缺陷版本单独训练（run.py），再 sum.py / watch.py 汇总。
#
# 用法（在仓库任意位置均可）：
#   bash Code/Default/scripts/run_oneshot_all_projects.sh
#   bash scripts/run_oneshot_all_projects.sh          # 已在 Code/Default 下时
#
# 常用覆盖：
#   PROJECTS="Lang" bash scripts/run_oneshot_all_projects.sh
#   PROJECTS="Lang Chart" LR=0.01 SEED=0 BATCH_SIZE=60 bash scripts/run_oneshot_all_projects.sh
#   JOBS=2 SKIP_EXISTING=1 bash scripts/run_oneshot_all_projects.sh
#
# 环境变量：
#   PROJECTS       默认 Lang Chart Closure Math Mockito Time
#   LR             默认 0.01
#   SEED           默认 0
#   BATCH_SIZE     默认 60
#   JOBS           同时跑的 run.py 进程数，默认 1（顺序最稳；GPU 内存够可调大）
#   SKIP_EXISTING  1 则已有对应 pkl 时跳过该 bug_id，默认 0
#   PYTHON         默认 python

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_DIR}"

PROJECTS="${PROJECTS:-Lang Chart Closure Math Mockito Time}"
LR="${LR:-0.01}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-60}"
JOBS="${JOBS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
PYTHON="${PYTHON:-python}"

mkdir -p logs result

echo "==== One-shot (original Grace / run.py) ===="
echo "CODE_DIR=${CODE_DIR}"
echo "PROJECTS=${PROJECTS}"
echo "LR=${LR} SEED=${SEED} BATCH_SIZE=${BATCH_SIZE} JOBS=${JOBS} SKIP_EXISTING=${SKIP_EXISTING}"
echo

for f in run.py sum.py watch.py; do
  if [[ ! -f "${f}" ]]; then
    echo "Error: cannot find ${f} in ${CODE_DIR}"
    exit 1
  fi
done

pkl_path() {
  local project="$1"
  if [[ -f "${project}.pkl" ]]; then
    echo "${project}.pkl"
  else
    echo "GraceDate/${project}.pkl"
  fi
}

n_bugs() {
  local project="$1"
  local path
  path="$(pkl_path "${project}")"
  if [[ ! -f "${path}" ]]; then
    echo "Error: data file not found: ${path}" >&2
    exit 1
  fi
  "${PYTHON}" - "${path}" <<'PY'
import pickle, sys
print(len(pickle.load(open(sys.argv[1], "rb"))))
PY
}

result_pkl() {
  local project="$1" bug_id="$2"
  echo "result/${project}/${project}res${bug_id}_${SEED}_${LR}_${BATCH_SIZE}.pkl"
}

wait_for_slot() {
  local n
  while true; do
    n="$(jobs -pr | wc -l | tr -d ' ')"
    if (( n < JOBS )); then
      return 0
    fi
    sleep 2
  done
}

run_one_bug() {
  local project="$1" bug_id="$2" n="$3"
  local out tag log
  out="$(result_pkl "${project}" "${bug_id}")"
  tag="${project}_oneshot_id${bug_id}_seed${SEED}_lr${LR}_bs${BATCH_SIZE}"
  log="logs/${tag}.log"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${out}" ]]; then
    echo "    skip bug_id=${bug_id} (exists: ${out})"
    return 0
  fi
  echo "    [${project}] bug_id=${bug_id}/${n}  log=${log}"
  "${PYTHON}" run.py "${bug_id}" "${project}" "${LR}" "${SEED}" "${BATCH_SIZE}" \
    >"${log}" 2>&1
}

run_project() {
  local project="$1"
  local n i
  n="$(n_bugs "${project}")"
  echo ">>> [one-shot][${project}] n_bugs=${n}"

  if (( JOBS <= 1 )); then
    for ((i = 0; i < n; i++)); do
      run_one_bug "${project}" "${i}" "$((n - 1))"
    done
  else
    set +e
    fail=0
    pids=()
    for ((i = 0; i < n; i++)); do
      wait_for_slot
      run_one_bug "${project}" "${i}" "$((n - 1))" &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        fail=1
      fi
    done
    set -e
    if (( fail != 0 )); then
      echo "Error: one or more run.py jobs failed for ${project}"
      exit 1
    fi
  fi

  echo ">>> [one-shot][${project}] sum.py"
  "${PYTHON}" sum.py "${project}" "${SEED}" "${LR}" "${BATCH_SIZE}"
  echo ">>> [one-shot][${project}] watch.py"
  "${PYTHON}" watch.py "${project}" "${SEED}" "${LR}" "${BATCH_SIZE}"
  echo ">>> [one-shot][${project}] done  merged=result/${project}/${project}res_${SEED}_${LR}_${BATCH_SIZE}.pkl"
  echo
}

for project in ${PROJECTS}; do
  run_project "${project}"
done

echo "All one-shot runs completed."
echo "Per-bug pkl / summary: result/<Project>/"
echo "Merged pkl: result/<Project>/<Project>res_${SEED}_${LR}_${BATCH_SIZE}.pkl"
echo "Final table: result/<Project>/<Project>result_final_${SEED}_${LR}_${BATCH_SIZE}"
echo "Logs: logs/<Project>_oneshot_id*_seed${SEED}_lr${LR}_bs${BATCH_SIZE}.log"
