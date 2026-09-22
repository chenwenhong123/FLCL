#!/usr/bin/env bash
set -euo pipefail

# 二维热力图实验（Replay-only，与一维扫描同一套取值）。
# 每次固定一个默认参数，另外两个做全组合。结果写入 result_heatmap2d/。
#
# 网格（与 run_basic_param_sweep_on_best.sh + run_base_epochs_extra_on_best.sh 一致）：
#   warmup_ids:  {6, 8, 10, 12, 14}           默认 10
#   base_epochs: {5,10,15,20,25,30,35,40,45,50}  默认 15
#   inc_epochs:  {1, 2, 3, 4, 5}              默认 2
#
# 三张热力图：
#   fix_warmup  固定 warm=10     →  10×5 = 50 格  (base × inc)
#   fix_base    固定 base=15     →   5×5 = 25 格  (warm × inc)
#   fix_inc     固定 inc=2       →  5×10 = 50 格  (warm × base)
# 去重后每项目 106 个 unique (w,b,i)；一维已跑过的格子默认从 result/ 拷贝，不重训。
#
# 用法：
#   bash scripts/run_heatmap2d_on_best.sh
#   PROJECTS="Lang" CUDA_VISIBLE_DEVICES=1 bash scripts/run_heatmap2d_on_best.sh
#   HEATMAPS="fix_warmup" bash scripts/run_heatmap2d_on_best.sh
#   COPY_1D=0 bash scripts/run_heatmap2d_on_best.sh   # 全部重跑进新目录

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_DIR}"

PROJECTS="${PROJECTS:-Lang Chart Closure Math Mockito Time}"
PYTHON="${PYTHON:-python}"
SCRIPT="${SCRIPT:-continual_run_with_replay.py}"

SEED="${SEED:-0}"
LR="${LR:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-60}"
CENTER_WARM="${CENTER_WARM:-10}"
CENTER_BASE="${CENTER_BASE:-15}"
CENTER_INC="${CENTER_INC:-2}"

WARMUP_LIST="${WARMUP_LIST:-6 8 10 12 14}"
BASE_EPOCHS_LIST="${BASE_EPOCHS_LIST:-5 10 15 20 25 30 35 40 45 50}"
INC_EPOCHS_LIST="${INC_EPOCHS_LIST:-1 2 3 4 5}"

REPLAY_SIZE="${REPLAY_SIZE:-20}"
REPLAY_PER_STEP="${REPLAY_PER_STEP:-2}"
REPLAY_BETA="${REPLAY_BETA:-1.0}"
DISTILL_ALPHA="${DISTILL_ALPHA:-0.0}"
TEMPERATURE="${TEMPERATURE:-2.0}"

HEATMAPS="${HEATMAPS:-fix_warmup fix_base fix_inc}"
SRC_RESULT="${SRC_RESULT:-${CODE_DIR}/result}"
OUT_RESULT="${OUT_RESULT:-${CODE_DIR}/result_heatmap2d}"
COPY_1D="${COPY_1D:-1}"

export GRACE_RESULT_ROOT="${OUT_RESULT}"

mkdir -p logs "${OUT_RESULT}"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "Error: cannot find ${SCRIPT} in ${CODE_DIR}"
  exit 1
fi

heatmap_enabled() {
  local name="$1"
  for h in ${HEATMAPS}; do
    if [[ "${h}" == "${name}" ]]; then
      return 0
    fi
  done
  return 1
}

run_id_glob() {
  local warm="$1" base="$2" inc="$3"
  echo "replay_base${base}_inc${inc}_lr${LR}_bs${BATCH_SIZE}_warm${warm}_rs${REPLAY_SIZE}_rps${REPLAY_PER_STEP}_rb${REPLAY_BETA}_da${DISTILL_ALPHA}_ts*"
}

latest_src_dir() {
  local project="$1" warm="$2" base="$3" inc="$4"
  local g
  g="$(run_id_glob "${warm}" "${base}" "${inc}")"
  # shellcheck disable=SC2086
  ls -dt "${SRC_RESULT}/${project}"/${g} 2>/dev/null | head -n 1 || true
}

already_in_out() {
  local project="$1" warm="$2" base="$3" inc="$4"
  local g
  g="$(run_id_glob "${warm}" "${base}" "${inc}")"
  # shellcheck disable=SC2086
  ls -d "${OUT_RESULT}/${project}"/${g} >/dev/null 2>&1
}

copy_or_run() {
  local project="$1" warm="$2" base="$3" inc="$4" plane="$5"
  mkdir -p "${OUT_RESULT}/${project}"
  if already_in_out "${project}" "${warm}" "${base}" "${inc}"; then
    echo "    skip exists  ${project} warm=${warm} base=${base} inc=${inc}  [${plane}]"
    return 0
  fi
  if [[ "${COPY_1D}" == "1" ]]; then
    local src
    src="$(latest_src_dir "${project}" "${warm}" "${base}" "${inc}")"
    if [[ -n "${src}" && -d "${src}" ]]; then
      local dest="${OUT_RESULT}/${project}/$(basename "${src}")"
      echo "    copy 1D      ${src} -> ${dest}  [${plane}]"
      cp -a "${src}" "${dest}"
      return 0
    fi
  fi
  local tag="${project}_heatmap2d_${plane}_warm${warm}_base${base}_inc${inc}"
  local log="logs/${tag}.log"
  echo ">>> train ${tag}"
  echo "    log=${log}"
  echo "    GRACE_RESULT_ROOT=${GRACE_RESULT_ROOT}"
  "${PYTHON}" "${SCRIPT}" \
    "${project}" "${LR}" "${SEED}" "${BATCH_SIZE}" \
    "${warm}" "${base}" "${inc}" \
    "${REPLAY_SIZE}" "${REPLAY_PER_STEP}" "${REPLAY_BETA}" "${DISTILL_ALPHA}" "${TEMPERATURE}" \
    | tee "${log}"
  echo ">>> ${tag} done"
  echo
}

echo "==== Heatmap 2D grids on best strategy (replay-only) ===="
echo "CODE_DIR=${CODE_DIR}"
echo "OUT_RESULT=${OUT_RESULT}"
echo "PROJECTS=${PROJECTS}"
echo "HEATMAPS=${HEATMAPS}"
echo "COPY_1D=${COPY_1D}  SRC_RESULT=${SRC_RESULT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "fixed: lr=${LR} bs=${BATCH_SIZE} seed=${SEED}  replay rs=${REPLAY_SIZE} rps=${REPLAY_PER_STEP} da=${DISTILL_ALPHA}"
echo "warmup: ${WARMUP_LIST}"
echo "base:   ${BASE_EPOCHS_LIST}"
echo "inc:    ${INC_EPOCHS_LIST}"
echo

manifest="${OUT_RESULT}/combo_manifest.tsv"
{
  echo -e "heatmap\tproject\twarmup_ids\tbase_epochs\tinc_epochs"
  for project in ${PROJECTS}; do
    if heatmap_enabled fix_warmup; then
      for base in ${BASE_EPOCHS_LIST}; do
        for inc in ${INC_EPOCHS_LIST}; do
          echo -e "fix_warmup\t${project}\t${CENTER_WARM}\t${base}\t${inc}"
        done
      done
    fi
    if heatmap_enabled fix_base; then
      for warm in ${WARMUP_LIST}; do
        for inc in ${INC_EPOCHS_LIST}; do
          echo -e "fix_base\t${project}\t${warm}\t${CENTER_BASE}\t${inc}"
        done
      done
    fi
    if heatmap_enabled fix_inc; then
      for warm in ${WARMUP_LIST}; do
        for base in ${BASE_EPOCHS_LIST}; do
          echo -e "fix_inc\t${project}\t${warm}\t${base}\t${CENTER_INC}"
        done
      done
    fi
  done
} >"${manifest}"

echo "Wrote ${manifest}"
echo

seen_keys=""
key_seen() {
  local k="$1"
  case " ${seen_keys} " in
    *" ${k} "*) return 0 ;;
    *) return 1 ;;
  esac
}

n_train_slots=0
for project in ${PROJECTS}; do
  echo "===== ${project} ====="
  seen_keys=""
  if heatmap_enabled fix_warmup; then
    echo "--- heatmap fix_warmup (warm=${CENTER_WARM}): base × inc ---"
    for base in ${BASE_EPOCHS_LIST}; do
      for inc in ${INC_EPOCHS_LIST}; do
        k="${project}|${CENTER_WARM}|${base}|${inc}"
        if key_seen "${k}"; then
          continue
        fi
        seen_keys="${seen_keys} ${k}"
        copy_or_run "${project}" "${CENTER_WARM}" "${base}" "${inc}" "fix_warmup"
        n_train_slots=$((n_train_slots + 1))
      done
    done
  fi
  if heatmap_enabled fix_base; then
    echo "--- heatmap fix_base (base=${CENTER_BASE}): warmup × inc ---"
    for warm in ${WARMUP_LIST}; do
      for inc in ${INC_EPOCHS_LIST}; do
        k="${project}|${warm}|${CENTER_BASE}|${inc}"
        if key_seen "${k}"; then
          continue
        fi
        seen_keys="${seen_keys} ${k}"
        copy_or_run "${project}" "${warm}" "${CENTER_BASE}" "${inc}" "fix_base"
        n_train_slots=$((n_train_slots + 1))
      done
    done
  fi
  if heatmap_enabled fix_inc; then
    echo "--- heatmap fix_inc (inc=${CENTER_INC}): warmup × base ---"
    for warm in ${WARMUP_LIST}; do
      for base in ${BASE_EPOCHS_LIST}; do
        k="${project}|${warm}|${base}|${CENTER_INC}"
        if key_seen "${k}"; then
          continue
        fi
        seen_keys="${seen_keys} ${k}"
        copy_or_run "${project}" "${warm}" "${base}" "${CENTER_INC}" "fix_inc"
        n_train_slots=$((n_train_slots + 1))
      done
    done
  fi
done

echo "==== Heatmap 2D finished (unique slots looped: ${n_train_slots}) ===="
echo "Results: ${OUT_RESULT}/<Project>/replay_base*_inc*_warm*_da${DISTILL_ALPHA}_ts*/"
echo "Manifest: ${manifest}"
echo "Logs: logs/<Project>_heatmap2d_*.log"
