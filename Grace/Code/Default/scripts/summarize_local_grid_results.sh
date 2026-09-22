#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_DIR}"

# 汇总 run_local_grid_replay_ewc_gem.sh 的归档结果
# 输入：grid_outputs/{ewc,gem}/*_summary.txt
# 输出：
#   grid_outputs/grid_summary_all.csv
#   grid_outputs/grid_summary_best_by_project_strategy.csv
#   grid_outputs/grid_summary_best_macro_by_strategy.csv
#
# 用法：
#   bash scripts/summarize_local_grid_results.sh

BASE_DIR="${BASE_DIR:-grid_outputs}"
OUT_ALL="${OUT_ALL:-${BASE_DIR}/grid_summary_all.csv}"
OUT_BEST_PROJ="${OUT_BEST_PROJ:-${BASE_DIR}/grid_summary_best_by_project_strategy.csv}"
OUT_BEST_MACRO="${OUT_BEST_MACRO:-${BASE_DIR}/grid_summary_best_macro_by_strategy.csv}"

mkdir -p "${BASE_DIR}"

python - <<'PY'
from pathlib import Path
import csv
import re
from statistics import mean

base = Path("grid_outputs")
all_rows = []

def parse_summary(path: Path):
    d = {}
    for line in path.read_text(errors="ignore").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    return d

for strategy in ["ewc", "gem"]:
    sdir = base / strategy
    if not sdir.exists():
        continue
    for f in sorted(sdir.glob("*_summary.txt")):
        # ewc: <project>_ewc_rpsX_elY_rbZ_summary.txt
        # gem: <project>_gem_rpsX_rbY_daZ_summary.txt
        stem = f.stem
        m = re.match(r"^(?P<project>[A-Za-z]+)_(?P<strategy>ewc|gem)_(?P<params>.+)_summary$", stem)
        if not m:
            continue
        info = parse_summary(f)
        row = {
            "project": m.group("project"),
            "strategy": m.group("strategy"),
            "tag_params": m.group("params"),
            "summary_file": str(f),
            "replay_size": info.get("replay_size", ""),
            "replay_per_step": info.get("replay_per_step", ""),
            "replay_beta": info.get("replay_beta", ""),
            "distill_alpha": info.get("distill_alpha", ""),
            "temperature": info.get("temperature", ""),
            "ewc_lambda": info.get("ewc_lambda", ""),
            "ewc_gamma": info.get("ewc_gamma", ""),
            "fisher_ids": info.get("fisher_ids", ""),
            "final_top1": float(info.get("final_top1", "nan")),
            "final_top3": float(info.get("final_top3", "nan")),
            "final_top5": float(info.get("final_top5", "nan")),
            "final_top1_count": float(info.get("final_top1_count", "nan")),
            "final_top3_count": float(info.get("final_top3_count", "nan")),
            "final_top5_count": float(info.get("final_top5_count", "nan")),
            "final_n": float(info.get("final_n", "nan")),
            "final_mfr": float(info.get("final_mfr", "nan")),
            "final_mar": float(info.get("final_mar", "nan")),
            "final_bwt_t": float(info.get("final_bwt_t", "nan")),
        }
        all_rows.append(row)

if not all_rows:
    raise SystemExit("No summary files found under grid_outputs/{ewc,gem}.")

all_rows.sort(key=lambda r: (r["strategy"], r["project"], -r["final_top1"], -r["final_top3"], r["final_mfr"]))

all_path = Path("grid_outputs/grid_summary_all.csv")
with all_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    w.writeheader()
    w.writerows(all_rows)

# best per (project, strategy)
best_proj = {}
for r in all_rows:
    k = (r["project"], r["strategy"])
    if k not in best_proj:
        best_proj[k] = r

best_proj_rows = list(best_proj.values())
best_proj_path = Path("grid_outputs/grid_summary_best_by_project_strategy.csv")
with best_proj_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(best_proj_rows[0].keys()))
    w.writeheader()
    w.writerows(best_proj_rows)

# best macro per strategy: first macro-average each tag across projects
by_strategy_tag = {}
for r in all_rows:
    k = (r["strategy"], r["tag_params"])
    by_strategy_tag.setdefault(k, []).append(r)

macro_rows = []
for (strategy, tag), rows in by_strategy_tag.items():
    # require same tag appears in all projects for fair macro
    projects = {x["project"] for x in rows}
    macro_rows.append({
        "strategy": strategy,
        "tag_params": tag,
        "project_count": len(projects),
        "macro_top1": mean([x["final_top1"] for x in rows]),
        "macro_top3": mean([x["final_top3"] for x in rows]),
        "macro_top5": mean([x["final_top5"] for x in rows]),
        "macro_top1_count": mean([x["final_top1_count"] for x in rows]),
        "macro_top3_count": mean([x["final_top3_count"] for x in rows]),
        "macro_top5_count": mean([x["final_top5_count"] for x in rows]),
        "macro_n": mean([x["final_n"] for x in rows]),
        "macro_mfr": mean([x["final_mfr"] for x in rows]),
        "macro_mar": mean([x["final_mar"] for x in rows]),
        "macro_bwt_t": mean([x["final_bwt_t"] for x in rows]),
    })

macro_rows.sort(key=lambda r: (r["strategy"], -r["macro_top1"], -r["macro_top3"], r["macro_mfr"]))

best_macro = {}
for r in macro_rows:
    s = r["strategy"]
    if s not in best_macro:
        best_macro[s] = r

best_macro_rows = list(best_macro.values())
best_macro_path = Path("grid_outputs/grid_summary_best_macro_by_strategy.csv")
with best_macro_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(best_macro_rows[0].keys()))
    w.writeheader()
    w.writerows(best_macro_rows)

print(f"Saved: {all_path}")
print(f"Saved: {best_proj_path}")
print(f"Saved: {best_macro_path}")
print(f"Total rows: {len(all_rows)}")
PY
