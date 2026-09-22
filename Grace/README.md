# Grace：基于图的故障定位表示学习

Grace 是一种基于覆盖信息的故障定位方法，使用图结构表示代码与缺陷上下文。本仓库包含相关数据与实现代码。

## 数据集

预处理数据需要从 [Google Drive 链接](https://drive.google.com/drive/folders/1QH_Y9fKaNrwQCT6hvAH9-73PBQQ3a4hL?usp=sharing) 下载。请确保 `.pkl` 等数据文件路径与代码中的解析逻辑一致（本仓库 `Code/Default` 下脚本通常从 `Code/Default/GraceDate/` 自动解析数据，“Date”是引入时手快写错，但对实验影响不大故没修改）。

## 运行方式（原始批式流程）

主入口为 `Code/Default/runtotal.py`。在项目根目录或 `Code/Default` 下执行：

```bash
python runtotal.py <子项目名>
```

例如：`python runtotal.py Lang`。

`runtotal.py` 会依次调用同一目录下的 `run.py`、`sum.py`、`watch.py`：

- **run.py**：对项目中每个缺陷版本分别训练与评测。
- **sum.py**：汇总该项目所有缺陷版本的结果。
- **watch.py**：打印/查看汇总结果。
- **Model.py**：模型定义。
- **Dataset.py**：数据集与构图逻辑。

最终批式结果通常写入形如 `result_final_XXXX` 的目录；结果文件中第三行一般为 Top-1 命中数等指标（以具体输出格式为准）。

## 环境

参考环境（可按本机实际调整）：

```
PyTorch: 1.7.1 及以上（建议使用当前环境已安装版本）
OS: Ubuntu 16.04.6 LTS 或更新系统
更具体的包环境参考总目录中的environment.yml文件
```

---

## 增量学习实验（持续学习脚本）

以下脚本均在 **`Code/Default/`** 目录下运行；每次运行的输出写入 **`Code/Default/result/<Project>/<当次运行标识符>/`**，目录内为 `metrics.csv`、`model.pt`、`summary.txt`（离线 `run.py` 为 `*res*.pkl` 与 `summary.txt`）。运行标识符含策略前缀、关键超参和时间戳 `%m%d%H%M`。

### 公共参数含义（多数增量脚本共用）

| 位置参数 | 含义 |
|---------|------|
| `Project` | 项目名，如 `Lang`、`Chart`、`Closure`、`Math`、`Mockito`、`Time`。 |
| `lr` | 学习率。 |
| `seed` | 随机种子。 |
| `batch_size` | 批大小。 |
| `warmup_ids` | Warmup 阶段使用的缺陷 ID 个数 \(K\)；前 \(K\) 个 ID 用于选基模型与初始化。 |
| `base_epochs` | Warmup 阶段每个候选基模型上的训练轮数。 |
| `inc_epochs` | 流式增量阶段，每到达一个新任务时的训练轮数。 |
| `replay_size` | 回放缓冲区最多保留的历史任务 ID 数量（水库采样上界）。 |
| `replay_per_step` | 每个增量步从缓冲区中采样的回放任务数（与缓冲区长度取 min）。 |
| `replay_beta` | 回放监督项权重（与当前任务损失相加）。 |
| `distill_alpha` | 回放蒸馏（DER++ 风格 KL）项权重。 |
| `temperature` | 蒸馏 KL 中使用的温度系数。 |

### 各脚本命令行参数（按 `sys.argv` 顺序）

以下均在 `Code/Default` 下执行：`python <脚本名> ...`

#### 1. `continual_run.py` — 基础增量（无回放、无 EWC/GEM）

```
<Project> <lr> <seed> <batch_size> [warmup_ids=10] [base_epochs=15] [inc_epochs=2]
```

#### 2. `continual_run_with_replay.py` — Replay + DER++（蒸馏）

```
<Project> <lr> <seed> <batch_size>
[warmup_ids=10] [base_epochs=15] [inc_epochs=2]
[replay_size=20] [replay_per_step=2] [replay_beta=1.0] [distill_alpha=0.5] [temperature=2.0]
```

#### 3. `continual_run_ewc.py` — Replay + DER++ + EWC

在「公共回放参数」之后增加：

| 参数 | 含义 |
|------|------|
| `ewc_lambda` | EWC 惩罚项系数。 |
| `ewc_gamma` | Fisher 信息矩阵的指数滑动平均系数。 |
| `fisher_ids` | 估计 Fisher 时每个增量步采样的任务数（含当前任务 + 回放池中抽样）。 |

完整顺序：

```
<Project> <lr> <seed> <batch_size>
[warmup_ids=10] [base_epochs=15] [inc_epochs=2]
[replay_size=20] [replay_per_step=2] [replay_beta=1.0] [distill_alpha=0.5] [temperature=2.0]
[ewc_lambda=10.0] [ewc_gamma=0.9] [fisher_ids=4]
```

#### 4. `continual_run_gem.py` — Replay + DER++ + A-GEM

参数与 `continual_run_with_replay.py` 相同（无独立 `ewc_*` 命令行项；GEM 通过回放参考梯度投影实现）。

```
<Project> <lr> <seed> <batch_size>
[warmup_ids=10] [base_epochs=15] [inc_epochs=2]
[replay_size=20] [replay_per_step=2] [replay_beta=1.0] [distill_alpha=0.5] [temperature=2.0]
```

#### 5. `continual_run_all.py` — Replay + DER++ + EWC + A-GEM（全开）

```
<Project> <lr> <seed> <batch_size>
[warmup_ids=10] [base_epochs=15] [inc_epochs=2]
[replay_size=20] [replay_per_step=2] [replay_beta=1.0] [distill_alpha=0.5] [temperature=2.0]
[ewc_lambda=10.0] [ewc_gamma=0.9] [fisher_ids=4]
```

#### 6. `continual_run_mask.py` — 任务门控掩码 + Replay + DER++ + 可选 EWC/GEM

在 `continual_run_all.py` 同款参数之后增加：

| 参数 | 含义 |
|------|------|
| `use_ewc` | `1` 启用 EWC，`0` 关闭。 |
| `use_gem` | `1` 启用 GEM，`0` 关闭。 |

```
... [fisher_ids=4] [use_ewc=1] [use_gem=1]
```

#### 7. `continual_dropout.py`（或同逻辑的 `continual_drpout.py`）— Dropout 一致性正则 + Replay + DER++ + 可选 EWC/GEM

在 `continual_run_all` 参数基础上增加：

| 参数 | 含义 |
|------|------|
| `dropout_alpha` | 两次随机前向的 Dropout 一致性 KL 正则权重。 |
| `dropout_temperature` | 上述 KL 使用的温度。 |
| `use_ewc` | `1`/`0`，是否把 EWC 惩罚加在当前任务分支。 |
| `use_gem` | `1`/`0`，是否对当前梯度做 GEM 投影。 |

`continual_dropout.py` 支持仅传项目名时使用默认 `lr=0.01, seed=0, batch_size=60`（以脚本内 `Usage` 为准）。

#### 8. `continual_run_adam_l2.py` — 基础增量 + Adam + L2（weight decay）

```
<Project> <lr> <seed> <batch_size>
[warmup_ids=10] [base_epochs=15] [inc_epochs=2]
[weight_decay=1e-4] [beta1=0.9] [beta2=0.999]
```

#### 9. `continual_run_sgd_l2.py` — 基础增量 + SGD + 可选 L2

```
<Project> <lr> <seed> <batch_size>
[warmup_ids=10] [base_epochs=15] [inc_epochs=2]
[weight_decay=0.0] [momentum=0.9]
```

### 批量与网格脚本（`Code/Default/scripts/`）

Shell 脚本已统一放到 `Code/Default/scripts/`。脚本会自动切到 `Code/Default` 再调用 Python 入口，结果仍写入 `result/`、`logs/`、`grid_outputs/`。

在 `Code/Default` 下执行，例如：

```bash
bash scripts/continual_run_parameter_tunning.sh
```

- `scripts/run_oneshot_all_projects.sh`：原始 one-shot（每个缺陷单独 `run.py`，再 `sum.py` / `watch.py`）。
- `scripts/run_continual_baseline_all_projects.sh`：基础增量（仅 `continual_run.py`）。
- `scripts/run_basic_param_sweep_on_best.sh`：在 Replay-only 上对 warmup / base_epochs / inc_epochs 做一维 5 点扫描；`lr=0.01`、`batch_size=60` 固定。
- `scripts/run_base_epochs_extra_on_best.sh`：同上策略，补扫 `base_epochs={30,35,40,45,50}`。
- `scripts/run_heatmap2d_on_best.sh`：在 Replay-only 上做三张二维网格（各固定一个默认参数），写入 `result_heatmap2d/`。
- `scripts/continual_run_parameter_tunning.sh`：全项目主配置扫一遍基础增量 / replay / ewc / gem / all / mask。
- `scripts/run_ewc_only_all_projects.sh`：EWC-only（关闭 replay）全项目。
- `scripts/run_continual_dropout_all_cases.sh`：Dropout 四组 EWC/GEM 开关。
- `scripts/run_three_optimizer_l2_cases.sh`：SGD/Adam 与 L2 三种对照。
- `scripts/run_local_grid_replay_ewc_gem.sh`：Replay+EWC 与 Replay+GEM 局部网格；每次结果复制到 `Code/Default/grid_outputs/{ewc,gem}/`。
- `scripts/summarize_local_grid_results.sh`：从 `grid_outputs` 汇总生成 CSV。

---

## 结果目录约定（增量）

- 默认：`Code/Default/result/<Project>/<当次运行标识符>/`（`metrics.csv`、`model.pt`、`summary.txt`）
- `sum.py` 仍把合并后的 `*res_{seed}_{lr}_{bs}.pkl` 写在 `result/<Project>/` 根下，供 `watch.py` 使用
- 论文整理用归档：`Code/Default/GraceResult/`（若你手动拷贝或脚本写入）
