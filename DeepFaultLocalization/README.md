# DeepFL

DeepFL 是一种基于深度学习的故障定位方法。本项目使用 [TensorFlow](https://www.tensorflow.org/) 实现了多种网络：单层 MLP（`mlp`）、双层 MLP（`mlp2`）、双向循环网络（`birnn`），以及两种面向缺陷定位定制的 MLP 变体（`mlp_dfl_1`、`mlp_dfl_2`）。基准数据来自 [Defects4J](https://github.com/rjust/defects4j)：开源仓库中提供多个项目的若干 buggy 版本及对应修复版本。特征维度包括谱系、变异、代码度量、文本相似度等多源信息（具体组合取决于所选 `tech` 数据集）。

## 环境依赖

- Python 2/3，需安装 Pandas、NumPy  
- TensorFlow（本项目使用 `tensorflow.compat.v1` 风格 API）
- 更详细的包内容请参考总目录上的environment.yml

## 数据集

数据集可从网盘下载：[Mega 云盘](https://mega.nz/#F!7rhCwQzT!OXUlRaIh-kyevSr6sTdwxA)。共 7 个 `.gz` 压缩包，对应DeepFL论文中的不同设置：

| 文件名 | 说明 |
|--------|------|
| `DeepFL.tar.gz` | 四维特征齐全 |
| `CrossDeepFL.tar.gz` | 跨项目场景下的四维特征 |
| `CrossValidation.tar.gz` | 用于 10 折交叉验证的四维特征 |
| `DeepFL-Spectrum.tar.gz` | 三维：变异、度量、文本相似度 |
| `DeepFL-Mutation.tar.gz` | 三维：谱系、度量、文本相似度 |
| `DeepFL-Metrics.tar.gz` | 三维：谱系、变异、文本相似度 |
| `DeepFL-Textual.tar.gz` | 三维：谱系、变异、度量 |

解压后将数据放到**自定义父目录**下，该路径将作为下文 `main.py` 与增量脚本的 `data_root` 使用（本文路径为DeepFaultLocalization/DeepFL）。

## 构建 DeepFL_CL 数据集（持续学习 / 流式专用）

流式脚本要求使用 **`DeepFL_CL`** 技术子目录（见下文「增量学习与持续学习」）。若你仅有批次训练用的 **`DeepFL`** 数据，可用脚本从 `DeepFL` 生成 **`DeepFL_CL`**，无需单独下载另一套压缩包。

**脚本**：`scripts/build_deepfl_cl_dataset.py`

**作用简述**：

- 对每个缺陷版本 `v`，读取源目录 `DeepFL/<subject>/<v>/` 下的 **`Test.csv` / `TestLabel.csv`**（原始测试候选与标签）；
- 写入目标目录 `DeepFL_CL/<subject>/<v>/`，并**同时**复制为 **`Train`/`TrainLabel` 与 `Test`/`TestLabel`**（内容相同），以匹配 continual 脚本的「先在该版本 Test 上评估、再用 Train 做本步增量」的 prequential 协议，且**不把其它版本的 Train 混入当前版本**，避免用未来信息训练；
- 自动生成 `DeepFL_CL/groupfile/<subject>/<v>/traidata.txt.group`（单组，组大小为样本行数）。

**命令示例**：

```bash
cd DeepFaultLocalization
python scripts/build_deepfl_cl_dataset.py --data_root /path/to/your/data
```

| 参数 | 含义 | 默认 |
|------|------|------|
| `--data_root` | 数据根目录，其下需已有 `DeepFL/<subject>/<ver>/Test.csv` 等 | `.` |
| `--src_tech` | 源技术子目录名 | `DeepFL` |
| `--dst_tech` | 生成的目标子目录名 | `DeepFL_CL` |

脚本内置与 `utils.get_max_ver` 一致的各项目最大版本号；若某版本缺少 `Test.csv` / `TestLabel.csv` 则跳过。完成后终端会打印 `converted_versions` 与 `dst=` 路径。生成完成后，将各 `continual_*.py` 的 `--data_root` 指到同一父目录，并设 `--tech DeepFL_CL` 即可。

## 批次训练（`main.py`）

进入项目根目录：

```bash
cd DeepFaultLocalization
```

对每个缺陷版本执行一次训练（留一风格由你在外层循环组织；本仓库以单版本单次调用为主）：

```bash
python main.py <数据父目录绝对路径> <结果输出目录绝对路径> <subject> <version> <model> <tech> <loss> <epoch> <dump_step> [optimizer_name] [use_l2]
```

### 参数说明

| 参数 | 含义 |
|------|------|
| 数据父目录 | 包含各 `tech` 子目录的数据根路径，例如 `/home/DeepLearningData`，其下应有 `DeepFL/Chart/1/...` 等结构 |
| 结果输出目录 | 排名与中间结果写入路径 |
| `subject` | 项目名：`Time`、`Chart`、`Lang`、`Math`、`Mockito`、`Closure` |
| `version` | 缺陷版本号。各项目最大版本数依次为：Time 27、Chart 26、Lang 65、Math 106、Mockito 38、Closure 133 |
| `model` | `mlp`、`mlp2`、`birnn`、`mlp_dfl_1`、`mlp_dfl_2` |
| `tech` | 特征/数据子目录名：`DeepFL`、`DeepFL-Metrics`、`DeepFL-Mutation`、`DeepFL-Spectrum`、`DeepFL-Textual`、`CrossDeepFL` 等 |
| `loss` | 损失：`softmax`、`epairwise` 等（与 `config.losses` 一致） |
| `epoch` | 训练轮数 |
| `dump_step` | 每隔多少 epoch 将结果写入结果文件（例如 10 表示在 10、20、30… 轮落盘） |
| `optimizer_name`（可选，第 10 个） | `adam` 或 `sgd`，默认 `adam` |
| `use_l2`（可选，第 11 个） | 是否启用 L2 正则：`true`/`false` 等，默认 `true` |

**CrossValidation 说明**：数据已混合并划分为 10 折。使用时将 `subject` 设为 `10fold`，`version` 取 `1`～`10`，`tech` 设为 `CrossValidation`。按论文设定，该设置上建议使用 `mlp_dfl_2` 与 `softmax`。

## 结果汇总（`rank_parser.py`）

跑完各版本后，可用下列命令汇总 Top-1/Top-3/Top-5、MFR、MAR：

```bash
python rank_parser.py <数据父目录绝对路径> <结果输出目录绝对路径> <tech> <model> <loss> <epoch> <sub>
```

其中 `<sub>` 可为 `all`（六项目）、或单个项目名等（与脚本内分支一致）。

由于参数随机初始化，复现实验可能与论文存在微小差异。

---

## 增量学习与持续学习（流式）

除传统 `main.py` 按版本独立训练外，本仓库提供**按缺陷顺序到达**的流式训练脚本：先对前 `warmup` 个缺陷合并做 warmup，再对每个新缺陷做增量更新，并输出 `{mmddHHMM}_continual_metrics.txt`（文件名以月份日期小时分钟开头，文件内含 `runtime_sec` / `runtime` 运行时间、`top1_count` / `top3_count` / `top5_count` 命中个数及原有比例指标）。

### 数据与 `tech` 约定

- 增量脚本默认且**仅允许** `tech=DeepFL_CL`（避免与批次 `DeepFL` 协议混用导致泄漏）；数据目录应为：`<data_root>/DeepFL_CL/<subject>/<ver>/Train.csv` 等。**若本地尚无 `DeepFL_CL`，请先按上文「构建 DeepFL_CL 数据集」运行 `scripts/build_deepfl_cl_dataset.py`。**
- 通用参数（各 `continual_*.py` 基本一致）：

| CLI 参数 | 说明 | 典型默认 |
|----------|------|----------|
| `subject` | 位置参数：项目名 | — |
| `model` | 位置参数：骨干（各脚本支持的集合略有不同，见下表） | — |
| `--data_root` | 数据父目录 | `.` |
| `--out_dir` | 输出根目录；省略则写入 `result_continual/...` 等默认路径 | 自动 |
| `--tech` | 必须为 `DeepFL_CL` | `DeepFL_CL` |
| `--loss` | 当前实现多为 `softmax` | `softmax` |
| `--training_epochs` | Warmup 阶段（合并前 K 个 bug）训练轮数 | `50` |
| `--warmup` | Warmup 覆盖的缺陷个数 K（需满足 `1 <= warmup < v_end`） | `10` |
| `--incr_epochs` | 每个新缺陷增量阶段 epoch 数 | `1` |
| `--v_end` | 流式结束版本号；省略则用 `utils.get_max_ver(subject)` | 项目最大 |
| `--dump_step` | 写入 config 的 dump 步（环境变量 `DEEPFL_DUMP_STEP`） | `10` |
| `--gpu_mem` | GPU 显存占比上限（实现中常截断到 0.2） | `0.2` |

### `continual_stream_train.py`（流式基线 + 可选 EWC）

- **入口**：`python continual_stream_train.py <subject> <model> [选项]`
- **模型**：`mlp`、`mlp2`、`mlp_dfl_1`、`mlp_dfl_2`、`rnn`、`birnn`
- **增量相关**：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--use_ewc` | 是否在增量阶段启用 EWC | `false` |
| `--ewc_lambda` | EWC 惩罚系数 | `10.0` |
| `--ewc_gamma` | Online EWC 衰减 | `0.9` |
| `--optimizer` | `adam` / `sgd`，与 `config.create_optimizer` 一致 | 环境或 `adam` |
| `--use_l2` | 是否在损失中加 L2（FC 分支与 `fc_based` 一致：无 collection 时不加） | `true` |

默认输出目录名会包含 `ue`（EWC 开关）、学习率、`el`/`eg`、warmup、训练轮数等信息；当 `optimizer` 非 `adam` 或 `use_l2` 为 `false` 时，目录 tag 会附加 `_<opt>_l20|1` 以免与旧实验混淆。

### `continual_replay.py`（Replay + DER++，可选 EWC）

| 参数 | 说明 | 默认 |
|------|------|------|
| `--use_replay` | 是否启用回放 + DER++ | `true` |
| `--use_ewc` | 是否同时启用 EWC | `false` |
| `--replay_size` | 回放缓冲区容量（样本条数） | `20` |
| `--replay_per_step` | 每个增量 batch 追加的回放批次数 | `2` |
| `--replay_beta` | DER++ 监督回放项强度 | `1.0` |
| `--distill_alpha` | DER++ 蒸馏项强度 | `0.5` |
| `--temperature` | 知识蒸馏温度 | `2.0` |
| `--ewc_lambda` | EWC 惩罚（`use_ewc` 为真时生效） | `10.0` |
| `--ewc_gamma` | Online EWC 衰减 | `0.9` |

### `continual_gem.py`（Replay + DER++ + A-GEM，可选 EWC）

参数集与 `continual_replay.py` 相同（`--use_replay`、`--use_ewc`、回放容量、`replay_per_step`、`replay_beta`、`distill_alpha`、`temperature`、`ewc_lambda`、`ewc_gamma`）；增量步内使用 A-GEM 式梯度投影（实现见脚本内 `_train_incremental_with_replay_agem`）。

### 非 `main` 入口的全局配置（环境变量）

`config.py` 在非 `main.py` 风格 argv 下会读取 `DEEPFL_*`。`utils.apply_deepfl_env_from_args` 会根据各脚本的 argparse 写入其中一部分；也可在启动前手动 `export`：

| 变量 | 含义 |
|------|------|
| `DEEPFL_DATA_DIR` | 数据根目录 |
| `DEEPFL_OUT_DIR` | 输出根目录 |
| `DEEPFL_SUB` / `DEEPFL_VER` / `DEEPFL_MODEL` | 项目、版本占位、模型名 |
| `DEEPFL_TECH` / `DEEPFL_LOSS` | 技术子目录、损失名 |
| `DEEPFL_EPOCHS` / `DEEPFL_DUMP_STEP` | 训练轮数、dump 步 |
| `DEEPFL_OPTIMIZER` | `adam` / `sgd`（与 `continual_stream_train --optimizer` 一致） |
| `DEEPFL_USE_L2` | `true` / `false` |

### 批量脚本

`scripts/` 目录下提供多组一键实验脚本（如仅 EWC、仅 Replay+DER++、全项目并行、局部网格搜索等）。例如局部网格（`mlp_dfl_1`、EWC-only 与 Replay+GEM）：

```bash
bash scripts/run_09_replay_der_local_grid.sh
```

具体组合与输出目录见该脚本内注释及生成的 `result_grid_local_mlpdfl1/grid_plan.tsv`。

---

## 许可证与引用

若使用本仓库，请同时遵循原 DeepFL 与 Defects4J 的相关许可与引用要求；论文中请按原作者与 Defects4J 规范标注数据来源。
