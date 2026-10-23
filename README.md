# FLCL：面向故障定位的持续学习实验代码

本仓库对应论文 **“Make Learning-based FL Keep Learning: An Empirical Study of Continual Learning for Incremental Fault Localization”** 的实验代码。研究在 Defects4J v1.2 的六个项目、共 395 个缺陷上，考察三种 DeepFL 配置（MLP、MLPDFL、BiRNN）与 Grace。两条实现分别以多源特征模型和图表示模型为基础，包含离线 leave-one-out（LOO）基线、基础增量训练，以及经验回放、DER++ 风格蒸馏、EWC、A-GEM 等持续学习策略与组合。

论文的实验分为三个阶段：比较离线 LOO 与增量训练；分析持续学习策略对定位效果和知识保持的影响；研究 warmup 缺陷数、基础训练轮数和增量训练轮数的敏感性。仓库提供相应的单次入口、批量运行和消融脚本。

本 README 是代码与复现入口；具体参数、数据格式和各实验脚本见 [DeepFL 说明](DeepFaultLocalization/README.md) 与 [Grace 说明](Grace/README.md)。两套实现的模型、数据和指标计算流程各自独立；请在同一实现及相同评估口径下比较实验结果。

## 仓库结构

- [`DeepFaultLocalization/`](DeepFaultLocalization/)：TensorFlow 兼容 v1 API 的 DeepFL 实现。`main.py` 为逐版本训练入口，`continual_stream_train.py`、`continual_replay.py`、`continual_gem.py` 为流式训练入口；`scripts/` 含数据转换及批量实验脚本。
- [`Grace/Code/Default/`](Grace/Code/Default/)：PyTorch 的 Grace 实现。`run.py`、`sum.py`、`watch.py` 与 `runtotal.py` 用于原始逐缺陷流程；`continual_run*.py` 为增量及策略组合入口；`scripts/` 含批量实验脚本。
- [`environment.yml`](environment.yml)：参考 Conda 环境，包含 Python 3.10、TensorFlow、PyTorch、NumPy、Pandas 等依赖。该导出文件含平台相关的精确构建版本，跨平台使用时可能需要调整。

## 数据与环境准备

**数据文件不随本仓库提供。**请先下载对应方法的预处理数据，并遵守原数据集的使用要求。两个方法使用 Defects4J 项目 `Chart`、`Lang`、`Math`、`Time`、`Closure`、`Mockito`；具体缺陷编号及数据映射由各自脚本决定。

1. 创建并激活参考环境：

   ```bash
   conda env create -f environment.yml
   conda activate fl
   ```

2. **DeepFL 数据**：从 [DeepFL 数据下载地址](https://mega.nz/#F!7rhCwQzT!OXUlRaIh-kyevSr6sTdwxA) 获取压缩包，解压后使数据父目录包含 `DeepFL/<Project>/<Version>/Train.csv`、`TrainLabel.csv`、`Test.csv`、`TestLabel.csv`，以及相应的 `groupfile/`。持续学习脚本固定使用单独的 `DeepFL_CL` 子目录；可用仓库脚本从原始 `DeepFL` 数据生成：

   ```bash
   cd DeepFaultLocalization
   python scripts/build_deepfl_cl_dataset.py --data_root /path/to/data
   ```

   生成脚本对每个缺陷版本复制其 `Test.csv` / `TestLabel.csv` 到 `DeepFL_CL` 中该版本的 Train 与 Test 文件，并生成 group 文件。流式脚本先对新版本评估，再在该版本数据上更新；请勿将原始批式目录直接指定为 `DeepFL_CL`。其他数据变体及目录约定见 [DeepFL 数据说明](DeepFaultLocalization/README.md#数据集)。

3. **Grace 数据**：从 [Grace 预处理数据下载地址](https://drive.google.com/drive/folders/1QH_Y9fKaNrwQCT6hvAH9-73PBQQ3a4hL?usp=sharing) 获取各项目的 `.pkl` 文件，放在 `Grace/Code/Default/GraceDate/` 下，或在 `Grace/Code/Default/` 当前目录下放置脚本所需文件。目录名 `GraceDate` 是代码中的既有拼写。数据读取及缓存细节见 [Grace 数据说明](Grace/README.md#数据集)。

以下命令均从对应方法目录执行。运行完整六项目实验前，建议先确认单项目数据已就位，并预留模型训练所需时间与计算资源。

## 快速复现

### DeepFL

在 `DeepFaultLocalization/` 下，先运行一个逐版本基线，再运行一个流式基线。`/path/to/data` 是包含 `DeepFL/` 与 `DeepFL_CL/` 的父目录；`/path/to/output` 是可写的结果目录。

```bash
cd DeepFaultLocalization
python main.py /path/to/data /path/to/output Chart 1 mlp DeepFL softmax 50 10
python continual_stream_train.py Chart mlp \
  --data_root /path/to/data --tech DeepFL_CL \
  --warmup 10 --training_epochs 50 --incr_epochs 1 --v_end 12 \
  --out_dir /path/to/output/deepfl_stream_chart
```

逐版本结果写到 `<输出目录>/<Project>/<Version>/<tech>/`；流式结果含 `*_continual_metrics.txt`，记录 Top-1/3/5、更新前指标、MFR/MAR、BWT 和运行时间。要运行 Replay + DER++ 或 A-GEM，可将流式入口分别改为 `continual_replay.py`、`continual_gem.py`；EWC 可通过相应脚本的 `--use_ewc true` 启用。完整选项及批量脚本见 [DeepFL 增量实验说明](DeepFaultLocalization/README.md#增量学习与持续学习流式)。

### Grace

在 `Grace/Code/Default/` 下，运行一个项目的逐缺陷基线或基础增量实验：

```bash
cd Grace/Code/Default
python runtotal.py Lang
python continual_run.py Lang 0.01 0 60 10 15 2
```

`runtotal.py` 会调用 `run.py`、`sum.py`、`watch.py`。增量实验按顺序接收缺陷 ID；上述位置参数依次为项目、学习率、随机种子、批大小、warmup 缺陷数、基础训练轮数、每步增量轮数。结果写入 `result/<Project>/<运行标识>/`，包括逐步 `metrics.csv`、最终 `model.pt` 和 `summary.txt`。Replay + DER++、EWC、A-GEM 及组合入口和完整参数顺序见 [Grace 增量实验说明](Grace/README.md#增量学习实验持续学习脚本)。

## 实验协议与指标

- **离线 LOO 基线**：每个目标缺陷使用独立训练的模型进行评估，目标缺陷不参与该次训练，用于与持续学习设置对照。DeepFL 使用 `main.py`；Grace 使用 `run.py` 或 `runtotal.py`。
- **流式设置**：先用前 `K` 个缺陷进行 warmup，再依次处理后续缺陷。DeepFL 使用版本号顺序；Grace 使用其预处理数据中的缺陷 ID 顺序。两者的 warmup 实现不同，复现时应保留各自脚本的原始协议。
- **排序结果**：Top-1/3/5 表示真实缺陷出现在前 1/3/5 个候选中的命中情况；MFR/MAR 为排名汇总指标；BWT 用于观察后续学习对已见缺陷表现的影响。输出同时包含命中个数与比例时，请区分两者。
- **口径提醒**：DeepFL 流式结果中的 `incr_pre_*` 汇总使用 warmup 后的评估及后续缺陷**本步更新前**的评估；Grace 基础增量的 `online_top1_t` 是**本步更新后**的当前缺陷命中。两项不可直接作为同一在线指标比较。请以具体结果文件的字段与相应脚本为准。
- **论文协议与脚本输出**：论文稿件将 warmup 缺陷的评估定义为仅在 warmup 集内进行 LOO，并将后续缺陷的评估定义为更新前预测。当前流式脚本的 warmup 指标和 Grace 的 `online_top1_t` 按各自实现生成；若要对照论文表格，请先核对 warmup LOO 结果与对应输出字段。

论文实验的批量运行、参数扫描和消融入口分别位于 [`DeepFaultLocalization/scripts/`](DeepFaultLocalization/scripts/) 与 [`Grace/Code/Default/scripts/`](Grace/Code/Default/scripts/)。其中 DeepFL 的 `run_10_oneshot_and_stream_baseline.sh` 与 Grace 的 `run_oneshot_and_continual_baseline.sh` 可运行各自的基线组合；其他脚本的策略及输出路径见子目录 README。

## 引用与数据来源

本仓库论文题名：**Make Learning-based FL Keep Learning: An Empirical Study of Continual Learning for Incremental Fault Localization**。目前提供的论文稿件未列出作者、DOI 或正式发表信息，因此此处不填写未经核实的 BibTeX；正式引文请以最终发表页面为准。

使用数据与基线实现时，也请引用原始 DeepFL、Grace 方法及 [Defects4J](https://github.com/rjust/defects4j) 数据集，并遵守其许可与引用要求。
