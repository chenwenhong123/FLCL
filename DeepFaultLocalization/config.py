'''
Configuration and parameters can be set in this file.

支持两种入口：
1) 传统：python main.py <dir> <out> <sub> <ver> <model> <tech> <loss> <epochs> <dump_step>
   此时 len(sys.argv) >= 10，从 argv 读取。
2) 其它脚本（如 continual_stream_train.py）：非 main 风格 argv 时从环境变量 DEEPFL_* 读取。
   注意：不能仅用 len(sys.argv)>=10 判断 main.py，否则 continual 带 --warmup 等长 argv 会把
   sys.argv[8] 误当成 epoch（得到 '--warmup'）。现用「argv[4] 为版本号且 argv[8/9] 为整数」判定 main。
'''

from __future__ import print_function

import os
import sys

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy
import input
import time

techNames = [
    "DeepFL",
    "DeepFL_CL",
    "DeepFL-Spectrum",
    "DeepFL-Mutation",
    "DeepFL-Metrics",
    "DeepFL-Textual",
    "CrossDeepFL",
    "CrossValidation",
]
featureDistr = [
    [34, 35, 35, 35, 35, 37, 15],
    [34, 35, 35, 35, 35, 37, 15],
    [35, 35, 35, 35, 37, 15],
    [34, 37, 15],
    [34, 35, 35, 35, 35, 15],
    [34, 35, 35, 35, 35, 37],
    [34, 35, 35, 35, 35, 37, 15],
    [10],
]
featuresize = [226, 226, 192, 86, 189, 211, 226, 10]
losses = [
    "wsoftmax",
    "softmax",
    "epairwise",
    "epairwiseSoftmax",
    "hpairwise",
    "hpairwiseSoftmax",
]

# 全工程（main / fc_based / 持续学习各模型与 DER++ replay 步）统一使用该学习率。
learning_rate = 0.001
batch_size = 500
display_step = 2
dropout_rate = 0.7
L2_value = 0.0001
optimizer_name = "adam"
use_l2 = True

train_file = "Train.csv"
train_label_file = "TrainLabel.csv"
test_file = "Test.csv"
test_label_file = "TestLabel.csv"
group_dir = "groupfile"
group_file = "traidata.txt.group"
susp_file = "rank"

# 由 _apply_config() 填充
dir = "."
out_dir = "."
sub = ""
v = "0"
model = ""
tech = "DeepFL"
loss = "softmax"
training_epochs = 50
dump_step = 10
featureDistribution = featureDistr[0]
feature = featuresize[0]
rnn_hidden = max(featureDistribution)


def _argv_looks_like_main_py():
    """main.py: <dir> <out> <sub> <ver> <model> <tech> <loss> <epochs> <dump_step> — ver 为数字。"""
    if len(sys.argv) < 10:
        return False
    if not sys.argv[4].isdigit():
        return False
    try:
        int(sys.argv[8])
        int(sys.argv[9])
    except (ValueError, IndexError):
        return False
    return True


def _apply_config():
    """从 sys.argv（main 风格）或环境变量加载与任务相关的全局量。"""
    global dir, out_dir, sub, v, model, tech, loss, training_epochs, dump_step
    global optimizer_name, use_l2
    global featureDistribution, feature, rnn_hidden

    if _argv_looks_like_main_py():
        dir = sys.argv[1]
        out_dir = sys.argv[2]
        sub = sys.argv[3]
        v = sys.argv[4]
        model = sys.argv[5]
        tech = sys.argv[6]
        loss = sys.argv[7]
        training_epochs = int(sys.argv[8])
        dump_step = int(sys.argv[9])
        optimizer_name = str(sys.argv[10]).lower() if len(sys.argv) > 10 else "adam"
        use_l2 = str(sys.argv[11]).lower() in ("1", "true", "yes", "y", "on") if len(sys.argv) > 11 else True
    else:
        dir = os.environ.get("DEEPFL_DATA_DIR", ".")
        out_dir = os.environ.get("DEEPFL_OUT_DIR", ".")
        sub = os.environ.get("DEEPFL_SUB", "")
        v = os.environ.get("DEEPFL_VER", "0")
        model = os.environ.get("DEEPFL_MODEL", "")
        tech = os.environ.get("DEEPFL_TECH", "DeepFL")
        loss = os.environ.get("DEEPFL_LOSS", "softmax")
        training_epochs = int(os.environ.get("DEEPFL_EPOCHS", "50"))
        dump_step = int(os.environ.get("DEEPFL_DUMP_STEP", "10"))
        optimizer_name = os.environ.get("DEEPFL_OPTIMIZER", "adam").lower()
        use_l2 = os.environ.get("DEEPFL_USE_L2", "true").lower() in ("1", "true", "yes", "y", "on")

    featureDistribution = featureDistr[techNames.index(tech)]
    feature = featuresize[techNames.index(tech)]
    rnn_hidden = max(featureDistribution)


def create_optimizer():
    name = str(optimizer_name).lower()
    if name == "adam":
        return tf.train.AdamOptimizer(learning_rate=learning_rate)
    if name == "sgd":
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    raise ValueError("unknown optimizer_name: %s (supported: adam, sgd)" % name)


_apply_config()
