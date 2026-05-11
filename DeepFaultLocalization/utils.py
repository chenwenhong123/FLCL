from __future__ import print_function

import os

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

try:
    from tqdm import tqdm as _tqdm_real
except ImportError:
    _tqdm_real = None


def tqdm_compat(it=None, total=None, desc=None, leave=True, **kwargs):
    """有 tqdm 则用 tqdm；否则返回可迭代的占位进度条。"""
    if _tqdm_real is not None:
        if it is not None:
            return _tqdm_real(it, desc=desc, leave=leave, **kwargs)
        return _tqdm_real(total=total, desc=desc, leave=leave, **kwargs)

    class _Pbar:
        def __init__(self, n):
            self.n = n or 0

        def update(self, x=1):
            pass

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    if it is not None:
        return it
    return _Pbar(total)


def get_max_ver(subject):
    m = {
        "Chart": 26,
        "Lang": 65,
        "Math": 106,
        "Time": 27,
        "Closure": 133,
        "Mockito": 38,
    }
    return m[subject]


def fill_matrix_by_feature_dist(x, feature_distribution):
    """与 recurrent_network.fillMatrix 数值一致；用 numpy 切片替代 Python 双层循环，适合大 batch。"""
    fd = np.asarray(feature_distribution, dtype=np.intp)
    max_g = int(fd.max())
    n_steps = int(fd.shape[0])
    b = int(x.shape[0])
    filled = np.zeros((b, max_g * n_steps), dtype=x.dtype)
    xi = 0
    for i, w in enumerate(fd):
        w = int(w)
        lo = i * max_g
        filled[:, lo : lo + w] = x[:, xi : xi + w]
        xi += w
    return filled.reshape(b, n_steps, max_g)


def parse_rank_min_avg(rank_scores, label_first_col):
    """与 rank_parser.parse 中由分数算名次的逻辑一致；(min_rank, avg_rank)；无正样本返回 (-1,-1)。"""
    rank_list = np.asarray(rank_scores, dtype=np.float32)
    label_list = np.asarray(label_first_col, dtype=int)
    u, v = np.unique(-rank_list, return_inverse=True)
    cost = (np.cumsum(np.bincount(v)))[v]
    ranks = []
    for i in range(len(label_list)):
        if label_list[i] == 1:
            ranks.append(cost[i])
    ranks = np.asarray(ranks, dtype=np.float32)
    if len(ranks) == 0:
        return -1.0, -1.0
    return float(ranks.min()), float(ranks.mean())


def topk_hit(min_rank, k):
    if min_rank < 0:
        return 0.0
    return 1.0 if min_rank <= k else 0.0


def aggregate_official_style(per_bug_eval):
    """per_bug_eval: 含 min、avg 的 dict 列表；与 rank_parser 项目级 tops + ranks 类似。"""
    tops = np.zeros(4)
    ranks = np.zeros(2)
    n = 0
    for e in per_bug_eval:
        if e is None or e["min"] < 0:
            continue
        n += 1
        if e["min"] <= 1:
            tops[0] += 1
        if e["min"] <= 3:
            tops[1] += 1
        if e["min"] <= 5:
            tops[2] += 1
        if e["min"] <= 10:
            tops[3] += 1
        ranks[0] += e["min"]
        ranks[1] += e["avg"]
    if n == 0:
        return None
    ranks = ranks / n
    return int(tops[0]), int(tops[1]), int(tops[2]), round(float(ranks[0]), 2), round(float(ranks[1]), 2)


def apply_deepfl_env_from_args(args):
    """在 import config 前写入 DEEPFL_*，供 config._apply_config() 使用。"""
    os.environ["DEEPFL_DATA_DIR"] = args.data_root
    # continual 等脚本常在默认 out_dir 推导之前调用本函数；out_dir 尚未赋值时不能写 None 到 environ
    os.environ["DEEPFL_OUT_DIR"] = getattr(args, "out_dir", None) or "."
    os.environ["DEEPFL_SUB"] = args.subject
    os.environ["DEEPFL_VER"] = "0"
    os.environ["DEEPFL_MODEL"] = args.model
    os.environ["DEEPFL_TECH"] = args.tech
    os.environ["DEEPFL_LOSS"] = args.loss
    os.environ["DEEPFL_EPOCHS"] = str(args.training_epochs)
    os.environ["DEEPFL_DUMP_STEP"] = str(getattr(args, "dump_step", 10))
    opt = getattr(args, "optimizer", None)
    if opt is not None:
        os.environ["DEEPFL_OPTIMIZER"] = str(opt).lower()
    if getattr(args, "use_l2", None) is not None:
        os.environ["DEEPFL_USE_L2"] = "true" if args.use_l2 else "false"


def loss_func(pred, y, loss, datasets, groups=[]):
    if loss == 0: # weighted softmax
        ratio=datasets.train.pos_instance_ratio()
        classes_weights=tf.constant([[1.0-ratio, ratio]])
        weight_per_label = tf.transpose( tf.matmul(y, tf.transpose(classes_weights)) )
        xent = tf.multiply(weight_per_label, tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y)) #shape [1, batch_size]              
        cost = tf.reduce_mean(xent)
    elif loss == 1: # softmax
        cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
    elif loss == 2 and groups!=[]: # group-based exponential pairwise implementation
        pred_scaled=tf.nn.softmax(pred)
        pred_1=tf.slice(pred_scaled,[0,0],[-1,1])
        y_1=tf.slice(y,[0,0],[-1,1])
        diff=tf.subtract(pred_1,tf.transpose(pred_1))
        grouping=tf.equal(groups, tf.transpose(groups))
        grouping=tf.cast(grouping,tf.float32)
        mask=y_1*tf.transpose(1-y_1)*grouping
        diff_exp=tf.exp(-diff)
        cost=tf.reduce_sum(mask*diff_exp) 
    elif loss == 3 and groups!=[]: # softmax + group-based exponential pairwise implementation
        pred_scaled=tf.nn.softmax(pred)
        pred_1=tf.slice(pred_scaled,[0,0],[-1,1])
        y_1=tf.slice(y,[0,0],[-1,1])
        diff=tf.subtract(pred_1,tf.transpose(pred_1))
        grouping=tf.equal(groups, tf.transpose(groups))
        grouping=tf.cast(grouping,tf.float32)
        mask=y_1*tf.transpose(1-y_1)*grouping
        pair_num=tf.reduce_sum(mask)
        diff_exp=tf.exp(-diff)
        softmax_cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
        cost= tf.cond(tf.greater(pair_num,tf.constant(0.0)), lambda: softmax_cost+tf.divide(tf.reduce_sum(mask*diff_exp),pair_num), lambda: softmax_cost)

    elif loss == 4 and groups!=[]: # group-based hinge pairwise implementation
        pred_scaled=tf.nn.softmax(pred)
        pred_1=tf.slice(pred_scaled,[0,0],[-1,1])
        y_1=tf.slice(y,[0,0],[-1,1])
        diff=tf.subtract(pred_1,tf.transpose(pred_1))
        grouping=tf.equal(groups, tf.transpose(groups))
        grouping=tf.cast(grouping,tf.float32)
        mask=y_1*tf.transpose(1-y_1)*grouping
        diff=mask*diff
        cost=tf.reduce_sum(tf.subtract(mask,diff)) 
    elif loss == 5 and groups!=[]: # softmax + group-based hinge pairwise implementation
        pred_scaled=tf.nn.softmax(pred)
        pred_1=tf.slice(pred_scaled,[0,0],[-1,1])
        y_1=tf.slice(y,[0,0],[-1,1])
        diff=tf.subtract(pred_1,tf.transpose(pred_1))
        grouping=tf.equal(groups, tf.transpose(groups))
        grouping=tf.cast(grouping,tf.float32)
        mask=y_1*tf.transpose(1-y_1)*grouping
        pair_num=tf.reduce_sum(mask)
        diff=mask*diff
        softmax_cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
        cost= tf.cond(tf.greater(pair_num,tf.constant(0.0)), lambda: softmax_cost+tf.reduce_sum(tf.divide(tf.subtract(mask, diff),pair_num)), lambda: softmax_cost)

    elif False: # optimized pairwise exponential function without grouping, only for reference 
        pred_scaled=tf.nn.softmax(pred)
        pred_1=tf.slice(pred_scaled,[0,0],[-1,1])
        y_1=tf.slice(y,[0,0],[-1,1])
        positive=tf.greater(y_1, 0.0)
        negative=tf.less(y_1, 1.0)
        pred_x=tf.boolean_mask(pred_1,positive)
        pred_x=tf.reshape(pred_x,[-1,1])
        pred_y=tf.boolean_mask(pred_1,negative)
        pred_y=tf.reshape(pred_y,[1,-1])
        diff=tf.subtract(pred_x,pred_y)
        diff_exp=tf.exp(-diff)
        cost=tf.reduce_sum(diff_exp) 
    
    return cost
