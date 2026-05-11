from __future__ import print_function

"""
数据约定（DeepFL 目录结构）：
每个 <tech>/<subject>/<ver>/ 对应「单个 bug 版本」。该目录下 Train/TrainLabel 是该 bug 的
候选实体（如语句）的特征与标签，不是「整个项目除一个 id」的全集划分；Test/TestLabel 是
同一 bug 下用于评估的另一批候选。训练只用 Train（与 main.py 各模型 run 一致），
eval_bug / 写 rank 用 Test，不把 Test 标签用于反向传播。

默认输出根目录：result_continual/<model>/<subject>/loss_<loss>_wup<warmup>_te<training_epochs>_incr<incr>_v<v_end>/
"""

import argparse
import math
import os
import sys

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import input
import utils as ut
from utils import tqdm_compat as tqdm

BATCH_SIZE = 500
DROPOUT_RATE = 0.7

TRAIN_FILE = "Train.csv"
TRAIN_LABEL_FILE = "TrainLabel.csv"
TEST_FILE = "Test.csv"
TEST_LABEL_FILE = "TestLabel.csv"
GROUP_DIR = "groupfile"
GROUP_FILE = "traidata.txt.group"

LOSSES = ["wsoftmax", "softmax", "epairwise", "epairwiseSoftmax", "hpairwise", "hpairwiseSoftmax"]
TECH_NAMES = ["DeepFL", "DeepFL_CL", "DeepFL-Spectrum", "DeepFL-Mutation", "DeepFL-Metrics", "DeepFL-Textual", "CrossDeepFL", "CrossValidation"]
FEATURE_SIZE = [226, 226, 192, 86, 189, 211, 226, 10]


def _to_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def load_one_bug(data_root, tech, subject, ver):
    base = os.path.join(data_root, tech, subject, str(ver))
    gpath = os.path.join(data_root, tech, GROUP_DIR, subject, str(ver), GROUP_FILE)
    ti = input.readFile(os.path.join(base, TRAIN_FILE))
    tl = input.readFile(os.path.join(base, TRAIN_LABEL_FILE))
    xi = input.readFile(os.path.join(base, TEST_FILE))
    xl = input.readFile(os.path.join(base, TEST_LABEL_FILE))
    groups = input.readGroup(gpath)
    return ti, tl, groups, xi, xl


def merge_train_bugs(data_root, tech, subject, v_start, v_end):
    """合并 [v_start, v_end]（闭区间）各 bug 的 Train + Label + group（仅训练集，见模块 docstring）。"""
    xs, ys, gs = [], [], []
    gid_off = 0
    for v in range(v_start, v_end + 1):
        ti, tl, g, _, _ = load_one_bug(data_root, tech, subject, v)
        xs.append(ti)
        ys.append(tl)
        gmax = int(g.max()) if g.size else 0
        gs.append(g + gid_off)
        gid_off += gmax + 1
    return np.vstack(xs), np.vstack(ys), np.vstack(gs)


def build_mlp_graph(n_input, n_hidden, loss_idx):
    import config as cfg

    lr = float(cfg.learning_rate)
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, 2])
    g = tf.placeholder(tf.int32, [None, 1])
    keep_prob = tf.placeholder(tf.float32)

    weights = {
        "h1": tf.Variable(tf.random_normal([n_input, n_hidden])),
        "out": tf.Variable(tf.random_normal([n_hidden, 2])),
    }
    biases = {
        "b1": tf.Variable(tf.random_normal([n_hidden])),
        "out": tf.Variable(tf.random_normal([2])),
    }

    layer_1 = tf.add(tf.matmul(x, weights["h1"]), biases["b1"])
    layer_1 = tf.nn.sigmoid(layer_1)
    drop_out = tf.nn.dropout(layer_1, keep_prob)
    pred = tf.matmul(drop_out, weights["out"]) + biases["out"]

    l2_w = tf.nn.l2_loss(weights["h1"]) + tf.nn.l2_loss(weights["out"])
    reg_term = (
        l2_w * float(cfg.L2_value) if cfg.use_l2 else tf.constant(0.0, dtype=tf.float32)
    )

    ref_holder = {"ds": None}

    def make_cost():
        ds = ref_holder["ds"]
        return ut.loss_func(pred, y, loss_idx, ds, g)

    ref_holder["ds"] = type("_PH", (), {"train": None, "test": None})()
    cost = make_cost()
    prob = tf.nn.softmax(pred)
    optimizer = cfg.create_optimizer().minimize(cost + reg_term)
    return {
        "x": x,
        "y": y,
        "g": g,
        "keep_prob": keep_prob,
        "pred": pred,
        "prob": prob,
        "cost": cost,
        "optimizer": optimizer,
        "ref_holder": ref_holder,
        "kind": "mlp",
        "learning_rate": lr,
    }


def build_fc_stream_graph(loss_idx, module_name, pred_attr):
    """DeepFL 226 维分支全连接模型（fc_based_1 / fc_based_2），依赖已配置好的 config。"""
    import importlib

    import config as cfg

    fc = importlib.import_module(module_name)
    pred_fn = getattr(fc, pred_attr)
    spec = tf.placeholder("float", [None, 34])
    mutation1 = tf.placeholder("float", [None, 35])
    mutation2 = tf.placeholder("float", [None, 35])
    mutation3 = tf.placeholder("float", [None, 35])
    mutation4 = tf.placeholder("float", [None, 35])
    complexity = tf.placeholder("float", [None, 37])
    similarity = tf.placeholder("float", [None, 15])
    y = tf.placeholder("float", [None, 2])
    g = tf.placeholder(tf.int32, [None, 1])
    is_training = tf.placeholder(tf.bool, name="is_training")
    keep_prob = tf.placeholder(tf.float32)

    pred = pred_fn(
        spec,
        mutation1,
        mutation2,
        mutation3,
        mutation4,
        complexity,
        similarity,
        keep_prob,
        is_training,
    )
    y_sg = tf.stop_gradient(y)
    ref_holder = {"ds": type("_PH", (), {"train": None, "test": None})()}
    cost = ut.loss_func(pred, y_sg, loss_idx, ref_holder["ds"], g)
    regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = (
        tf.add_n(regs)
        if (cfg.use_l2 and regs)
        else tf.constant(0.0, dtype=tf.float32)
    )
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = cfg.create_optimizer().minimize(cost + reg_term)
    prob = tf.nn.softmax(pred)
    return {
        "kind": "fc",
        "spec": spec,
        "mutation1": mutation1,
        "mutation2": mutation2,
        "mutation3": mutation3,
        "mutation4": mutation4,
        "complexity": complexity,
        "similarity": similarity,
        "y": y,
        "g": g,
        "keep_prob": keep_prob,
        "is_training": is_training,
        "pred": pred,
        "prob": prob,
        "cost": cost,
        "optimizer": optimizer,
        "ref_holder": ref_holder,
        "learning_rate": float(cfg.learning_rate),
    }


def _feed_fc(graph, batch_x):
    return {
        graph["spec"]: batch_x[:, :34],
        graph["mutation1"]: batch_x[:, 34:69],
        graph["mutation2"]: batch_x[:, 69:104],
        graph["mutation3"]: batch_x[:, 104:139],
        graph["mutation4"]: batch_x[:, 139:174],
        graph["complexity"]: batch_x[:, 174:211],
        graph["similarity"]: batch_x[:, -15:],
    }


def build_seq_stream_graph(seq_kind, loss_idx, feat_dist):
    """rnn / birnn，与 recurrent_network / bidirectional_rnn 一致。"""
    import config as cfg
    import recurrent_network as rn

    n_input = int(np.array(feat_dist).max())
    n_steps = len(feat_dist)
    n_hidden = int(np.array(feat_dist).max())
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, 2])
    g = tf.placeholder(tf.int32, [None, 1])
    keep_prob = tf.placeholder(tf.float32)
    if seq_kind == "rnn":
        weights = {"out": tf.Variable(tf.random_normal([n_hidden, 2]))}
        biases = {"out": tf.Variable(tf.random_normal([2]))}
        pred = rn.RNN(x, weights, biases, n_hidden, n_steps, keep_prob)
    elif seq_kind == "birnn":
        import bidirectional_rnn as bd

        weights = {"out": tf.Variable(tf.random_normal([2 * n_hidden, 2]))}
        biases = {"out": tf.Variable(tf.random_normal([2]))}
        pred = bd.BiRNN(x, weights, biases, n_hidden, n_steps, keep_prob)
    else:
        raise ValueError("unknown seq_kind %s" % seq_kind)

    ref_holder = {"ds": type("_PH", (), {"train": None, "test": None})()}
    cost = ut.loss_func(pred, y, loss_idx, ref_holder["ds"], g)
    variables = tf.trainable_variables()
    l2_w = tf.add_n([tf.nn.l2_loss(v) for v in variables if "bias" not in v.name])
    reg_term = (
        l2_w * float(cfg.L2_value)
        if cfg.use_l2
        else tf.constant(0.0, dtype=tf.float32)
    )
    optimizer = cfg.create_optimizer().minimize(cost + reg_term)
    prob = tf.nn.softmax(pred)
    return {
        "kind": seq_kind,
        "x": x,
        "y": y,
        "g": g,
        "keep_prob": keep_prob,
        "pred": pred,
        "prob": prob,
        "cost": cost,
        "optimizer": optimizer,
        "ref_holder": ref_holder,
        "n_input": n_input,
        "n_steps": n_steps,
        "feat_dist": list(feat_dist),
        "learning_rate": float(cfg.learning_rate),
    }


def build_mlp2_graph(n_input, n_hidden, loss_idx):
    """与 multilayer_perceptron_two_hidden_layer 一致的两层 MLP；学习率与全工程统一为 config.learning_rate。"""
    import config as cfg

    lr = float(cfg.learning_rate)
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, 2])
    g = tf.placeholder(tf.int32, [None, 1])
    keep_prob = tf.placeholder(tf.float32)

    weights = {
        "h1": tf.Variable(tf.random_normal([n_input, n_hidden])),
        "h2": tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        "out": tf.Variable(tf.random_normal([n_hidden, 2])),
    }
    biases = {
        "b1": tf.Variable(tf.random_normal([n_hidden])),
        "b2": tf.Variable(tf.random_normal([n_hidden])),
        "out": tf.Variable(tf.random_normal([2])),
    }

    layer_1 = tf.add(tf.matmul(x, weights["h1"]), biases["b1"])
    layer_1 = tf.nn.sigmoid(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    layer_2 = tf.add(tf.matmul(layer_1, weights["h2"]), biases["b2"])
    layer_2 = tf.nn.sigmoid(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob)
    pred = tf.matmul(layer_2, weights["out"]) + biases["out"]

    l2_w = (
        tf.nn.l2_loss(weights["h1"])
        + tf.nn.l2_loss(weights["h2"])
        + tf.nn.l2_loss(weights["out"])
    )
    reg_term = (
        l2_w * float(cfg.L2_value) if cfg.use_l2 else tf.constant(0.0, dtype=tf.float32)
    )

    ref_holder = {"ds": None}

    def make_cost():
        ds = ref_holder["ds"]
        return ut.loss_func(pred, y, loss_idx, ds, g)

    ref_holder["ds"] = type("_PH", (), {"train": None, "test": None})()
    cost = make_cost()
    prob = tf.nn.softmax(pred)
    optimizer = cfg.create_optimizer().minimize(cost + reg_term)
    return {
        "x": x,
        "y": y,
        "g": g,
        "keep_prob": keep_prob,
        "pred": pred,
        "prob": prob,
        "cost": cost,
        "optimizer": optimizer,
        "ref_holder": ref_holder,
        "kind": "mlp2",
        "learning_rate": lr,
    }


def _train_feed_dict(graph, batch_x, batch_y, batch_g, keep_prob):
    kind = graph["kind"]
    if kind in ("mlp", "mlp2"):
        return {
            graph["x"]: batch_x,
            graph["y"]: batch_y,
            graph["g"]: batch_g,
            graph["keep_prob"]: keep_prob,
        }
    if kind == "fc":
        fd = _feed_fc(graph, batch_x)
        fd[graph["y"]] = batch_y
        fd[graph["g"]] = batch_g
        fd[graph["keep_prob"]] = keep_prob
        fd[graph["is_training"]] = keep_prob < 1.0
        return fd
    if kind in ("rnn", "birnn"):
        bx2 = ut.fill_matrix_by_feature_dist(batch_x, graph["feat_dist"])
        return {
            graph["x"]: bx2,
            graph["y"]: batch_y,
            graph["g"]: batch_g,
            graph["keep_prob"]: keep_prob,
        }
    raise ValueError("unknown graph kind %s" % kind)


def build_ewc_state(graph, ewc_lambda):
    """Build EWC tensors; reused by continual_replay.py."""
    vars_all = tf.trainable_variables()
    omega_ph = []
    star_ph = []
    penalty_terms = []
    for i, v in enumerate(vars_all):
        op = tf.placeholder(tf.float32, shape=v.shape, name="ewc_omega_%d" % i)
        sp = tf.placeholder(tf.float32, shape=v.shape, name="ewc_star_%d" % i)
        omega_ph.append(op)
        star_ph.append(sp)
        penalty_terms.append(tf.reduce_sum(op * tf.square(v - sp)))
    if penalty_terms:
        ewc_penalty = 0.5 * tf.add_n(penalty_terms)
    else:
        ewc_penalty = tf.constant(0.0, dtype=tf.float32)
    import config as cfg

    total_cost = graph["cost"] + float(ewc_lambda) * ewc_penalty
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer_ewc = cfg.create_optimizer().minimize(total_cost)
    grads = tf.gradients(graph["cost"], vars_all)
    return {
        "vars": vars_all,
        "omega_ph": omega_ph,
        "star_ph": star_ph,
        "optimizer_ewc": optimizer_ewc,
        "grads": grads,
    }


def _ewc_feed(ewc_state, omega, star):
    fd = {}
    for i, ph in enumerate(ewc_state["omega_ph"]):
        fd[ph] = omega[i]
    for i, ph in enumerate(ewc_state["star_ph"]):
        fd[ph] = star[i]
    return fd


def _estimate_fisher(sess, graph, ewc_state, train_x, train_y, train_g):
    n = train_x.shape[0]
    ds = input.DataSet(train_x, train_y, train_g)
    total_batch = max(1, int(math.ceil(n / float(BATCH_SIZE))))
    fisher = [np.zeros(v.shape.as_list(), dtype=np.float32) for v in ewc_state["vars"]]
    cnt = 0.0
    for _ in range(total_batch):
        bx, by, bg = ds.next_batch(BATCH_SIZE)
        fd = _train_feed_dict(graph, bx, by, bg, 1.0)
        gv = sess.run(ewc_state["grads"], feed_dict=fd)
        for i, g in enumerate(gv):
            if g is None:
                continue
            fisher[i] += np.square(g).astype(np.float32)
        cnt += 1.0
    if cnt > 0:
        fisher = [f / cnt for f in fisher]
    return fisher


def update_ewc_stats(sess, graph, ewc_state, train_x, train_y, train_g, omega, gamma):
    fisher = _estimate_fisher(sess, graph, ewc_state, train_x, train_y, train_g)
    new_omega = [float(gamma) * o + f for o, f in zip(omega, fisher)]
    star = sess.run(ewc_state["vars"])
    return new_omega, star


def train_epochs(sess, graph, train_x, train_y, train_g, epochs, desc="train", ewc_state=None, ewc_stats=None):
    """在固定数据上训练若干 epoch；支持 mlp / mlp2 / fc / rnn / birnn。"""
    n = train_x.shape[0]
    ds = input.DataSet(train_x, train_y, train_g)
    graph["ref_holder"]["ds"].train = ds
    total_batch = max(1, int(math.ceil(n / float(BATCH_SIZE))))
    total_steps = max(0, epochs * total_batch)
    pbar = tqdm(total=total_steps, desc=desc, leave=False)
    try:
        for _ in range(epochs):
            ds._index_in_epoch = 0
            ds._epochs_completed = 0
            for __ in range(total_batch):
                bx, by, bg = ds.next_batch(BATCH_SIZE)
                fd = _train_feed_dict(graph, bx, by, bg, DROPOUT_RATE)
                if ewc_state is not None and ewc_stats is not None:
                    fd.update(_ewc_feed(ewc_state, ewc_stats["omega"], ewc_stats["star"]))
                    sess.run(ewc_state["optimizer_ewc"], feed_dict=fd)
                else:
                    sess.run(graph["optimizer"], feed_dict=fd)
                pbar.update(1)
    finally:
        pbar.close()


def _eval_feed_dict(graph, kind, tx, ty):
    """单段 test 切片对应的 feed_dict（tx/ty 行数一致）。"""
    if kind in ("mlp", "mlp2"):
        return {
            graph["x"]: tx,
            graph["y"]: ty,
            graph["keep_prob"]: 1.0,
            graph["g"]: np.zeros((tx.shape[0], 1), dtype=np.int32),
        }
    if kind == "fc":
        fd = _feed_fc(graph, tx)
        fd[graph["y"]] = ty
        fd[graph["keep_prob"]] = 1.0
        fd[graph["is_training"]] = False
        fd[graph["g"]] = np.zeros((tx.shape[0], 1), dtype=np.int32)
        return fd
    if kind in ("rnn", "birnn"):
        td = ut.fill_matrix_by_feature_dist(tx, graph["feat_dist"])
        return {
            graph["x"]: td,
            graph["y"]: ty,
            graph["keep_prob"]: 1.0,
            graph["g"]: np.zeros((tx.shape[0], 1), dtype=np.int32),
        }
    raise ValueError("unknown graph kind %s" % kind)


def eval_bug(sess, graph, test_x, test_y):
    if test_x.shape[0] == 0:
        return None
    kind = graph["kind"]
    n = test_x.shape[0]
    prob_tensor = graph["prob"]
    scores_parts = []
    for s in range(0, n, BATCH_SIZE):
        e = min(s + BATCH_SIZE, n)
        fd = _eval_feed_dict(graph, kind, test_x[s:e], test_y[s:e])
        p = sess.run(prob_tensor, feed_dict=fd)
        scores_parts.append(p[:, 0])
    prob_col0 = np.concatenate(scores_parts, axis=0)
    scores = prob_col0.tolist()
    labels = test_y[:, 0].astype(int)
    mn, avg = ut.parse_rank_min_avg(scores, labels)
    return {
        "scores": scores,
        "min": mn,
        "avg": avg,
        "top1": ut.topk_hit(mn, 1),
        "top3": ut.topk_hit(mn, 3),
        "top5": ut.topk_hit(mn, 5),
    }


def main():
    pa = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="流式持续学习。最短: python continual_stream_train.py Lang mlp",
    )
    pa.add_argument("subject", help="项目: Chart|Lang|Math|Time|Closure|Mockito")
    pa.add_argument(
        "model",
        help="mlp | mlp2 | mlp_dfl_1 | mlp_dfl_2 | rnn | birnn",
    )
    pa.add_argument("--data_root", default=".", help="数据父目录（含 <tech>/<subject>/<ver>/）")
    pa.add_argument(
        "--out_dir",
        default=None,
        help="输出根目录；省略则为 result_continual/<model>/<subject>/wup*_te*_incr*_v*/",
    )
    pa.add_argument("--tech", default="DeepFL_CL", help="数据与输出路径中的技术子目录名（持续学习固定使用 DeepFL_CL）")
    pa.add_argument("--loss", default="softmax", help="损失函数名（当前仅 softmax）")
    pa.add_argument(
        "--training_epochs",
        type=int,
        default=50,
        help="warmup 阶段（合并前 K 个 bug）的 epoch 数",
    )
    pa.add_argument("--warmup", type=int, default=10, help="前 K 个版本合并 warmup")
    pa.add_argument("--incr_epochs", type=int, default=1, help="每个新 bug 增量训练 epoch 数")
    pa.add_argument("--v_end", type=int, default=None, help="流式结束版本（默认该项目最大）")
    pa.add_argument("--dump_step", type=int, default=10, help="写入 config 的 DEEPFL_DUMP_STEP")
    pa.add_argument("--gpu_mem", type=float, default=0.2, help="GPU 显存占比（上限 0.2，超出会自动截断）")
    pa.add_argument("--use_ewc", type=_to_bool, default=False, help="是否启用 EWC（仅增量阶段，默认关）")
    pa.add_argument("--ewc_lambda", type=float, default=10.0, help="EWC 惩罚系数")
    pa.add_argument("--ewc_gamma", type=float, default=0.9, help="Online EWC 衰减系数")
    _opt_def = str(os.environ.get("DEEPFL_OPTIMIZER", "adam")).lower()
    if _opt_def not in ("adam", "sgd"):
        _opt_def = "adam"
    _l2_def = str(os.environ.get("DEEPFL_USE_L2", "true")).lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    )
    pa.add_argument(
        "--optimizer",
        choices=("adam", "sgd"),
        default=_opt_def,
        help="与 main / config.create_optimizer 一致；在 import config 前写入 DEEPFL_OPTIMIZER",
    )
    pa.add_argument(
        "--use_l2",
        type=_to_bool,
        default=_l2_def,
        help="是否在损失中加入 L2 正则（fc 分支与 fc_based 一致：无正则项 collection 时不加）",
    )
    args = pa.parse_args()
    args.gpu_mem = min(max(float(args.gpu_mem), 1e-6), 0.2)

    supported = ("mlp", "mlp2", "mlp_dfl_1", "mlp_dfl_2", "rnn", "birnn")
    if args.model not in supported:
        print("continual_stream_train: model 必须是: %s" % " ".join(supported), file=sys.stderr)
        sys.exit(1)
    if args.loss != "softmax":
        print("continual_stream_train: 当前仅支持 loss=softmax", file=sys.stderr)
        sys.exit(1)
    if args.tech != "DeepFL_CL":
        print("continual_stream_train: 为避免数据协议泄漏，tech 仅允许 DeepFL_CL", file=sys.stderr)
        sys.exit(1)

    tech = args.tech
    sub = args.subject
    if tech not in TECH_NAMES:
        print("unknown tech", tech, file=sys.stderr)
        sys.exit(1)
    v_end = args.v_end if args.v_end is not None else ut.get_max_ver(sub)
    warmup = args.warmup
    if warmup < 1 or warmup >= v_end:
        print("warmup 需满足 1 <= warmup < v_end", file=sys.stderr)
        sys.exit(1)

    ut.apply_deepfl_env_from_args(args)
    import config as cfg  # noqa: F401 — 触发根据 DEEPFL_* 加载全局配置

    lr_for_tag = float(cfg.learning_rate)
    run_tag = "ue%d_lr%.4f_el%.1f_eg%.2f_wup%d_te%d_incr%d" % (
        1 if args.use_ewc else 0,
        lr_for_tag,
        args.ewc_lambda,
        args.ewc_gamma,
        warmup,
        args.training_epochs,
        args.incr_epochs,
    )
    if args.optimizer != "adam" or not args.use_l2:
        run_tag += "_%s_l2%d" % (args.optimizer, 1 if args.use_l2 else 0)
    if args.out_dir is None:
        args.out_dir = os.path.join("result_continual", args.model, sub, run_tag)

    loss_idx = LOSSES.index(args.loss)
    n_input = FEATURE_SIZE[TECH_NAMES.index(tech)]
    n_hidden = n_input

    out_root = args.out_dir
    os.makedirs(out_root, exist_ok=True)
    rank_summary_lines = []

    tf.reset_default_graph()
    if args.model == "mlp":
        graph = build_mlp_graph(n_input, n_hidden, loss_idx)
    elif args.model == "mlp2":
        graph = build_mlp2_graph(n_input, n_hidden, loss_idx)
    elif args.model == "mlp_dfl_1":
        graph = build_fc_stream_graph(loss_idx, "fc_based_1", "fc_2_layers")
    elif args.model == "mlp_dfl_2":
        graph = build_fc_stream_graph(loss_idx, "fc_based_2", "mutation_spec_first")
    elif args.model == "rnn":
        graph = build_seq_stream_graph("rnn", loss_idx, cfg.featureDistribution)
    elif args.model == "birnn":
        graph = build_seq_stream_graph("birnn", loss_idx, cfg.featureDistribution)
    else:
        sys.exit(1)
    ewc_state = build_ewc_state(graph, args.ewc_lambda) if args.use_ewc else None
    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)
    sess_cfg = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        intra_op_parallelism_threads=0,
        inter_op_parallelism_threads=0,
    )
    sess = tf.Session(config=sess_cfg)
    sess.run(init)

    # ---------- Warmup：1..warmup ----------
    wx, wy, wg = merge_train_bugs(args.data_root, tech, sub, 1, warmup)
    train_epochs(sess, graph, wx, wy, wg, args.training_epochs, desc="warmup")
    ewc_stats = None
    if args.use_ewc:
        omega0 = [np.zeros(v.shape.as_list(), dtype=np.float32) for v in ewc_state["vars"]]
        omega0, star0 = update_ewc_stats(sess, graph, ewc_state, wx, wy, wg, omega0, args.ewc_gamma)
        ewc_stats = {"omega": omega0, "star": star0}

    # 刚结束 warmup 时，对每个已见 bug 的「即时」Top1（用于 BWT 的 R_i,i，i<=warmup）
    acc_right_after = {}
    for v in tqdm(range(1, warmup + 1), desc="eval_after_warmup"):
        _, _, _, txi, tyl = load_one_bug(args.data_root, tech, sub, v)
        acc_right_after[v] = eval_bug(sess, graph, txi, tyl)

    # ---------- 增量：warmup+1 .. v_end ----------
    pre_before_incr = {}
    for v in tqdm(range(warmup + 1, v_end + 1), desc="incremental"):
        txi, tyl, _, _, _ = load_one_bug(args.data_root, tech, sub, v)
        # 用「尚未用本 bug 的 Train 做本步增量」的模型，对 v 的 Test 打分（仅写入汇总文件）
        pre = eval_bug(sess, graph, txi, tyl)
        pre_before_incr[v] = pre
        if pre is not None:
            rank_summary_lines.append(
                "PRE\t%d\t%s"
                % (v, " ".join(str(s) for s in pre["scores"]))
            )

        ti, tl, tg, _, _ = load_one_bug(args.data_root, tech, sub, v)
        train_epochs(
            sess,
            graph,
            ti,
            tl,
            tg,
            args.incr_epochs,
            desc="incr v=%d" % v,
            ewc_state=ewc_state,
            ewc_stats=ewc_stats,
        )
        if args.use_ewc:
            new_omega, new_star = update_ewc_stats(
                sess, graph, ewc_state, ti, tl, tg, ewc_stats["omega"], args.ewc_gamma
            )
            ewc_stats["omega"] = new_omega
            ewc_stats["star"] = new_star

        acc_right_after[v] = eval_bug(sess, graph, txi, tyl)

    # ---------- 最终：用最终模型评估所有已见 bug ----------
    final_eval = {}
    for v in tqdm(range(1, v_end + 1), desc="final_eval"):
        _, _, _, txi, tyl = load_one_bug(args.data_root, tech, sub, v)
        ev = eval_bug(sess, graph, txi, tyl)
        final_eval[v] = ev
        if ev is not None:
            rank_summary_lines.append(
                "FINAL\t%d\t%s"
                % (v, " ".join(str(s) for s in ev["scores"]))
            )

    # counting_metrics
    # 每个 bug v：用与 rank_parser 一致的 1-based min_rank；Top1/3/5 命中 <=> min_rank<=1/3/5
    seen = list(range(1, v_end + 1))

    def _hit_topk(ev, k):
        """无 Test、无 fault 正例(min<0) 视为未命中 0（与「只对有效 bug 取均值」的 final_acc_top1_t 不同）。"""
        if ev is None or ev["min"] < 0:
            return 0.0
        return 1.0 if ev["min"] <= float(k) else 0.0

    final_top1_list = [final_eval[v]["top1"] for v in seen if final_eval[v] and final_eval[v]["min"] >= 0]
    final_top3_list = [final_eval[v]["top3"] for v in seen if final_eval[v] and final_eval[v]["min"] >= 0]
    final_top5_list = [final_eval[v]["top5"] for v in seen if final_eval[v] and final_eval[v]["min"] >= 0]

    # 仅对「Test 上至少有一个 fault 标注」的 bug 取均值（分母 < v_end 时会高于「全体 bug」均值）
    final_acc_top1_t = float(np.mean(final_top1_list)) if final_top1_list else 0.0
    final_acc_top3_t = float(np.mean(final_top3_list)) if final_top3_list else 0.0
    final_acc_top5_t = float(np.mean(final_top5_list)) if final_top5_list else 0.0

    # warmup 之后每个新 bug：先 PRE 评 v 的 Test、再训 v 的 Train；此处统计 PRE 的 Topk 命中率，
    # 分母固定为 (v_end - warmup)，无 Test / 无 fault 正例计 0（例如流式到第 14 个增量 id 时累计为 hits/4）。
    seen_no_warmup = [v for v in seen if v > warmup]
    incr_pre_acc_top1 = (
        float(np.mean([_hit_topk(pre_before_incr.get(v), 1) for v in seen_no_warmup]))
        if seen_no_warmup
        else 0.0
    )
    incr_pre_acc_top3 = (
        float(np.mean([_hit_topk(pre_before_incr.get(v), 3) for v in seen_no_warmup]))
        if seen_no_warmup
        else 0.0
    )
    incr_pre_acc_top5 = (
        float(np.mean([_hit_topk(pre_before_incr.get(v), 5) for v in seen_no_warmup]))
        if seen_no_warmup
        else 0.0
    )

    bwt_terms = []
    bwt_terms_top5 = []
    for v in seen:
        fe = final_eval[v]
        ae = acc_right_after[v]
        if not fe or fe["min"] < 0 or not ae or ae["min"] < 0:
            continue
        bwt_terms.append(fe["top1"] - ae["top1"])
        bwt_terms_top5.append(fe["top5"] - ae["top5"])
    final_bwt_t = float(np.mean(bwt_terms)) if bwt_terms else 0.0
    final_bwt_t5 = float(np.mean(bwt_terms_top5)) if bwt_terms_top5 else 0.0

    agg = ut.aggregate_official_style([final_eval[v] for v in seen])
    lines = [
        "subject=%s tech=%s model=%s loss=softmax" % (sub, tech, args.model),
        "warmup=%d training_epochs(warmup)=%d incr_epochs=%d v_end=%d"
        % (warmup, args.training_epochs, args.incr_epochs, v_end),
        "optimizer=%s use_l2=%s" % (args.optimizer, str(args.use_l2)),
        "use_ewc=%s ewc_lambda=%.3f ewc_gamma=%.3f"
        % (str(args.use_ewc), float(args.ewc_lambda), float(args.ewc_gamma)),
        "final_acc_top1_t (mean over bugs with Test fault only): %.4f" % final_acc_top1_t,
        "final_acc_top3_t (mean over bugs with Test fault only): %.4f" % final_acc_top3_t,
        "final_acc_top5_t (mean over bugs with Test fault only): %.4f" % final_acc_top5_t,
        "incr_pre_acc_top1 (v>warmup, PRE-before-train, denom=v_end-warmup, no-fault=0): %.4f"
        % incr_pre_acc_top1,
        "incr_pre_acc_top3 (v>warmup, PRE-before-train, denom=v_end-warmup, no-fault=0): %.4f"
        % incr_pre_acc_top3,
        "incr_pre_acc_top5 (v>warmup, PRE-before-train, denom=v_end-warmup, no-fault=0): %.4f"
        % incr_pre_acc_top5,
        "final_bwt_t (Top1, mean_v acc_final[v]-acc_after_v[v]): %.4f" % final_bwt_t,
        "final_bwt_t5 (Top5, mean_v acc_final[v]-acc_after_v[v]): %.4f" % final_bwt_t5,
    ]
    if agg:
        lines.append(
            "final_top1/top3/top5/mfr/mar (final model, paper-style on all seen): %d %d %d %s %s"
            % (agg[0], agg[1], agg[2], agg[3], agg[4])
        )

    # result_saving
    rep = "\n".join(lines) + "\n"
    with open(os.path.join(out_root, "continual_metrics.txt"), "w") as f:
        f.write(rep)

    sum_path = os.path.join(out_root, "rank_scores_all_one_line.txt")
    header = (
        "# subject=%s tech=%s model=%s\n"
        "# 数据行：制表符分隔三列 — 类型、版本号 v、分数列（多个分数用空格分隔，与 DeepFL/%s/v/Test.csv 行序一致）\n"
        "# PRE ：对 bug v，在「用 v 的 Train 做本步增量训练之前」的模型对 v 的 Test 的 fault 概率（softmax 第 0 维）。\n"
        "# FINAL：流式全部结束后，最终模型对 v 的 Test 的同上。\n"
        % (sub, tech, args.model, sub)
    )
    with open(sum_path, "w") as f:
        f.write(header)
        f.write("\n".join(rank_summary_lines) + "\n")

    sess.close()


if __name__ == "__main__":
    main()
