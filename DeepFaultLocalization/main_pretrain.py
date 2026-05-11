# -*- coding: utf-8 -*-
"""
全局预训练 + 单 bug 微调入口（独立于 main.py / continual_stream_train.py）。

1) 若 checkpoint 不存在：合并多个项目在 ver=1..v_max 上的 Train，训练单层 MLP（softmax），保存到 --ckpt_dir/global_mlp_pretrain*。
2) 若已存在：跳过预训练，直接 restore。
3) 在 --target_sub / --target_ver 上读取该 bug 的 Train/Test，继续训练并写 susp（与 main 输出目录习惯一致）。
"""
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import input
import utils as ut

SOFTMAX_LOSS = 1
DEFAULT_SUBJECTS = ("Chart", "Lang", "Math", "Time", "Closure", "Mockito")


def _parse_subjects(s):
    if not s or not str(s).strip():
        return list(DEFAULT_SUBJECTS)
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _ckpt_exists(prefix):
    return os.path.isfile(prefix + ".index") or os.path.isfile(prefix + ".meta")


def _merge_global_pretrain_pool(data_root, tech, subjects, v_max):
    from continual_stream_train import TRAIN_FILE, load_one_bug

    xs, ys, gs = [], [], []
    gid_off = 0
    used = []
    for sub in subjects:
        for ver in range(1, int(v_max) + 1):
            tp = os.path.join(data_root, tech, sub, str(ver), TRAIN_FILE)
            if not os.path.isfile(tp):
                continue
            try:
                ti, tl, g, _, _ = load_one_bug(data_root, tech, sub, ver)
            except (IOError, OSError, ValueError):
                continue
            if ti is None or ti.shape[0] == 0:
                continue
            xs.append(ti)
            ys.append(tl)
            gmax = int(g.max()) if g.size else 0
            gs.append(g + gid_off)
            gid_off += gmax + 1
            used.append((sub, ver))
    if not xs:
        return None, None, None, used
    return np.vstack(xs), np.vstack(ys), np.vstack(gs), used


def _reload_config(data_root, tech, out_dir, epochs, dump_step):
    os.environ["DEEPFL_DATA_DIR"] = data_root
    os.environ["DEEPFL_OUT_DIR"] = out_dir or "."
    os.environ["DEEPFL_SUB"] = "Chart"
    os.environ["DEEPFL_VER"] = "1"
    os.environ["DEEPFL_MODEL"] = "mlp"
    os.environ["DEEPFL_TECH"] = tech
    os.environ["DEEPFL_LOSS"] = "softmax"
    os.environ["DEEPFL_EPOCHS"] = str(int(epochs))
    os.environ["DEEPFL_DUMP_STEP"] = str(int(dump_step))
    import importlib

    import config

    importlib.reload(config)


def _build_mlp(n_input, n_hidden, datasets_for_loss):
    from config import L2_value, create_optimizer, dropout_rate, use_l2
    from multilayer_perceptron_one_hidden_layer import multilayer_perceptron

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
    pred = multilayer_perceptron(x, weights, biases, keep_prob)
    reg = (
        (tf.nn.l2_loss(weights["h1"]) + tf.nn.l2_loss(weights["out"])) * L2_value
        if use_l2
        else tf.constant(0.0, dtype=tf.float32)
    )
    cost = ut.loss_func(pred, y, SOFTMAX_LOSS, datasets_for_loss, g)
    optimizer = create_optimizer().minimize(cost + reg)
    return x, y, g, keep_prob, pred, cost, optimizer


def _run_epochs(sess, x, y, g, keep_prob, pred, cost, optimizer, ds, epochs, dropout_p, bs, display_step):
    for epoch in range(int(epochs)):
        avg_cost = 0.0
        total_batch = max(1, int(ds.num_instances / bs))
        for _ in range(total_batch):
            bx, by, bg = ds.next_batch(bs)
            _, c = sess.run(
                [optimizer, cost],
                feed_dict={x: bx, y: by, g: bg, keep_prob: dropout_p},
            )
            avg_cost += c / total_batch
        if epoch % display_step == 0:
            print("epoch", epoch + 1, "cost", "%.9f" % avg_cost, flush=True)


def main():
    pa = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="跨项目 id 1..v_max 全局预训练 checkpoint，再在 target_sub/target_ver 上微调并写 susp。",
    )
    pa.add_argument("--data_root", default=".", help="数据根目录 <tech>/<subject>/<ver>/")
    pa.add_argument("--out_dir", default=".", help="susp 输出根（与 main 的 out 类似）")
    pa.add_argument("--ckpt_dir", default="pretrain_ckpts", help="全局 ckpt 保存目录")
    pa.add_argument("--tech", default="DeepFL_CL", help="技术子目录名")
    pa.add_argument(
        "--subjects",
        default=",".join(DEFAULT_SUBJECTS),
        help="参与预训练的项目，逗号分隔，默认六项目",
    )
    pa.add_argument("--v_max", type=int, default=15, help="每个项目合并 Train 的最大版本号（含）")
    pa.add_argument("--pretrain_epochs", type=int, default=50, help="预训练 epoch 数")
    pa.add_argument("--finetune_epochs", type=int, default=50, help="目标 bug 微调 epoch 数")
    pa.add_argument("--target_sub", default=None, help="微调目标项目，如 Lang")
    pa.add_argument("--target_ver", type=int, default=None, help="微调目标版本号")
    pa.add_argument("--dump_step", type=int, default=10, help="与 main 一致：在 dump_step-1 轮写 susp")
    pa.add_argument("--display_step", type=int, default=2, help="打印间隔")
    pa.add_argument("--gpu_mem", type=float, default=0.8, help="GPU 显存占比上限")
    pa.add_argument("--pretrain_batch_size", type=int, default=2048, help="预训练 batch size")
    pa.add_argument("--finetune_batch_size", type=int, default=2048, help="微调 batch size")
    pa.add_argument("--force_pretrain", action="store_true", help="即使已有 ckpt 也重新预训练并覆盖")
    args = pa.parse_args()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    if args.target_sub is None or args.target_ver is None:
        print("需要 --target_sub 与 --target_ver 以进行微调与写 susp。", file=sys.stderr)
        sys.exit(2)

    subjects = _parse_subjects(args.subjects)
    ckpt_prefix = os.path.join(args.ckpt_dir, "global_mlp_pretrain")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    _reload_config(
        args.data_root,
        args.tech,
        args.out_dir,
        max(args.pretrain_epochs, args.finetune_epochs),
        args.dump_step,
    )
    import importlib

    import multilayer_perceptron_one_hidden_layer as mpl

    importlib.reload(mpl)

    from config import (
        dropout_rate,
        feature,
        train_file,
        train_label_file,
        test_file,
        test_label_file,
        group_dir,
        group_file,
        losses,
    )

    n_input = int(feature)
    n_hidden = n_input

    train_path = os.path.join(
        args.data_root, args.tech, args.target_sub, str(args.target_ver), train_file
    )
    train_label_path = os.path.join(
        args.data_root, args.tech, args.target_sub, str(args.target_ver), train_label_file
    )
    test_path = os.path.join(args.data_root, args.tech, args.target_sub, str(args.target_ver), test_file)
    test_label_path = os.path.join(
        args.data_root, args.tech, args.target_sub, str(args.target_ver), test_label_file
    )
    group_path = os.path.join(
        args.data_root, args.tech, group_dir, args.target_sub, str(args.target_ver), group_file
    )

    datasets = input.read_data_sets(
        train_path, train_label_path, test_path, test_label_path, group_path
    )

    tf.reset_default_graph()
    x, y, g, keep_prob, pred, cost, optimizer = _build_mlp(n_input, n_hidden, datasets)
    init = tf.global_variables_initializer()
    # 分开管理 checkpoint，避免 finetune 保存时清掉 global_mlp_pretrain。
    saver_pretrain = tf.train.Saver(max_to_keep=1)
    saver_finetune = tf.train.Saver(max_to_keep=1)
    gpus = tf.config.list_physical_devices("GPU")
    print("visible_gpus=%d %s" % (len(gpus), [d.name for d in gpus]), flush=True)
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=float(args.gpu_mem),
        allow_growth=True,
    )
    sess_cfg = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        intra_op_parallelism_threads=0,
        inter_op_parallelism_threads=0,
    )
    sess = tf.Session(config=sess_cfg)
    sess.run(init)

    need_pretrain = args.force_pretrain or not _ckpt_exists(ckpt_prefix)
    if need_pretrain:
        px, py, pg, used = _merge_global_pretrain_pool(
            args.data_root, args.tech, subjects, args.v_max
        )
        if px is None or px.shape[0] == 0:
            print("预训练池为空，请检查 data_root/tech/subjects/v_max。", file=sys.stderr)
            sys.exit(1)
        print("pretrain pooled rows=%d bugs=%d" % (px.shape[0], len(used)), flush=True)
        pre_ds = input.DataSet(px, py, pg)
        _run_epochs(
            sess,
            x,
            y,
            g,
            keep_prob,
            pred,
            cost,
            optimizer,
            pre_ds,
            args.pretrain_epochs,
            dropout_rate,
            args.pretrain_batch_size,
            args.display_step,
        )
        saver_pretrain.save(sess, ckpt_prefix)
        print("saved", ckpt_prefix, flush=True)
    else:
        saver_pretrain.restore(sess, ckpt_prefix)
        print("loaded", ckpt_prefix, flush=True)

    susp_dir = os.path.join(args.out_dir, args.target_sub, str(args.target_ver), args.tech)
    if not os.path.exists(susp_dir):
        os.makedirs(susp_dir)
    susp_path = os.path.join(susp_dir, "main_pretrain-" + losses[SOFTMAX_LOSS])

    from config import dump_step

    for epoch in range(args.finetune_epochs):
        avg_cost = 0.0
        total_batch = max(1, int(datasets.train.num_instances / args.finetune_batch_size))
        for _ in range(total_batch):
            bx, by, bg = datasets.train.next_batch(args.finetune_batch_size)
            _, c = sess.run(
                [optimizer, cost],
                feed_dict={x: bx, y: by, g: bg, keep_prob: dropout_rate},
            )
            avg_cost += c / total_batch
        if epoch % args.display_step == 0:
            print("finetune", epoch + 1, "cost", "%.9f" % avg_cost, flush=True)
        if epoch % dump_step == (dump_step - 1):
            res = sess.run(
                tf.nn.softmax(pred),
                feed_dict={
                    x: datasets.test.instances,
                    y: datasets.test.labels,
                    keep_prob: 1.0,
                },
            )
            with open(susp_path + "-" + str(epoch + 1), "w") as f:
                for susp in res[:, 0]:
                    f.write(str(susp) + "\n")

    finetune_dir = os.path.join(args.ckpt_dir, "finetune")
    if not os.path.exists(finetune_dir):
        os.makedirs(finetune_dir)
    saver_finetune.save(
        sess,
        os.path.join(
            finetune_dir,
            "global_mlp_finetune_" + args.target_sub + "_" + str(args.target_ver),
        ),
    )
    print("Optimization Finished!", flush=True)


if __name__ == "__main__":
    main()
