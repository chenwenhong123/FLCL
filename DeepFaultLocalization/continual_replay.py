from __future__ import print_function

import argparse
import math
import os
import sys

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import continual_stream_train as cst
import input
import utils as ut
from utils import tqdm_compat as tqdm


class ReplayBuffer(object):
    """Fixed-size reservoir replay buffer with optional stored logits."""

    def __init__(self, capacity, feature_dim):
        self.capacity = int(max(1, capacity))
        self.feature_dim = int(feature_dim)
        self.x = np.zeros((self.capacity, self.feature_dim), dtype=np.float32)
        self.y = np.zeros((self.capacity, 2), dtype=np.float32)
        self.g = np.zeros((self.capacity, 1), dtype=np.int32)
        self.logits = np.zeros((self.capacity, 2), dtype=np.float32)
        self.size = 0
        self.seen = 0

    def add_batch(self, bx, by, bg, blogits):
        for i in range(bx.shape[0]):
            self._add_one(bx[i], by[i], bg[i], blogits[i])

    def _add_one(self, x, y, g, logits):
        self.seen += 1
        if self.size < self.capacity:
            idx = self.size
            self.size += 1
        else:
            j = np.random.randint(0, self.seen)
            if j >= self.capacity:
                return
            idx = j
        self.x[idx] = x
        self.y[idx] = y
        self.g[idx] = g
        self.logits[idx] = logits

    def sample(self, n):
        if self.size == 0:
            return None
        n = int(max(1, min(n, self.size)))
        idx = np.random.choice(self.size, size=n, replace=False)
        return self.x[idx], self.y[idx], self.g[idx], self.logits[idx]


def _train_feed_dict(graph, bx, by, bg, keep_prob):
    kind = graph["kind"]
    if kind in ("mlp", "mlp2"):
        return {
            graph["x"]: bx,
            graph["y"]: by,
            graph["g"]: bg,
            graph["keep_prob"]: keep_prob,
        }
    if kind == "fc":
        fd = cst._feed_fc(graph, bx)
        fd[graph["y"]] = by
        fd[graph["g"]] = bg
        fd[graph["keep_prob"]] = keep_prob
        fd[graph["is_training"]] = True
        return fd
    if kind in ("rnn", "birnn"):
        bx2 = ut.fill_matrix_by_feature_dist(bx, graph["feat_dist"])
        return {
            graph["x"]: bx2,
            graph["y"]: by,
            graph["g"]: bg,
            graph["keep_prob"]: keep_prob,
        }
    raise ValueError("unknown graph kind %s" % kind)


def _predict_logits(sess, graph, bx):
    kind = graph["kind"]
    zeros_y = np.zeros((bx.shape[0], 2), dtype=np.float32)
    zeros_g = np.zeros((bx.shape[0], 1), dtype=np.int32)
    if kind in ("mlp", "mlp2"):
        fd = {
            graph["x"]: bx,
            graph["y"]: zeros_y,
            graph["g"]: zeros_g,
            graph["keep_prob"]: 1.0,
        }
    elif kind == "fc":
        fd = cst._feed_fc(graph, bx)
        fd[graph["y"]] = zeros_y
        fd[graph["g"]] = zeros_g
        fd[graph["keep_prob"]] = 1.0
        fd[graph["is_training"]] = False
    elif kind in ("rnn", "birnn"):
        bx2 = ut.fill_matrix_by_feature_dist(bx, graph["feat_dist"])
        fd = {
            graph["x"]: bx2,
            graph["y"]: zeros_y,
            graph["g"]: zeros_g,
            graph["keep_prob"]: 1.0,
        }
    else:
        raise ValueError("unknown graph kind %s" % kind)
    return sess.run(graph["pred"], feed_dict=fd)


def _build_derpp_replay_op(graph, lr):
    teacher_logits = tf.placeholder(tf.float32, [None, 2], name="teacher_logits_replay")
    temperature = tf.placeholder(tf.float32, shape=(), name="derpp_temperature")
    replay_beta_ph = tf.placeholder(tf.float32, shape=(), name="replay_beta_weight")
    distill_alpha_ph = tf.placeholder(tf.float32, shape=(), name="distill_alpha_weight")

    sup_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=graph["pred"], labels=graph["y"]))
    teacher_prob = tf.nn.softmax(teacher_logits / temperature)
    student_logit_scaled = graph["pred"] / temperature
    # DER++ style KD: CE(teacher_prob, student/T) * T^2
    kd_ce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=teacher_prob, logits=student_logit_scaled)
    distill_loss = tf.reduce_mean(kd_ce) * tf.square(temperature)
    replay_loss = replay_beta_ph * sup_loss + distill_alpha_ph * distill_loss

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        replay_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(replay_loss)
    return teacher_logits, temperature, replay_beta_ph, distill_alpha_ph, replay_op


def _train_incremental_with_replay(
    sess,
    graph,
    train_x,
    train_y,
    train_g,
    epochs,
    use_replay,
    replay_buffer,
    replay_per_step,
    replay_beta,
    distill_alpha,
    temperature,
    derpp_teacher_ph,
    derpp_temp_ph,
    replay_beta_ph,
    distill_alpha_ph,
    derpp_replay_op,
    use_ewc,
    ewc_state,
    ewc_stats,
    desc="incr",
):
    n = train_x.shape[0]
    ds = input.DataSet(train_x, train_y, train_g)
    graph["ref_holder"]["ds"].train = ds
    total_batch = max(1, int(math.ceil(n / float(cst.BATCH_SIZE))))
    total_steps = max(0, epochs * total_batch)
    pbar = tqdm(total=total_steps, desc=desc, leave=False)
    try:
        for _ in range(epochs):
            ds._index_in_epoch = 0
            ds._epochs_completed = 0
            for __ in range(total_batch):
                bx, by, bg = ds.next_batch(cst.BATCH_SIZE)

                # Cache logits before learning current batch; used as DER++ targets in replay buffer.
                before_logits = _predict_logits(sess, graph, bx)

                fd_cur = _train_feed_dict(graph, bx, by, bg, cst.DROPOUT_RATE)
                if use_ewc and ewc_state is not None and ewc_stats is not None:
                    fd_cur.update(cst._ewc_feed(ewc_state, ewc_stats["omega"], ewc_stats["star"]))
                    sess.run(ewc_state["optimizer_ewc"], feed_dict=fd_cur)
                else:
                    sess.run(graph["optimizer"], feed_dict=fd_cur)

                if use_replay and replay_buffer is not None and replay_buffer.size > 0:
                    for _j in range(int(max(1, replay_per_step))):
                        rb = replay_buffer.sample(cst.BATCH_SIZE)
                        if rb is None:
                            continue
                        rx, ry, rg, rlogits = rb
                        if replay_beta > 0.0 or distill_alpha > 0.0:
                            # Replay step with true weighted loss, not "extra steps as weight proxy".
                            fd_rep = _train_feed_dict(graph, rx, ry, rg, cst.DROPOUT_RATE)
                            fd_rep[derpp_teacher_ph] = rlogits
                            fd_rep[derpp_temp_ph] = float(temperature)
                            fd_rep[replay_beta_ph] = float(replay_beta)
                            fd_rep[distill_alpha_ph] = float(distill_alpha)
                            sess.run(derpp_replay_op, feed_dict=fd_rep)

                if use_replay and replay_buffer is not None:
                    replay_buffer.add_batch(bx, by, bg, before_logits)
                pbar.update(1)
    finally:
        pbar.close()


def _to_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def main():
    pa = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="流式持续学习 + replay + DER++（增量阶段启用）。",
    )
    pa.add_argument("subject", help="项目: Chart|Lang|Math|Time|Closure|Mockito")
    pa.add_argument("model", help="mlp | mlp2 | mlp_dfl_1 | mlp_dfl_2 | rnn | birnn")
    pa.add_argument("--data_root", default=".", help="数据父目录（含 <tech>/<subject>/<ver>/）")
    pa.add_argument(
        "--out_dir",
        default=None,
        help="输出根目录；省略则为 result_continual/<model>/<subject>/rps*_da*_wup*_te*_incr*_v*_replay/",
    )
    pa.add_argument("--tech", default="DeepFL_CL", help="数据与输出路径中的技术子目录名（持续学习固定使用 DeepFL_CL）")
    pa.add_argument("--loss", default="softmax", help="损失函数名（当前仅 softmax）")
    pa.add_argument("--training_epochs", type=int, default=50, help="warmup 阶段（合并前 K 个 bug）的 epoch 数")
    pa.add_argument("--warmup", type=int, default=10, help="前 K 个版本合并 warmup")
    pa.add_argument("--incr_epochs", type=int, default=1, help="每个新 bug 增量训练 epoch 数")
    pa.add_argument("--v_end", type=int, default=None, help="流式结束版本（默认该项目最大）")
    pa.add_argument("--dump_step", type=int, default=10, help="写入 config 的 DEEPFL_DUMP_STEP")
    pa.add_argument("--gpu_mem", type=float, default=0.2, help="GPU 显存占比（上限 0.2，超出会自动截断）")
    pa.add_argument("--use_replay", type=_to_bool, default=True, help="是否启用 replay+DER++（默认开）")
    pa.add_argument("--use_ewc", type=_to_bool, default=False, help="是否启用 EWC（默认关）")
    pa.add_argument("--ewc_lambda", type=float, default=10.0, help="EWC 惩罚系数")
    pa.add_argument("--ewc_gamma", type=float, default=0.9, help="Online EWC 衰减系数")
    pa.add_argument("--replay_size", type=int, default=20, help="回放缓冲区容量（样本条数）")
    pa.add_argument("--replay_per_step", type=int, default=2, help="每个增量 batch 追加的回放批次数")
    pa.add_argument("--replay_beta", type=float, default=1.0, help="DER++ 监督回放项强度（以步数近似）")
    pa.add_argument("--distill_alpha", type=float, default=0.5, help="DER++ 蒸馏项强度（以步数近似）")
    pa.add_argument("--temperature", type=float, default=2.0, help="知识蒸馏温度系数")
    args = pa.parse_args()
    args.gpu_mem = min(max(float(args.gpu_mem), 1e-6), 0.2)

    supported = ("mlp", "mlp2", "mlp_dfl_1", "mlp_dfl_2", "rnn", "birnn")
    if args.model not in supported:
        print("continual_replay: model 必须是: %s" % " ".join(supported), file=sys.stderr)
        sys.exit(1)
    if args.loss != "softmax":
        print("continual_replay: 当前仅支持 loss=softmax", file=sys.stderr)
        sys.exit(1)
    if args.tech != "DeepFL_CL":
        print("continual_replay: 为避免数据协议泄漏，tech 仅允许 DeepFL_CL", file=sys.stderr)
        sys.exit(1)

    tech = args.tech
    sub = args.subject
    if tech not in cst.TECH_NAMES:
        print("unknown tech", tech, file=sys.stderr)
        sys.exit(1)
    v_end = args.v_end if args.v_end is not None else ut.get_max_ver(sub)
    warmup = args.warmup
    if warmup < 1 or warmup >= v_end:
        print("warmup 需满足 1 <= warmup < v_end", file=sys.stderr)
        sys.exit(1)

    ut.apply_deepfl_env_from_args(args)
    import config as cfg  # noqa: F401

    lr_for_tag = float(cfg.learning_rate)
    run_tag = "ur%d_ue%d_lr%.4f_rps%d_da%.2f_el%.1f_eg%.2f_wup%d_te%d_incr%d" % (
        1 if args.use_replay else 0,
        1 if args.use_ewc else 0,
        lr_for_tag,
        args.replay_per_step,
        args.distill_alpha,
        args.ewc_lambda,
        args.ewc_gamma,
        warmup,
        args.training_epochs,
        args.incr_epochs,
    )
    if args.out_dir is None:
        args.out_dir = os.path.join("result_continual", args.model, sub, run_tag)

    loss_idx = cst.LOSSES.index(args.loss)
    n_input = cst.FEATURE_SIZE[cst.TECH_NAMES.index(tech)]
    n_hidden = n_input

    out_root = args.out_dir
    os.makedirs(out_root, exist_ok=True)
    rank_summary_lines = []

    tf.reset_default_graph()
    if args.model == "mlp":
        graph = cst.build_mlp_graph(n_input, n_hidden, loss_idx)
    elif args.model == "mlp2":
        graph = cst.build_mlp2_graph(n_input, n_hidden, loss_idx)
    elif args.model == "mlp_dfl_1":
        graph = cst.build_fc_stream_graph(loss_idx, "fc_based_1", "fc_2_layers")
    elif args.model == "mlp_dfl_2":
        graph = cst.build_fc_stream_graph(loss_idx, "fc_based_2", "mutation_spec_first")
    elif args.model == "rnn":
        graph = cst.build_seq_stream_graph("rnn", loss_idx, cfg.featureDistribution)
    elif args.model == "birnn":
        graph = cst.build_seq_stream_graph("birnn", loss_idx, cfg.featureDistribution)
    else:
        sys.exit(1)
    ewc_state = cst.build_ewc_state(graph, args.ewc_lambda) if args.use_ewc else None

    derpp_teacher_ph, derpp_temp_ph, replay_beta_ph, distill_alpha_ph, derpp_replay_op = _build_derpp_replay_op(
        graph, float(graph["learning_rate"])
    )

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

    replay_buffer = ReplayBuffer(args.replay_size, n_input) if args.use_replay else None

    # ---------- Warmup：1..warmup ----------
    wx, wy, wg = cst.merge_train_bugs(args.data_root, tech, sub, 1, warmup)
    cst.train_epochs(sess, graph, wx, wy, wg, args.training_epochs, desc="warmup")
    ewc_stats = None
    if args.use_ewc:
        omega0 = [np.zeros(v.shape.as_list(), dtype=np.float32) for v in ewc_state["vars"]]
        omega0, star0 = cst.update_ewc_stats(sess, graph, ewc_state, wx, wy, wg, omega0, args.ewc_gamma)
        ewc_stats = {"omega": omega0, "star": star0}

    # 刚结束 warmup 时，对每个已见 bug 的「即时」Topk（用于 BWT 的 R_i,i，i<=warmup）
    acc_right_after = {}
    for v in tqdm(range(1, warmup + 1), desc="eval_after_warmup"):
        _, _, _, txi, tyl = cst.load_one_bug(args.data_root, tech, sub, v)
        acc_right_after[v] = cst.eval_bug(sess, graph, txi, tyl)

    # ---------- 增量：warmup+1 .. v_end ----------
    pre_before_incr = {}
    for v in tqdm(range(warmup + 1, v_end + 1), desc="incremental"):
        txi, tyl, _, _, _ = cst.load_one_bug(args.data_root, tech, sub, v)
        pre = cst.eval_bug(sess, graph, txi, tyl)
        pre_before_incr[v] = pre
        if pre is not None:
            rank_summary_lines.append("PRE\t%d\t%s" % (v, " ".join(str(s) for s in pre["scores"])))

        ti, tl, tg, _, _ = cst.load_one_bug(args.data_root, tech, sub, v)
        _train_incremental_with_replay(
            sess=sess,
            graph=graph,
            train_x=ti,
            train_y=tl,
            train_g=tg,
            epochs=args.incr_epochs,
            use_replay=args.use_replay,
            replay_buffer=replay_buffer,
            replay_per_step=args.replay_per_step,
            replay_beta=args.replay_beta,
            distill_alpha=args.distill_alpha,
            temperature=args.temperature,
            derpp_teacher_ph=derpp_teacher_ph,
            derpp_temp_ph=derpp_temp_ph,
            replay_beta_ph=replay_beta_ph,
            distill_alpha_ph=distill_alpha_ph,
            derpp_replay_op=derpp_replay_op,
            use_ewc=args.use_ewc,
            ewc_state=ewc_state,
            ewc_stats=ewc_stats,
            desc="incr v=%d" % v,
        )
        if args.use_ewc:
            new_omega, new_star = cst.update_ewc_stats(
                sess, graph, ewc_state, ti, tl, tg, ewc_stats["omega"], args.ewc_gamma
            )
            ewc_stats["omega"] = new_omega
            ewc_stats["star"] = new_star

        acc_right_after[v] = cst.eval_bug(sess, graph, txi, tyl)

    # ---------- 最终：用最终模型评估所有已见 bug ----------
    final_eval = {}
    for v in tqdm(range(1, v_end + 1), desc="final_eval"):
        _, _, _, txi, tyl = cst.load_one_bug(args.data_root, tech, sub, v)
        ev = cst.eval_bug(sess, graph, txi, tyl)
        final_eval[v] = ev
        if ev is not None:
            rank_summary_lines.append("FINAL\t%d\t%s" % (v, " ".join(str(s) for s in ev["scores"])))

    seen = list(range(1, v_end + 1))

    def _hit_topk(ev, k):
        if ev is None or ev["min"] < 0:
            return 0.0
        return 1.0 if ev["min"] <= float(k) else 0.0

    final_top1_list = [final_eval[v]["top1"] for v in seen if final_eval[v] and final_eval[v]["min"] >= 0]
    final_top3_list = [final_eval[v]["top3"] for v in seen if final_eval[v] and final_eval[v]["min"] >= 0]
    final_top5_list = [final_eval[v]["top5"] for v in seen if final_eval[v] and final_eval[v]["min"] >= 0]
    final_acc_top1_t = float(np.mean(final_top1_list)) if final_top1_list else 0.0
    final_acc_top3_t = float(np.mean(final_top3_list)) if final_top3_list else 0.0
    final_acc_top5_t = float(np.mean(final_top5_list)) if final_top5_list else 0.0

    seen_no_warmup = [v for v in seen if v > warmup]
    incr_pre_acc_top1 = (
        float(np.mean([_hit_topk(pre_before_incr.get(v), 1) for v in seen_no_warmup])) if seen_no_warmup else 0.0
    )
    incr_pre_acc_top3 = (
        float(np.mean([_hit_topk(pre_before_incr.get(v), 3) for v in seen_no_warmup])) if seen_no_warmup else 0.0
    )
    incr_pre_acc_top5 = (
        float(np.mean([_hit_topk(pre_before_incr.get(v), 5) for v in seen_no_warmup])) if seen_no_warmup else 0.0
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
        "use_replay=%s use_ewc=%s replay_size=%d replay_per_step=%d replay_beta=%.3f distill_alpha=%.3f temperature=%.3f ewc_lambda=%.3f ewc_gamma=%.3f"
        % (
            str(args.use_replay),
            str(args.use_ewc),
            int(args.replay_size),
            int(args.replay_per_step),
            float(args.replay_beta),
            float(args.distill_alpha),
            float(args.temperature),
            float(args.ewc_lambda),
            float(args.ewc_gamma),
        ),
        "final_acc_top1_t (mean over bugs with Test fault only): %.4f" % final_acc_top1_t,
        "final_acc_top3_t (mean over bugs with Test fault only): %.4f" % final_acc_top3_t,
        "final_acc_top5_t (mean over bugs with Test fault only): %.4f" % final_acc_top5_t,
        "incr_pre_acc_top1 (v>warmup, PRE-before-train, denom=v_end-warmup, no-fault=0): %.4f" % incr_pre_acc_top1,
        "incr_pre_acc_top3 (v>warmup, PRE-before-train, denom=v_end-warmup, no-fault=0): %.4f" % incr_pre_acc_top3,
        "incr_pre_acc_top5 (v>warmup, PRE-before-train, denom=v_end-warmup, no-fault=0): %.4f" % incr_pre_acc_top5,
        "final_bwt_t (Top1, mean_v acc_final[v]-acc_after_v[v]): %.4f" % final_bwt_t,
        "final_bwt_t5 (Top5, mean_v acc_final[v]-acc_after_v[v]): %.4f" % final_bwt_t5,
    ]
    if agg:
        lines.append(
            "final_top1/top3/top5/mfr/mar (final model, paper-style on all seen): %d %d %d %s %s"
            % (agg[0], agg[1], agg[2], agg[3], agg[4])
        )

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
