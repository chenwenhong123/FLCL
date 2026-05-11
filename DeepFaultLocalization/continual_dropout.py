from __future__ import print_function

import argparse
import math
import os
import sys

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import continual_replay as cr
import continual_stream_train as cst
import continual_mask as cm
import input
import utils as ut
from utils import tqdm_compat as tqdm


class TaskReplayBuffer(cr.ReplayBuffer):
    """Replay buffer with task id so we can apply task-specific gating/dropout."""

    def __init__(self, capacity, feature_dim):
        super(TaskReplayBuffer, self).__init__(capacity, feature_dim)
        self.tid = np.zeros((self.capacity,), dtype=np.int32)

    def add_batch(self, bx, by, bg, blogits, task_id):
        for i in range(bx.shape[0]):
            self._add_one_with_tid(bx[i], by[i], bg[i], blogits[i], task_id)

    def _add_one_with_tid(self, x, y, g, logits, task_id):
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
        self.tid[idx] = int(task_id)

    def sample(self, n):
        if self.size == 0:
            return None
        n = int(max(1, min(n, self.size)))
        idx = np.random.choice(self.size, size=n, replace=False)
        return self.x[idx], self.y[idx], self.g[idx], self.logits[idx], self.tid[idx]


def _build_agem_ops(graph):
    vars_all = tf.trainable_variables()
    grads = tf.gradients(graph["cost"], vars_all)
    grad_ph = []
    for i, v in enumerate(vars_all):
        grad_ph.append(tf.placeholder(tf.float32, shape=v.shape, name="agem_grad_%d" % i))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        apply_op = tf.train.AdamOptimizer(learning_rate=float(graph["learning_rate"])).apply_gradients(
            list(zip(grad_ph, vars_all))
        )
    return {"vars": vars_all, "grads": grads, "grad_ph": grad_ph, "apply_op": apply_op}


def _flatten_grads(grads):
    parts = []
    for g in grads:
        if g is None:
            continue
        parts.append(g.reshape(-1))
    if not parts:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def _project_gradients(g_cur, g_ref):
    dot = float(np.dot(g_cur, g_ref))
    if dot >= 0.0:
        return g_cur
    denom = float(np.dot(g_ref, g_ref)) + 1e-12
    return g_cur - (dot / denom) * g_ref


def _agem_apply(sess, agem_state, fd, g_vec):
    fd2 = dict(fd)
    off = 0
    for i, v in enumerate(agem_state["vars"]):
        shape = v.shape.as_list()
        n = int(np.prod(shape))
        sl = g_vec[off : off + n].reshape(shape).astype(np.float32)
        fd2[agem_state["grad_ph"][i]] = sl
        off += n
    sess.run(agem_state["apply_op"], feed_dict=fd2)


def _apply_masked_input_dropout(batch_x, mask, dropout_unmask_p):
    """Train-time masked dropout on input features.

    mask is a float vector in [0,1], same length as feature dim (e.g., 226).
    For feature j: p_drop(j) = dropout_unmask_p * (1 - mask(j))
    """
    if mask is None:
        return batch_x
    x_det = batch_x * mask
    if dropout_unmask_p <= 0.0:
        return x_det
    p_drop = dropout_unmask_p * (1.0 - mask)
    # keep ~ Bernoulli(1 - p_drop)
    noise_keep = (np.random.rand(*batch_x.shape) >= p_drop).astype(batch_x.dtype)
    return x_det * noise_keep


def _apply_mask_eval(batch_x, mask):
    if mask is None:
        return batch_x
    return batch_x * mask


def _build_hard_mask_for_task(model, train_x, keep_ratio, feat_dist):
    if model == "birnn":
        hard_step = cm._build_step_mask_from_data(train_x, feat_dist, keep_ratio)
        return cm._expand_step_mask_to_feature_mask(hard_step, feat_dist)
    # mlp_dfl_1 / mlp2: feature-level mask
    return cm._build_task_mask_from_data(train_x, keep_ratio)


def _current_mask_for_eval(task_id, task_masks):
    if task_id in task_masks:
        return task_masks[task_id]
    return None


def _train_incremental_with_replay_dropout(
    sess,
    graph,
    train_x,
    train_y,
    train_g,
    epochs,
    current_task_id,
    task_masks,
    dropout_unmask_p,
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
    use_gem,
    agem_state,
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

                mask_cur = task_masks.get(int(current_task_id), None)
                bx_det = _apply_mask_eval(bx, mask_cur)
                # teacher logits for DER++: use deterministic masked input (no stochastic dropout)
                before_logits = cr._predict_logits(sess, graph, bx_det)

                bx_train = _apply_masked_input_dropout(bx, mask_cur, dropout_unmask_p)
                fd_cur = cr._train_feed_dict(graph, bx_train, by, bg, cst.DROPOUT_RATE)

                # Main update
                did_main_update = False
                if use_gem and use_replay and replay_buffer is not None and replay_buffer.size > 0:
                    rb_ref = replay_buffer.sample(cst.BATCH_SIZE)
                    if rb_ref is not None:
                        rx_ref, ry_ref, rg_ref, _rlogits, tid_ref = rb_ref
                        # Build rx_train row-wise (mask-specific stochastic dropout).
                        # For ref samples: tid_ref varies per example, so use per-row mask.
                        rx_train = rx_ref.copy()
                        for i in range(rx_ref.shape[0]):
                            m = task_masks.get(int(tid_ref[i]), None)
                            rx_train[i] = _apply_masked_input_dropout(rx_ref[i : i + 1], m, dropout_unmask_p)[0]

                        fd_ref = cr._train_feed_dict(graph, rx_train, ry_ref, rg_ref, cst.DROPOUT_RATE)
                        g_cur_list = sess.run(agem_state["grads"], feed_dict=fd_cur)
                        g_ref_list = sess.run(agem_state["grads"], feed_dict=fd_ref)
                        g_cur = _flatten_grads(g_cur_list)
                        g_ref = _flatten_grads(g_ref_list)
                        if g_cur.shape[0] > 0 and g_ref.shape[0] == g_cur.shape[0]:
                            g_proj = _project_gradients(g_cur, g_ref)
                            _agem_apply(sess, agem_state, fd_cur, g_proj)
                            did_main_update = True

                if not did_main_update:
                    if use_ewc and ewc_state is not None and ewc_stats is not None:
                        fd_main = dict(fd_cur)
                        fd_main.update(cst._ewc_feed(ewc_state, ewc_stats["omega"], ewc_stats["star"]))
                        sess.run(ewc_state["optimizer_ewc"], feed_dict=fd_main)
                    else:
                        sess.run(graph["optimizer"], feed_dict=fd_cur)

                # DER++ replay steps (with masked stochastic dropout)
                if use_replay and replay_buffer is not None and replay_buffer.size > 0:
                    for _j in range(int(max(1, replay_per_step))):
                        rb = replay_buffer.sample(cst.BATCH_SIZE)
                        if rb is None:
                            continue
                        rx, ry, rg, rlogits, tid = rb

                        # Build rx_train row-wise
                        rx_train = rx.copy()
                        for i in range(rx.shape[0]):
                            m = task_masks.get(int(tid[i]), None)
                            rx_train[i] = _apply_masked_input_dropout(rx[i : i + 1], m, dropout_unmask_p)[0]

                        if replay_beta > 0.0 or distill_alpha > 0.0:
                            fd_rep = cr._train_feed_dict(graph, rx_train, ry, rg, cst.DROPOUT_RATE)
                            fd_rep[derpp_teacher_ph] = rlogits
                            fd_rep[derpp_temp_ph] = float(temperature)
                            fd_rep[replay_beta_ph] = float(replay_beta)
                            fd_rep[distill_alpha_ph] = float(distill_alpha)
                            sess.run(derpp_replay_op, feed_dict=fd_rep)

                # Add current batch to replay buffer (store deterministic-teacher logits)
                if use_replay and replay_buffer is not None:
                    replay_buffer.add_batch(bx, by, bg, before_logits, current_task_id)

                pbar.update(1)
    finally:
        pbar.close()


def _eval_bug_with_mask_dropout(sess, graph, test_x, test_y, task_id, task_masks, use_mask_dropout):
    if test_x.shape[0] == 0:
        return None
    if not use_mask_dropout:
        return cst.eval_bug(sess, graph, test_x, test_y)
    m = _current_mask_for_eval(int(task_id), task_masks)
    tx = _apply_mask_eval(test_x, m) if m is not None else test_x
    return cst.eval_bug(sess, graph, tx, test_y)


def main():
    pa = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="流式持续学习 + replay + DER++ + (可选)A-GEM + 增量门控Dropout（mlp_dfl_1/mlp2/birnn）。",
    )
    pa.add_argument("subject", help="项目: Chart|Lang|Math|Time|Closure|Mockito")
    pa.add_argument("model", help="mlp_dfl_1 | mlp2 | birnn")
    pa.add_argument("--data_root", default=".", help="数据父目录（含 <tech>/<subject>/<ver>/）")
    pa.add_argument("--out_dir", default=None, help="输出根目录；省略则为 result_continual/<model>/<sub>/<tag>/")
    pa.add_argument("--tech", default="DeepFL_CL", help="数据与输出路径中的技术子目录名（持续学习固定使用 DeepFL_CL）")
    pa.add_argument("--loss", default="softmax", help="损失函数名（当前仅 softmax）")
    pa.add_argument("--training_epochs", type=int, default=50, help="warmup 阶段 epoch 数")
    pa.add_argument("--warmup", type=int, default=10, help="前 K 个版本合并 warmup")
    pa.add_argument("--incr_epochs", type=int, default=1, help="每个新 bug 增量训练 epoch 数")
    pa.add_argument("--v_end", type=int, default=None, help="流式结束版本（默认该项目最大）")
    pa.add_argument("--dump_step", type=int, default=10, help="写入 config 的 DEEPFL_DUMP_STEP")
    pa.add_argument("--gpu_mem", type=float, default=0.2, help="GPU 显存占比（上限 0.2）")

    pa.add_argument("--use_replay", type=cr._to_bool, default=True, help="是否启用 replay+DER++（默认开）")
    pa.add_argument("--use_gem", type=cr._to_bool, default=True, help="是否启用 A-GEM 投影（默认开）")
    pa.add_argument("--use_ewc", type=cr._to_bool, default=False, help="是否启用 EWC（默认关）")
    pa.add_argument("--ewc_lambda", type=float, default=10.0, help="EWC 惩罚系数")
    pa.add_argument("--ewc_gamma", type=float, default=0.9, help="Online EWC 衰减系数")

    pa.add_argument("--replay_size", type=int, default=20, help="回放缓冲区容量（样本条数）")
    pa.add_argument("--replay_per_step", type=int, default=2, help="每个增量 batch 的回放批次数")
    pa.add_argument("--replay_beta", type=float, default=1.0, help="DER++ 监督回放项权重")
    pa.add_argument("--distill_alpha", type=float, default=0.5, help="DER++ 蒸馏项权重")
    pa.add_argument("--temperature", type=float, default=2.0, help="知识蒸馏温度 T")

    pa.add_argument("--use_mask_dropout", type=cr._to_bool, default=True, help="是否启用门控Dropout（默认开）")
    pa.add_argument("--mask_keep_ratio", type=float, default=0.6, help="每任务门控mask保留比例")
    pa.add_argument("--mask_ema", type=float, default=0.9, help="任务mask EMA 平滑系数")
    pa.add_argument("--dropout_unmask_p", type=float, default=0.3, help="对mask未保留部分的输入dropout概率（增量改进）")
    args = pa.parse_args()

    args.gpu_mem = min(max(float(args.gpu_mem), 1e-6), 0.2)
    args.mask_keep_ratio = min(max(float(args.mask_keep_ratio), 1e-3), 1.0)
    args.mask_ema = min(max(float(args.mask_ema), 0.0), 0.999)
    args.dropout_unmask_p = min(max(float(args.dropout_unmask_p), 0.0), 1.0)

    supported = ("mlp_dfl_1", "mlp2", "birnn")
    if args.model not in supported:
        print("continual_dropout: model 必须是: %s" % " ".join(supported), file=sys.stderr)
        sys.exit(1)
    if args.loss != "softmax":
        print("continual_dropout: 当前仅支持 loss=softmax", file=sys.stderr)
        sys.exit(1)
    if args.tech != "DeepFL_CL":
        print("continual_dropout: 为避免数据协议泄漏，tech 仅允许 DeepFL_CL", file=sys.stderr)
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

    # Apply env for config(featureDistribution) — 需在 import config 前写入 DEEPFL_*
    ut.apply_deepfl_env_from_args(args)
    import config as cfg  # noqa: F401

    lr_for_tag = float(cfg.learning_rate)

    run_tag = "ur%d_ug%d_ue%d_um%d_ud%.2f_lr%.4f_mr%.2f_me%.2f_rps%d_da%.2f_el%.1f_eg%.2f_wup%d_te%d_incr%d" % (
        1 if args.use_replay else 0,
        1 if args.use_gem else 0,
        1 if args.use_ewc else 0,
        1 if args.use_mask_dropout else 0,
        args.dropout_unmask_p,
        lr_for_tag,
        args.mask_keep_ratio,
        args.mask_ema,
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
    os.makedirs(args.out_dir, exist_ok=True)

    loss_idx = cst.LOSSES.index(args.loss)
    n_input = cst.FEATURE_SIZE[cst.TECH_NAMES.index(tech)]
    n_hidden = n_input

    tf.reset_default_graph()
    if args.model == "mlp_dfl_1":
        graph = cst.build_fc_stream_graph(loss_idx, "fc_based_1", "fc_2_layers")
    elif args.model == "mlp2":
        graph = cst.build_mlp2_graph(n_input, n_hidden, loss_idx)
    elif args.model == "birnn":
        graph = cst.build_seq_stream_graph("birnn", loss_idx, cfg.featureDistribution)
    else:
        sys.exit(1)

    derpp_teacher_ph, derpp_temp_ph, replay_beta_ph, distill_alpha_ph, derpp_replay_op = cr._build_derpp_replay_op(
        graph, float(graph["learning_rate"])
    )

    ewc_state = cst.build_ewc_state(graph, args.ewc_lambda) if args.use_ewc else None

    agem_state = _build_agem_ops(graph) if args.use_gem else None

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

    replay_buffer = TaskReplayBuffer(args.replay_size, n_input) if args.use_replay else None

    # ---------- Warmup ----------
    wx, wy, wg = cst.merge_train_bugs(args.data_root, tech, sub, 1, warmup)
    cst.train_epochs(sess, graph, wx, wy, wg, args.training_epochs, desc="warmup")

    # Initialize task masks
    task_masks = {}
    if args.use_mask_dropout:
        # In warmup: no meaningful gating, set identity masks
        for v in range(1, warmup + 1):
            task_masks[v] = np.ones((n_input,), dtype=np.float32)
    else:
        task_masks = {}

    prev_mask = np.ones((n_input,), dtype=np.float32) if task_masks else None

    # EWC init stats
    ewc_stats = None
    if args.use_ewc:
        omega0 = [np.zeros(v.shape.as_list(), dtype=np.float32) for v in ewc_state["vars"]]
        omega0, star0 = cst.update_ewc_stats(sess, graph, ewc_state, wx, wy, wg, omega0, args.ewc_gamma)
        ewc_stats = {"omega": omega0, "star": star0}

    acc_right_after = {}
    pre_before_incr = {}
    rank_summary_lines = []

    # warmup-side evaluation on each seen bug (optional: mask ones => identity)
    for v in tqdm(range(1, warmup + 1), desc="eval_after_warmup"):
        _, _, _, txi, tyl = cst.load_one_bug(args.data_root, tech, sub, v)
        acc_right_after[v] = _eval_bug_with_mask_dropout(sess, graph, txi, tyl, v, task_masks, args.use_mask_dropout)

    # ---------- Incremental ----------
    feat_dist = cfg.featureDistribution
    for v in tqdm(range(warmup + 1, v_end + 1), desc="incremental"):
        ti, tl, tg, txi, tyl = cst.load_one_bug(args.data_root, tech, sub, v)

        # Build and update gate mask for this task
        if args.use_mask_dropout:
            hard_mask = _build_hard_mask_for_task(args.model, ti, args.mask_keep_ratio, feat_dist)
            prev_mask = cm._update_task_mask(prev_mask, hard_mask, args.mask_ema)
            task_masks[v] = prev_mask

            pre = _eval_bug_with_mask_dropout(sess, graph, txi, tyl, v, task_masks, True)
        else:
            pre = _eval_bug_with_mask_dropout(sess, graph, txi, tyl, v, task_masks, False)

        pre_before_incr[v] = pre
        if pre is not None:
            rank_summary_lines.append("PRE\t%d\t%s" % (v, " ".join(str(s) for s in pre["scores"])))

        _train_incremental_with_replay_dropout(
            sess=sess,
            graph=graph,
            train_x=ti,
            train_y=tl,
            train_g=tg,
            epochs=args.incr_epochs,
            current_task_id=v,
            task_masks=task_masks,
            dropout_unmask_p=args.dropout_unmask_p,
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
            use_gem=args.use_gem,
            agem_state=agem_state,
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

        acc_right_after[v] = _eval_bug_with_mask_dropout(sess, graph, txi, tyl, v, task_masks, args.use_mask_dropout)

    # ---------- Final evaluation ----------
    final_eval = {}
    for v in tqdm(range(1, v_end + 1), desc="final_eval"):
        _, _, _, txi, tyl = cst.load_one_bug(args.data_root, tech, sub, v)
        ev = _eval_bug_with_mask_dropout(sess, graph, txi, tyl, v, task_masks, args.use_mask_dropout)
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
    incr_pre_acc_top1 = float(np.mean([_hit_topk(pre_before_incr.get(v), 1) for v in seen_no_warmup])) if seen_no_warmup else 0.0
    incr_pre_acc_top3 = float(np.mean([_hit_topk(pre_before_incr.get(v), 3) for v in seen_no_warmup])) if seen_no_warmup else 0.0
    incr_pre_acc_top5 = float(np.mean([_hit_topk(pre_before_incr.get(v), 5) for v in seen_no_warmup])) if seen_no_warmup else 0.0

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
        "warmup=%d training_epochs(warmup)=%d incr_epochs=%d v_end=%d" % (warmup, args.training_epochs, args.incr_epochs, v_end),
        "use_replay=%s use_gem=%s use_ewc=%s use_mask_dropout=%s replay_size=%d replay_per_step=%d replay_beta=%.3f distill_alpha=%.3f temperature=%.3f ewc_lambda=%.3f ewc_gamma=%.3f dropout_unmask_p=%.3f mask_keep_ratio=%.3f mask_ema=%.3f"
        % (
            str(args.use_replay),
            str(args.use_gem),
            str(args.use_ewc),
            str(args.use_mask_dropout),
            int(args.replay_size),
            int(args.replay_per_step),
            float(args.replay_beta),
            float(args.distill_alpha),
            float(args.temperature),
            float(args.ewc_lambda),
            float(args.ewc_gamma),
            float(args.dropout_unmask_p),
            float(args.mask_keep_ratio),
            float(args.mask_ema),
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
    with open(os.path.join(args.out_dir, "continual_metrics.txt"), "w") as f:
        f.write(rep)

    sum_path = os.path.join(args.out_dir, "rank_scores_all_one_line.txt")
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

