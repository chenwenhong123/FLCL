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
import input
import utils as ut
from utils import tqdm_compat as tqdm


class TaskReplayBuffer(cr.ReplayBuffer):
    """Replay buffer that also stores task ids for task-specific masks."""

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


def _build_task_mask_from_data(train_x, keep_ratio):
    d = train_x.shape[1]
    if d == 0:
        return np.zeros((0,), dtype=np.float32)
    k = int(max(1, round(float(keep_ratio) * d)))
    score = np.mean(np.abs(train_x), axis=0)
    top_idx = np.argsort(-score)[:k]
    m = np.zeros((d,), dtype=np.float32)
    m[top_idx] = 1.0
    return m


def _build_step_mask_from_data(train_x, feat_dist, keep_ratio):
    fd = np.asarray(feat_dist, dtype=np.int32)
    n_steps = int(fd.shape[0])
    if n_steps == 0:
        return np.zeros((0,), dtype=np.float32)
    score = np.zeros((n_steps,), dtype=np.float32)
    off = 0
    for i, w in enumerate(fd):
        w = int(w)
        seg = train_x[:, off : off + w]
        score[i] = float(np.mean(np.abs(seg)))
        off += w
    k = int(max(1, round(float(keep_ratio) * n_steps)))
    top_idx = np.argsort(-score)[:k]
    m = np.zeros((n_steps,), dtype=np.float32)
    m[top_idx] = 1.0
    return m


def _expand_step_mask_to_feature_mask(step_mask, feat_dist):
    fd = np.asarray(feat_dist, dtype=np.int32)
    total_dim = int(fd.sum())
    out = np.zeros((total_dim,), dtype=np.float32)
    off = 0
    for i, w in enumerate(fd):
        w = int(w)
        out[off : off + w] = float(step_mask[i])
        off += w
    return out


def _update_task_mask(prev_mask, new_hard_mask, ema):
    if prev_mask is None:
        return new_hard_mask.astype(np.float32)
    out = float(ema) * prev_mask + (1.0 - float(ema)) * new_hard_mask
    return np.clip(out.astype(np.float32), 0.0, 1.0)


def _rowwise_apply_task_mask(batch_x, task_ids, task_masks):
    if batch_x.shape[0] == 0:
        return batch_x
    out = np.array(batch_x, copy=True)
    for i in range(batch_x.shape[0]):
        tid = int(task_ids[i])
        m = task_masks.get(tid)
        if m is not None:
            out[i] *= m
    return out


def _current_apply_mask(batch_x, current_task_id, task_masks):
    m = task_masks.get(int(current_task_id))
    if m is None:
        return batch_x
    return batch_x * m


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


def _train_incremental_with_replay_mask(
    sess,
    graph,
    train_x,
    train_y,
    train_g,
    epochs,
    current_task_id,
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
    use_gem,
    agem_state,
    use_mask,
    task_masks,
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
                bx_cur = _current_apply_mask(bx, current_task_id, task_masks) if use_mask else bx
                before_logits = _predict_logits(sess, graph, bx_cur)
                fd_cur = cr._train_feed_dict(graph, bx_cur, by, bg, cst.DROPOUT_RATE)

                did_main_update = False
                if (
                    use_gem
                    and use_replay
                    and replay_buffer is not None
                    and replay_buffer.size > 0
                ):
                    rb_ref = replay_buffer.sample(cst.BATCH_SIZE)
                    if rb_ref is not None:
                        rx_ref, ry_ref, rg_ref, _, tid_ref = rb_ref
                        rx_ref2 = _rowwise_apply_task_mask(rx_ref, tid_ref, task_masks) if use_mask else rx_ref
                        fd_ref = cr._train_feed_dict(graph, rx_ref2, ry_ref, rg_ref, cst.DROPOUT_RATE)
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

                if use_replay and replay_buffer is not None and replay_buffer.size > 0:
                    for _j in range(int(max(1, replay_per_step))):
                        rb = replay_buffer.sample(cst.BATCH_SIZE)
                        if rb is None:
                            continue
                        rx, ry, rg, rlogits, tid = rb
                        rx2 = _rowwise_apply_task_mask(rx, tid, task_masks) if use_mask else rx
                        if replay_beta > 0.0 or distill_alpha > 0.0:
                            fd_rep = cr._train_feed_dict(graph, rx2, ry, rg, cst.DROPOUT_RATE)
                            fd_rep[derpp_teacher_ph] = rlogits
                            fd_rep[derpp_temp_ph] = float(temperature)
                            fd_rep[replay_beta_ph] = float(replay_beta)
                            fd_rep[distill_alpha_ph] = float(distill_alpha)
                            sess.run(derpp_replay_op, feed_dict=fd_rep)

                if use_replay and replay_buffer is not None:
                    replay_buffer.add_batch(bx, by, bg, before_logits, current_task_id)
                pbar.update(1)
    finally:
        pbar.close()


def _eval_bug_with_mask(sess, graph, test_x, test_y, task_id, task_masks, use_mask):
    if not use_mask:
        return cst.eval_bug(sess, graph, test_x, test_y)
    tx = _current_apply_mask(test_x, task_id, task_masks)
    return cst.eval_bug(sess, graph, tx, test_y)


def main():
    pa = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="流式持续学习 + replay + DER++ + 可选A-GEM + 可选门控掩码。",
    )
    pa.add_argument("subject", help="项目: Chart|Lang|Math|Time|Closure|Mockito")
    pa.add_argument("model", help="mlp2 | mlp_dfl_1 | birnn")
    pa.add_argument("--data_root", default=".", help="数据父目录（含 <tech>/<subject>/<ver>/）")
    pa.add_argument("--out_dir", default=None, help="输出根目录；省略则带 g/m 标识")
    pa.add_argument("--tech", default="DeepFL_CL", help="数据与输出路径中的技术子目录名（持续学习固定使用 DeepFL_CL）")
    pa.add_argument("--loss", default="softmax", help="损失函数名（当前仅 softmax）")
    pa.add_argument("--training_epochs", type=int, default=50, help="warmup 阶段 epoch 数")
    pa.add_argument("--warmup", type=int, default=10, help="前 K 个版本合并 warmup")
    pa.add_argument("--incr_epochs", type=int, default=1, help="每个新 bug 增量训练 epoch 数")
    pa.add_argument("--v_end", type=int, default=None, help="流式结束版本")
    pa.add_argument("--dump_step", type=int, default=10, help="写入 config 的 DEEPFL_DUMP_STEP")
    pa.add_argument("--gpu_mem", type=float, default=0.2, help="GPU 显存占比（上限 0.2）")
    pa.add_argument("--use_replay", type=cr._to_bool, default=True, help="是否启用 replay+DER++（默认开）")
    pa.add_argument("--use_ewc", type=cr._to_bool, default=False, help="是否启用 EWC（默认关）")
    pa.add_argument("--use_gem", type=cr._to_bool, default=True, help="是否启用 A-GEM（默认开）")
    pa.add_argument("--use_mask", type=cr._to_bool, default=True, help="是否启用门控掩码（默认开）")
    pa.add_argument("--ewc_lambda", type=float, default=10.0, help="EWC 惩罚系数")
    pa.add_argument("--ewc_gamma", type=float, default=0.9, help="Online EWC 衰减系数")
    pa.add_argument("--replay_size", type=int, default=20, help="回放缓冲区容量")
    pa.add_argument("--replay_per_step", type=int, default=2, help="每个增量 batch 的回放批次数")
    pa.add_argument("--replay_beta", type=float, default=1.0, help="DER++ 监督回放项权重")
    pa.add_argument("--distill_alpha", type=float, default=0.5, help="DER++ 蒸馏项权重")
    pa.add_argument("--temperature", type=float, default=2.0, help="蒸馏温度系数")
    pa.add_argument("--mask_keep_ratio", type=float, default=0.6, help="每任务门控保留特征比例（仅 mlp_dfl_1）")
    pa.add_argument("--mask_ema", type=float, default=0.9, help="任务掩码 EMA 平滑系数")
    args = pa.parse_args()
    args.gpu_mem = min(max(float(args.gpu_mem), 1e-6), 0.2)
    args.mask_keep_ratio = min(max(float(args.mask_keep_ratio), 1e-3), 1.0)
    args.mask_ema = min(max(float(args.mask_ema), 0.0), 0.999)

    supported = ("mlp2", "mlp_dfl_1", "birnn")
    if args.model not in supported:
        print("continual_mask: model 必须是: %s" % " ".join(supported), file=sys.stderr)
        sys.exit(1)
    if args.loss != "softmax":
        print("continual_mask: 当前仅支持 loss=softmax", file=sys.stderr)
        sys.exit(1)
    if args.tech != "DeepFL_CL":
        print("continual_mask: 为避免数据协议泄漏，tech 仅允许 DeepFL_CL", file=sys.stderr)
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

    use_mask_eff = bool(args.use_mask and args.model in ("mlp_dfl_1", "mlp2", "birnn"))

    ut.apply_deepfl_env_from_args(args)
    import config as cfg  # noqa: F401

    lr_for_tag = float(cfg.learning_rate)
    run_tag = "ur%d_ue%d_lr%.4f_ug%d_um%d_mr%.2f_me%.2f_rps%d_da%.2f_el%.1f_eg%.2f_wup%d_te%d_incr%d" % (
        1 if args.use_replay else 0,
        1 if args.use_ewc else 0,
        lr_for_tag,
        1 if args.use_gem else 0,
        1 if use_mask_eff else 0,
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

    loss_idx = cst.LOSSES.index(args.loss)
    n_input = cst.FEATURE_SIZE[cst.TECH_NAMES.index(tech)]
    n_hidden = n_input
    out_root = args.out_dir
    os.makedirs(out_root, exist_ok=True)
    rank_summary_lines = []

    tf.reset_default_graph()
    if args.model == "mlp2":
        graph = cst.build_mlp2_graph(n_input, n_hidden, loss_idx)
    elif args.model == "mlp_dfl_1":
        graph = cst.build_fc_stream_graph(loss_idx, "fc_based_1", "fc_2_layers")
    elif args.model == "birnn":
        graph = cst.build_seq_stream_graph("birnn", loss_idx, cfg.featureDistribution)
    else:
        sys.exit(1)

    ewc_state = cst.build_ewc_state(graph, args.ewc_lambda) if args.use_ewc else None
    agem_state = _build_agem_ops(graph) if args.use_gem else None
    derpp_teacher_ph, derpp_temp_ph, replay_beta_ph, distill_alpha_ph, derpp_replay_op = cr._build_derpp_replay_op(
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

    replay_buffer = TaskReplayBuffer(args.replay_size, n_input) if args.use_replay else None
    task_masks = {}

    wx, wy, wg = cst.merge_train_bugs(args.data_root, tech, sub, 1, warmup)
    cst.train_epochs(sess, graph, wx, wy, wg, args.training_epochs, desc="warmup")
    if use_mask_eff:
        task_masks[0] = np.ones((n_input,), dtype=np.float32)
        for v in range(1, warmup + 1):
            task_masks[v] = np.ones((n_input,), dtype=np.float32)
    ewc_stats = None
    if args.use_ewc:
        omega0 = [np.zeros(v.shape.as_list(), dtype=np.float32) for v in ewc_state["vars"]]
        wx_for_ewc = _current_apply_mask(wx, 0, task_masks) if use_mask_eff else wx
        omega0, star0 = cst.update_ewc_stats(sess, graph, ewc_state, wx_for_ewc, wy, wg, omega0, args.ewc_gamma)
        ewc_stats = {"omega": omega0, "star": star0}

    acc_right_after = {}
    for v in tqdm(range(1, warmup + 1), desc="eval_after_warmup"):
        _, _, _, txi, tyl = cst.load_one_bug(args.data_root, tech, sub, v)
        acc_right_after[v] = _eval_bug_with_mask(sess, graph, txi, tyl, v, task_masks, use_mask_eff)

    pre_before_incr = {}
    prev_mask = task_masks.get(warmup, np.ones((n_input,), dtype=np.float32)) if use_mask_eff else None
    mask_mode = "step" if args.model == "birnn" else "feature"
    for v in tqdm(range(warmup + 1, v_end + 1), desc="incremental"):
        ti, tl, tg, txi, tyl = cst.load_one_bug(args.data_root, tech, sub, v)
        if use_mask_eff:
            if mask_mode == "step":
                hard_step = _build_step_mask_from_data(ti, cfg.featureDistribution, args.mask_keep_ratio)
                hard = _expand_step_mask_to_feature_mask(hard_step, cfg.featureDistribution)
            else:
                hard = _build_task_mask_from_data(ti, args.mask_keep_ratio)
            m = _update_task_mask(prev_mask, hard, args.mask_ema)
            task_masks[v] = m
            prev_mask = m

        pre = _eval_bug_with_mask(sess, graph, txi, tyl, v, task_masks, use_mask_eff)
        pre_before_incr[v] = pre
        if pre is not None:
            rank_summary_lines.append("PRE\t%d\t%s" % (v, " ".join(str(s) for s in pre["scores"])))

        _train_incremental_with_replay_mask(
            sess=sess,
            graph=graph,
            train_x=ti,
            train_y=tl,
            train_g=tg,
            epochs=args.incr_epochs,
            current_task_id=v,
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
            use_gem=args.use_gem,
            agem_state=agem_state,
            use_mask=use_mask_eff,
            task_masks=task_masks,
            desc="incr v=%d" % v,
        )

        if args.use_ewc:
            ti_for_ewc = _current_apply_mask(ti, v, task_masks) if use_mask_eff else ti
            new_omega, new_star = cst.update_ewc_stats(
                sess, graph, ewc_state, ti_for_ewc, tl, tg, ewc_stats["omega"], args.ewc_gamma
            )
            ewc_stats["omega"] = new_omega
            ewc_stats["star"] = new_star

        acc_right_after[v] = _eval_bug_with_mask(sess, graph, txi, tyl, v, task_masks, use_mask_eff)

    final_eval = {}
    for v in tqdm(range(1, v_end + 1), desc="final_eval"):
        _, _, _, txi, tyl = cst.load_one_bug(args.data_root, tech, sub, v)
        ev = _eval_bug_with_mask(sess, graph, txi, tyl, v, task_masks, use_mask_eff)
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
        "use_replay=%s use_ewc=%s use_gem=%s use_mask=%s mask_mode=%s replay_size=%d replay_per_step=%d replay_beta=%.3f distill_alpha=%.3f temperature=%.3f ewc_lambda=%.3f ewc_gamma=%.3f mask_keep_ratio=%.3f mask_ema=%.3f"
        % (
            str(args.use_replay),
            str(args.use_ewc),
            str(args.use_gem),
            str(use_mask_eff),
            mask_mode if use_mask_eff else "off",
            int(args.replay_size),
            int(args.replay_per_step),
            float(args.replay_beta),
            float(args.distill_alpha),
            float(args.temperature),
            float(args.ewc_lambda),
            float(args.ewc_gamma),
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
