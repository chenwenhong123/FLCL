import copy
import csv
import os
import pickle
import random
import sys
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from continual_run_with_replay import (
    DEVICE,
    NlEncoder,
    ScheduledOptim,
    _resolve_data_path,
    _result_dir,
    _run_out_dir,
    build_one_id_dataset,
    calc_summary,
    get_batch_for_id,
    get_device,
    init_args,
    rank_metrics_from_batch,
    reservoir_update,
    set_seed,
)


def _method_kl_div(new_pre, old_pre, node_ids, temperature: float = 2.0):
    method_mask = torch.eq(node_ids, 2)
    if method_mask.sum().item() == 0:
        return torch.tensor(0.0, device=new_pre.device)
    new_logits = (new_pre / temperature).masked_fill(~method_mask, -1e9)
    old_logits = (old_pre / temperature).masked_fill(~method_mask, -1e9)
    new_log_prob = F.log_softmax(new_logits, dim=-1)
    old_prob = F.softmax(old_logits, dim=-1)
    return F.kl_div(new_log_prob, old_prob, reduction="batchmean") * (temperature ** 2)


def _trainable_params(model: NlEncoder):
    return [p for p in model.parameters() if p.requires_grad]


def _grads_to_vector(params):
    vec = []
    for p in params:
        if p.grad is None:
            vec.append(torch.zeros_like(p).view(-1))
        else:
            vec.append(p.grad.detach().view(-1))
    return torch.cat(vec)


def _vector_to_grads(vec, params):
    offset = 0
    for p in params:
        numel = p.numel()
        p.grad = vec[offset : offset + numel].view_as(p).clone()
        offset += numel


def clone_trainable_params(model: NlEncoder) -> Dict[str, torch.Tensor]:
    out = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            out[n] = p.detach().clone()
    return out


def init_fisher(model: NlEncoder) -> Dict[str, torch.Tensor]:
    out = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            out[n] = torch.zeros_like(p, device=p.device)
    return out


def compute_ewc_penalty(
    model: NlEncoder,
    prev_params: Dict[str, torch.Tensor],
    fisher: Dict[str, torch.Tensor],
):
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        penalty = penalty + (fisher[n] * (p - prev_params[n]).pow(2)).sum()
    return penalty


def estimate_fisher_from_ids(
    model: NlEncoder,
    args,
    project: str,
    sample_ids: List[int],
    ds_cache,
    batch_cache,
    cache_on_device: bool,
):
    fisher_new = init_fisher(model)
    if len(sample_ids) == 0:
        return fisher_new
    model.train()
    for sid in sample_ids:
        batch = get_batch_for_id(args, project, sid, ds_cache, batch_cache, cache_on_device)
        model.zero_grad(set_to_none=True)
        loss, _, _ = model(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7])
        loss = loss.mean()
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher_new[n] += p.grad.detach().pow(2)
    for n in fisher_new:
        fisher_new[n] = fisher_new[n] / float(len(sample_ids))
    return fisher_new


def train_incremental_with_all(
    model: NlEncoder,
    optimizer: ScheduledOptim,
    args,
    project: str,
    cur_id: int,
    inc_epochs: int,
    replay_ids: List[int],
    replay_per_step: int,
    replay_beta: float,
    distill_alpha: float,
    temperature: float,
    prev_params: Dict[str, torch.Tensor],
    fisher: Dict[str, torch.Tensor],
    ewc_lambda: float,
    ds_cache,
    batch_cache,
    cache_on_device: bool,
):
    """
    Replay + DER++ + EWC + A-GEM:
    - g_cur 来自 current supervised + EWC
    - g_ref 来自 replay supervised + replay distill
    - 若冲突则用 A-GEM 投影 g_cur
    """
    model.train()
    params = _trainable_params(model)
    old_model = copy.deepcopy(model).to(DEVICE).eval()

    for _ in range(inc_epochs):
        # current side (add EWC here)
        cur_batch = get_batch_for_id(args, project, cur_id, ds_cache, batch_cache, cache_on_device)
        cur_loss, _, _ = model(
            cur_batch[0], cur_batch[1], cur_batch[2], cur_batch[3], cur_batch[4], cur_batch[5], cur_batch[6], cur_batch[7]
        )
        total_cur = cur_loss.mean()
        if ewc_lambda > 0:
            total_cur = total_cur + ewc_lambda * compute_ewc_penalty(model, prev_params, fisher)

        optimizer.zero_grad()
        total_cur.backward()
        g_cur = _grads_to_vector(params)

        # replay reference side
        use_replay = min(replay_per_step, len(replay_ids))
        if use_replay > 0:
            picked = random.sample(replay_ids, use_replay)
            rep_sup_losses = []
            rep_kls = []
            for rid in picked:
                rep_batch = get_batch_for_id(args, project, rid, ds_cache, batch_cache, cache_on_device)
                rep_loss, rep_pre, _ = model(
                    rep_batch[0],
                    rep_batch[1],
                    rep_batch[2],
                    rep_batch[3],
                    rep_batch[4],
                    rep_batch[5],
                    rep_batch[6],
                    rep_batch[7],
                )
                rep_sup_losses.append(rep_loss.mean())
                with torch.no_grad():
                    _, old_pre, _ = old_model(
                        rep_batch[0],
                        rep_batch[1],
                        rep_batch[2],
                        rep_batch[3],
                        rep_batch[4],
                        rep_batch[5],
                        rep_batch[6],
                        rep_batch[7],
                    )
                rep_kls.append(_method_kl_div(rep_pre, old_pre, rep_batch[0], temperature))

            rep_sup = torch.stack(rep_sup_losses).mean()
            rep_kd = torch.stack(rep_kls).mean()
            ref_loss = replay_beta * rep_sup + distill_alpha * rep_kd

            optimizer.zero_grad()
            ref_loss.backward()
            g_ref = _grads_to_vector(params)

            dot = torch.dot(g_cur, g_ref)
            if dot < 0:
                ref_norm_sq = torch.dot(g_ref, g_ref).clamp(min=1e-12)
                g_proj = g_cur - (dot / ref_norm_sq) * g_ref
            else:
                g_proj = g_cur
        else:
            g_proj = g_cur

        optimizer.zero_grad()
        _vector_to_grads(g_proj, params)
        optimizer.step_and_update_lr()


def main():
    if len(sys.argv) < 5:
        print(
            "Usage: python continual_run_all.py <Project> <lr> <seed> <batch_size> "
            "[warmup_ids=10] [base_epochs=15] [inc_epochs=2] "
            "[replay_size=20] [replay_per_step=2] [replay_beta=1.0] [distill_alpha=0.5] [temperature=2.0] "
            "[ewc_lambda=10.0] [ewc_gamma=0.9] [fisher_ids=4]"
        )
        sys.exit(1)

    project = sys.argv[1]
    lr = float(sys.argv[2])
    seed = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    warmup_ids = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    base_epochs = int(sys.argv[6]) if len(sys.argv) > 6 else 15
    inc_epochs = int(sys.argv[7]) if len(sys.argv) > 7 else 2
    replay_size = int(sys.argv[8]) if len(sys.argv) > 8 else 20
    replay_per_step = int(sys.argv[9]) if len(sys.argv) > 9 else 2
    replay_beta = float(sys.argv[10]) if len(sys.argv) > 10 else 1.0
    distill_alpha = float(sys.argv[11]) if len(sys.argv) > 11 else 0.5
    temperature = float(sys.argv[12]) if len(sys.argv) > 12 else 2.0
    ewc_lambda = float(sys.argv[13]) if len(sys.argv) > 13 else 10.0
    ewc_gamma = float(sys.argv[14]) if len(sys.argv) > 14 else 0.9
    fisher_ids = int(sys.argv[15]) if len(sys.argv) > 15 else 4
    run_t0 = time.perf_counter()
    run_ts = datetime.now().strftime("%m%d%H%M")
    run_id = (
        f"all_base{base_epochs}_inc{inc_epochs}_lr{lr}_bs{batch_size}_warm{warmup_ids}"
        f"_rs{replay_size}_rps{replay_per_step}_rb{replay_beta}_da{distill_alpha}"
        f"_el{ewc_lambda}_eg{ewc_gamma}_fid{fisher_ids}_ts{run_ts}"
    )

    set_seed(seed)
    args = init_args(project, lr, seed, batch_size)
    print(f"using device: {get_device()}")

    cache_on_device = DEVICE.type == "cuda" and (args.NlLen + args.CodeLen) <= 1200
    ds_cache = {}
    batch_cache = {}
    print(f"batch_cache_on_device: {cache_on_device}")

    data = pickle.load(open(_resolve_data_path(project + ".pkl"), "rb"))
    total_ids = len(data)
    if total_ids <= warmup_ids:
        raise ValueError(f"{project} only has {total_ids} ids, warmup_ids={warmup_ids} is too large.")

    warmup_range = list(range(warmup_ids))
    stream_range = list(range(warmup_ids, total_ids))

    best_base = None
    best_state = None
    best_score = (-1, -1, float("inf"))

    for bid in tqdm(warmup_range, desc="Warmup Base Selection", ncols=100):
        set_seed(seed + bid)
        ds = build_one_id_dataset(args, project, bid)
        ds_cache[bid] = ds
        model = NlEncoder(args).to(DEVICE)
        optimizer = ScheduledOptim(optim.Adam(model.parameters(), lr=args.lr), args.embedding_size, 4000)
        for _ in range(base_epochs):
            batch = get_batch_for_id(args, project, bid, ds_cache, batch_cache, cache_on_device)
            loss, _, _ = model(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7])
            optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            optimizer.step_and_update_lr()
        batch = get_batch_for_id(args, project, bid, ds_cache, batch_cache, cache_on_device)
        rank, _ = rank_metrics_from_batch(model, batch, data[bid]["ans"])
        top1 = 1 if rank == 0 else 0
        top3 = 1 if rank < 3 else 0
        mfr = float(rank)
        score = (top1, top3, -mfr)
        if score > (best_score[0], best_score[1], -best_score[2]):
            best_score = (top1, top3, mfr)
            best_state = copy.deepcopy(model.state_dict())
            best_base = bid

    if best_state is None:
        raise RuntimeError("Failed to choose a base model from warmup ids.")

    model = NlEncoder(args).to(DEVICE)
    model.load_state_dict(best_state)
    optimizer = ScheduledOptim(optim.Adam(model.parameters(), lr=args.lr), args.embedding_size, 4000)

    first_learn_top1: Dict[int, float] = {}
    first_learn_top3: Dict[int, float] = {}
    replay_buffer_ids: List[int] = list(warmup_range[: max(0, replay_size)])
    seen_count = warmup_ids - 1

    prev_params = clone_trainable_params(model)
    fisher = init_fisher(model)

    for bid in warmup_range:
        batch = get_batch_for_id(args, project, bid, ds_cache, batch_cache, cache_on_device)
        rank, _ = rank_metrics_from_batch(model, batch, data[bid]["ans"])
        first_learn_top1[bid] = 1.0 if rank == 0 else 0.0
        first_learn_top3[bid] = 1.0 if rank < 3 else 0.0

    rows = []
    for t in tqdm(stream_range, desc="Incremental Stream (Replay+DER+++A-GEM+EWC)", ncols=100):
        train_incremental_with_all(
            model=model,
            optimizer=optimizer,
            args=args,
            project=project,
            cur_id=t,
            inc_epochs=inc_epochs,
            replay_ids=replay_buffer_ids,
            replay_per_step=replay_per_step,
            replay_beta=replay_beta,
            distill_alpha=distill_alpha,
            temperature=temperature,
            prev_params=prev_params,
            fisher=fisher,
            ewc_lambda=ewc_lambda,
            ds_cache=ds_cache,
            batch_cache=batch_cache,
            cache_on_device=cache_on_device,
        )

        cur_batch = get_batch_for_id(args, project, t, ds_cache, batch_cache, cache_on_device)
        cur_rank, cur_mar = rank_metrics_from_batch(model, cur_batch, data[t]["ans"])
        online_top1 = 1.0 if cur_rank == 0 else 0.0
        first_learn_top1[t] = online_top1
        first_learn_top3[t] = 1.0 if cur_rank < 3 else 0.0

        seen_ids = list(range(0, t + 1))
        seen_ranks = []
        seen_mars = []
        seen_top1 = []
        seen_top3 = []
        for sid in seen_ids:
            batch = get_batch_for_id(args, project, sid, ds_cache, batch_cache, cache_on_device)
            r, m = rank_metrics_from_batch(model, batch, data[sid]["ans"])
            seen_ranks.append(r)
            seen_mars.append(m)
            seen_top1.append(1.0 if r == 0 else 0.0)
            seen_top3.append(1.0 if r < 3 else 0.0)

        acc_top1_t = float(np.mean(seen_top1))
        acc_top3_t = float(np.mean(seen_top3))
        old_ids = list(range(0, t))
        if len(old_ids) > 0:
            bwt_terms = [seen_top1[idx] - first_learn_top1[idx] for idx in old_ids]
            bwt_t = float(np.mean(bwt_terms))
        else:
            bwt_t = 0.0

        summary = calc_summary(seen_ranks, seen_mars)
        rows.append(
            {
                "t": t,
                "online_top1_t": online_top1,
                "acc_top1_t": acc_top1_t,
                "acc_top3_t": acc_top3_t,
                "bwt_t": bwt_t,
                "top1": summary["top1"],
                "top3": summary["top3"],
                "top5": summary["top5"],
                "top1_count": summary["top1_count"],
                "top3_count": summary["top3_count"],
                "top5_count": summary["top5_count"],
                "n": summary["n"],
                "mfr": summary["mfr"],
                "mar": summary["mar"],
                "current_rank": cur_rank,
                "current_mar": cur_mar,
            }
        )
        tqdm.write(
            f"[t={t}] online_top1={online_top1:.3f} acc_top1={acc_top1_t:.3f} acc_top3={acc_top3_t:.3f} bwt={bwt_t:.3f}"
        )

        seen_count += 1
        reservoir_update(replay_buffer_ids, seen_count, t, replay_size)

        fisher_sample_ids = [t]
        if len(replay_buffer_ids) > 0 and fisher_ids > 1:
            k = min(fisher_ids - 1, len(replay_buffer_ids))
            fisher_sample_ids += random.sample(replay_buffer_ids, k)
        fisher_new = estimate_fisher_from_ids(
            model, args, project, fisher_sample_ids, ds_cache, batch_cache, cache_on_device
        )
        for n in fisher:
            fisher[n] = ewc_gamma * fisher[n] + (1.0 - ewc_gamma) * fisher_new[n]
        prev_params = clone_trainable_params(model)

    out_dir = _run_out_dir(project, run_id)
    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "t",
                "online_top1_t",
                "acc_top1_t",
                "acc_top3_t",
                "bwt_t",
                "top1",
                "top3",
                "top5",
                "top1_count",
                "top3_count",
                "top5_count",
                "n",
                "mfr",
                "mar",
                "current_rank",
                "current_mar",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    model_path = os.path.join(out_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    runtime_seconds = time.perf_counter() - run_t0
    summary_path = os.path.join(out_dir, "summary.txt")
    last = rows[-1] if len(rows) > 0 else {}
    with open(summary_path, "w") as f:
        f.write(f"project: {project}\n")
        f.write(f"run_id: {run_id}\n")
        f.write(f"device: {DEVICE}\n")
        f.write(f"warmup_ids: {warmup_ids}\n")
        f.write(f"base_epochs: {base_epochs}\n")
        f.write(f"inc_epochs: {inc_epochs}\n")
        f.write(f"replay_size: {replay_size}\n")
        f.write(f"replay_per_step: {replay_per_step}\n")
        f.write(f"replay_beta: {replay_beta}\n")
        f.write(f"distill_alpha: {distill_alpha}\n")
        f.write(f"temperature: {temperature}\n")
        f.write(f"ewc_lambda: {ewc_lambda}\n")
        f.write(f"ewc_gamma: {ewc_gamma}\n")
        f.write(f"fisher_ids: {fisher_ids}\n")
        f.write(f"selected_base_id: {best_base}\n")
        f.write(f"runtime_seconds: {runtime_seconds:.3f}\n")
        if last:
            f.write(f"final_online_top1_t: {last['online_top1_t']:.6f}\n")
            f.write(f"final_acc_top1_t: {last['acc_top1_t']:.6f}\n")
            f.write(f"final_acc_top3_t: {last['acc_top3_t']:.6f}\n")
            f.write(f"final_bwt_t: {last['bwt_t']:.6f}\n")
            f.write(f"final_top1: {last['top1']:.6f}\n")
            f.write(f"final_top3: {last['top3']:.6f}\n")
            f.write(f"final_top5: {last['top5']:.6f}\n")
            f.write(f"final_top1_count: {int(last['top1_count'])}\n")
            f.write(f"final_top3_count: {int(last['top3_count'])}\n")
            f.write(f"final_top5_count: {int(last['top5_count'])}\n")
            f.write(f"final_n: {int(last['n'])}\n")
            f.write(f"final_mfr: {last['mfr']:.6f}\n")
            f.write(f"final_mar: {last['mar']:.6f}\n")

    print(f"Saved metrics csv: {csv_path}")
    print(f"Saved model: {model_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
