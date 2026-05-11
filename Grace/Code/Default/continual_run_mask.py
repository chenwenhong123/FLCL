import copy
import csv
import os
import pickle
import random
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from continual_run_with_replay import (
    DEVICE,
    NlEncoder,
    ScheduledOptim,
    _resolve_data_path,
    _result_dir,
    build_one_id_dataset,
    calc_summary,
    get_batch_for_id,
    get_device,
    init_args,
    rank_metrics_from_batch,  # kept for compatibility, not used directly here
    reservoir_update,
    set_seed,
)


class TaskGate(nn.Module):
    """Task-wise gating mask: one gate vector per task id."""

    def __init__(self, num_tasks: int, hidden_size: int):
        super().__init__()
        self.task_embed = nn.Embedding(num_tasks, hidden_size)
        nn.init.zeros_(self.task_embed.weight)

    def gate(self, task_id: int, batch_size: int, device: torch.device):
        tid = torch.full((batch_size,), int(task_id), dtype=torch.long, device=device)
        g = torch.sigmoid(self.task_embed(tid))  # [B, H]
        return g


def _method_kl_div(new_pre, old_pre, node_ids, temperature: float = 2.0):
    method_mask = torch.eq(node_ids, 2)
    if method_mask.sum().item() == 0:
        return torch.tensor(0.0, device=new_pre.device)
    new_logits = (new_pre / temperature).masked_fill(~method_mask, -1e9)
    old_logits = (old_pre / temperature).masked_fill(~method_mask, -1e9)
    new_log_prob = F.log_softmax(new_logits, dim=-1)
    old_prob = F.softmax(old_logits, dim=-1)
    return F.kl_div(new_log_prob, old_prob, reduction="batchmean") * (temperature ** 2)


def forward_with_task_mask(model: NlEncoder, gate_net: TaskGate, batch, task_id: int):
    """
    Recompute localization scores with task gate on top of encoder features.
    This changes model framework while keeping original backbone intact.
    """
    _, _, x = model(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7])
    g = gate_net.gate(task_id, x.size(0), x.device).unsqueeze(1)  # [B, 1, H]
    x_masked = x * g
    resmask = torch.eq(batch[0], 2)
    pre = F.softmax(model.resLinear2(x_masked).squeeze(-1).masked_fill(resmask == 0, -1e9), dim=-1)
    loss = -torch.log(pre.clamp(min=1e-10, max=1.0)) * batch[3]
    loss = loss.sum(dim=-1)
    return loss, pre, x_masked


def rank_metrics_from_batch_masked(model: NlEncoder, gate_net: TaskGate, batch, ans_list: List[int], task_id: int):
    model.eval()
    gate_net.eval()
    with torch.no_grad():
        _, pre, _ = forward_with_task_mask(model, gate_net, batch, task_id)
        resmask = torch.eq(batch[0], 2)
        s = -pre
        s = s.masked_fill(resmask == 0, 1e9)
        pred = s.argsort(dim=-1)
        pred = pred.data.cpu().numpy()
        pred_list = pred[0].tolist()[: resmask.sum(dim=-1)[0].item()]
        ranks = [pred_list.index(x) for x in ans_list]
        min_rank = min(ranks)
        mar = float(np.mean(ranks))
    return min_rank, mar


def _trainable_params(model: NlEncoder, gate_net: TaskGate):
    params = [p for p in model.parameters() if p.requires_grad]
    params.extend([p for p in gate_net.parameters() if p.requires_grad])
    return params


def _named_trainable_params(model: NlEncoder, gate_net: TaskGate):
    items = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            items.append(("model." + n, p))
    for n, p in gate_net.named_parameters():
        if p.requires_grad:
            items.append(("gate." + n, p))
    return items


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


def clone_trainable_params(model: NlEncoder, gate_net: TaskGate) -> Dict[str, torch.Tensor]:
    out = {}
    for n, p in _named_trainable_params(model, gate_net):
        out[n] = p.detach().clone()
    return out


def init_fisher(model: NlEncoder, gate_net: TaskGate) -> Dict[str, torch.Tensor]:
    out = {}
    for n, p in _named_trainable_params(model, gate_net):
        out[n] = torch.zeros_like(p, device=p.device)
    return out


def compute_ewc_penalty(
    model: NlEncoder,
    gate_net: TaskGate,
    prev_params: Dict[str, torch.Tensor],
    fisher: Dict[str, torch.Tensor],
):
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for n, p in _named_trainable_params(model, gate_net):
        penalty = penalty + (fisher[n] * (p - prev_params[n]).pow(2)).sum()
    return penalty


def estimate_fisher_from_ids(
    model: NlEncoder,
    gate_net: TaskGate,
    args,
    project: str,
    sample_ids: List[int],
    ds_cache,
    batch_cache,
    cache_on_device: bool,
):
    fisher_new = init_fisher(model, gate_net)
    if len(sample_ids) == 0:
        return fisher_new
    model.train()
    gate_net.train()
    for sid in sample_ids:
        batch = get_batch_for_id(args, project, sid, ds_cache, batch_cache, cache_on_device)
        model.zero_grad(set_to_none=True)
        gate_net.zero_grad(set_to_none=True)
        loss, _, _ = forward_with_task_mask(model, gate_net, batch, sid)
        loss = loss.mean()
        loss.backward()
        for n, p in _named_trainable_params(model, gate_net):
            if p.grad is not None:
                fisher_new[n] += p.grad.detach().pow(2)
    for n in fisher_new:
        fisher_new[n] = fisher_new[n] / float(len(sample_ids))
    return fisher_new


def train_incremental_with_mask(
    model: NlEncoder,
    gate_net: TaskGate,
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
    use_ewc: bool,
    use_gem: bool,
    prev_params: Dict[str, torch.Tensor],
    fisher: Dict[str, torch.Tensor],
    ewc_lambda: float,
    ds_cache,
    batch_cache,
    cache_on_device: bool,
):
    model.train()
    gate_net.train()
    params = _trainable_params(model, gate_net)
    old_model = copy.deepcopy(model).to(DEVICE).eval()
    old_gate = copy.deepcopy(gate_net).to(DEVICE).eval()

    for _ in range(inc_epochs):
        # current branch (+ optional EWC)
        cur_batch = get_batch_for_id(args, project, cur_id, ds_cache, batch_cache, cache_on_device)
        cur_loss, _, _ = forward_with_task_mask(model, gate_net, cur_batch, cur_id)
        total_cur = cur_loss.mean()
        if use_ewc and ewc_lambda > 0:
            total_cur = total_cur + ewc_lambda * compute_ewc_penalty(model, gate_net, prev_params, fisher)

        optimizer.zero_grad()
        total_cur.backward()
        g_cur = _grads_to_vector(params)

        # replay reference branch (DER++)
        use_replay = min(replay_per_step, len(replay_ids))
        if use_replay > 0:
            picked = random.sample(replay_ids, use_replay)
            rep_sup_losses = []
            rep_kls = []
            for rid in picked:
                rep_batch = get_batch_for_id(args, project, rid, ds_cache, batch_cache, cache_on_device)
                rep_loss, rep_pre, _ = forward_with_task_mask(model, gate_net, rep_batch, rid)
                rep_sup_losses.append(rep_loss.mean())
                with torch.no_grad():
                    _, old_pre, _ = forward_with_task_mask(old_model, old_gate, rep_batch, rid)
                rep_kls.append(_method_kl_div(rep_pre, old_pre, rep_batch[0], temperature))

            rep_sup = torch.stack(rep_sup_losses).mean()
            rep_kd = torch.stack(rep_kls).mean()
            ref_loss = replay_beta * rep_sup + distill_alpha * rep_kd

            optimizer.zero_grad()
            ref_loss.backward()
            g_ref = _grads_to_vector(params)
        else:
            g_ref = None

        # optional A-GEM projection
        if use_gem and g_ref is not None:
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
            "Usage: python continual_run_mask.py <Project> <lr> <seed> <batch_size> "
            "[warmup_ids=10] [base_epochs=15] [inc_epochs=2] "
            "[replay_size=20] [replay_per_step=2] [replay_beta=1.0] [distill_alpha=0.5] [temperature=2.0] "
            "[ewc_lambda=10.0] [ewc_gamma=0.9] [fisher_ids=4] [use_ewc=1] [use_gem=1]"
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
    use_ewc = int(sys.argv[16]) if len(sys.argv) > 16 else 1
    use_gem = int(sys.argv[17]) if len(sys.argv) > 17 else 1

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
    best_gate_state = None
    best_score = (-1, -1, float("inf"))

    # warmup: pick best base model + gate
    for bid in tqdm(warmup_range, desc="Warmup Base Selection", ncols=100):
        set_seed(seed + bid)
        ds = build_one_id_dataset(args, project, bid)
        ds_cache[bid] = ds
        model = NlEncoder(args).to(DEVICE)
        gate_net = TaskGate(total_ids, args.embedding_size).to(DEVICE)
        optimizer = ScheduledOptim(
            optim.Adam(list(model.parameters()) + list(gate_net.parameters()), lr=args.lr),
            args.embedding_size,
            4000,
        )
        for _ in range(base_epochs):
            batch = get_batch_for_id(args, project, bid, ds_cache, batch_cache, cache_on_device)
            loss, _, _ = forward_with_task_mask(model, gate_net, batch, bid)
            optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            optimizer.step_and_update_lr()
        batch = get_batch_for_id(args, project, bid, ds_cache, batch_cache, cache_on_device)
        rank, _ = rank_metrics_from_batch_masked(model, gate_net, batch, data[bid]["ans"], bid)
        top1 = 1 if rank == 0 else 0
        top3 = 1 if rank < 3 else 0
        mfr = float(rank)
        score = (top1, top3, -mfr)
        if score > (best_score[0], best_score[1], -best_score[2]):
            best_score = (top1, top3, mfr)
            best_state = copy.deepcopy(model.state_dict())
            best_gate_state = copy.deepcopy(gate_net.state_dict())
            best_base = bid

    if best_state is None or best_gate_state is None:
        raise RuntimeError("Failed to choose a base model from warmup ids.")

    model = NlEncoder(args).to(DEVICE)
    gate_net = TaskGate(total_ids, args.embedding_size).to(DEVICE)
    model.load_state_dict(best_state)
    gate_net.load_state_dict(best_gate_state)
    optimizer = ScheduledOptim(
        optim.Adam(list(model.parameters()) + list(gate_net.parameters()), lr=args.lr),
        args.embedding_size,
        4000,
    )

    first_learn_top1: Dict[int, float] = {}
    first_learn_top3: Dict[int, float] = {}
    replay_buffer_ids: List[int] = list(warmup_range[: max(0, replay_size)])
    seen_count = warmup_ids - 1

    prev_params = clone_trainable_params(model, gate_net)
    fisher = init_fisher(model, gate_net)

    for bid in warmup_range:
        batch = get_batch_for_id(args, project, bid, ds_cache, batch_cache, cache_on_device)
        rank, _ = rank_metrics_from_batch_masked(model, gate_net, batch, data[bid]["ans"], bid)
        first_learn_top1[bid] = 1.0 if rank == 0 else 0.0
        first_learn_top3[bid] = 1.0 if rank < 3 else 0.0

    rows = []
    for t in tqdm(stream_range, desc="Incremental Stream (Mask+Replay+DER++)", ncols=100):
        train_incremental_with_mask(
            model=model,
            gate_net=gate_net,
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
            use_ewc=bool(use_ewc),
            use_gem=bool(use_gem),
            prev_params=prev_params,
            fisher=fisher,
            ewc_lambda=ewc_lambda,
            ds_cache=ds_cache,
            batch_cache=batch_cache,
            cache_on_device=cache_on_device,
        )

        cur_batch = get_batch_for_id(args, project, t, ds_cache, batch_cache, cache_on_device)
        cur_rank, cur_mar = rank_metrics_from_batch_masked(model, gate_net, cur_batch, data[t]["ans"], t)
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
            r, m = rank_metrics_from_batch_masked(model, gate_net, batch, data[sid]["ans"], sid)
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

        if bool(use_ewc):
            fisher_sample_ids = [t]
            if len(replay_buffer_ids) > 0 and fisher_ids > 1:
                k = min(fisher_ids - 1, len(replay_buffer_ids))
                fisher_sample_ids += random.sample(replay_buffer_ids, k)
            fisher_new = estimate_fisher_from_ids(
                model, gate_net, args, project, fisher_sample_ids, ds_cache, batch_cache, cache_on_device
            )
            for n in fisher:
                fisher[n] = ewc_gamma * fisher[n] + (1.0 - ewc_gamma) * fisher_new[n]
            prev_params = clone_trainable_params(model, gate_net)

    out_dir = _result_dir(project)
    csv_path = os.path.join(
        out_dir,
        f"{project}_continual_mask_metrics_use_ewc{use_ewc}_use_gem{use_gem}_base{base_epochs}_inc{inc_epochs}_lr{lr}_bs{batch_size}_warm{warmup_ids}.csv",
    )
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
                "mfr",
                "mar",
                "current_rank",
                "current_mar",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    model_path = os.path.join(
        out_dir,
        f"{project}_continual_mask_model_use_ewc{use_ewc}_use_gem{use_gem}_base{base_epochs}_inc{inc_epochs}_lr{lr}_bs{batch_size}_warm{warmup_ids}.pt",
    )
    torch.save({"model": model.state_dict(), "gate": gate_net.state_dict()}, model_path)

    summary_path = os.path.join(
        out_dir,
        f"{project}_continual_mask_summary_use_ewc{use_ewc}_use_gem{use_gem}_base{base_epochs}_inc{inc_epochs}_lr{lr}_bs{batch_size}_warm{warmup_ids}.txt",
    )
    last = rows[-1] if len(rows) > 0 else {}
    with open(summary_path, "w") as f:
        f.write(f"project: {project}\n")
        f.write(f"device: {DEVICE}\n")
        f.write(f"warmup_ids: {warmup_ids}\n")
        f.write(f"base_epochs: {base_epochs}\n")
        f.write(f"inc_epochs: {inc_epochs}\n")
        f.write(f"replay_size: {replay_size}\n")
        f.write(f"replay_per_step: {replay_per_step}\n")
        f.write(f"replay_beta: {replay_beta}\n")
        f.write(f"distill_alpha: {distill_alpha}\n")
        f.write(f"temperature: {temperature}\n")
        f.write(f"use_ewc: {use_ewc}\n")
        f.write(f"use_gem: {use_gem}\n")
        f.write(f"ewc_lambda: {ewc_lambda}\n")
        f.write(f"ewc_gamma: {ewc_gamma}\n")
        f.write(f"fisher_ids: {fisher_ids}\n")
        f.write(f"selected_base_id: {best_base}\n")
        if last:
            f.write(f"final_online_top1_t: {last['online_top1_t']:.6f}\n")
            f.write(f"final_acc_top1_t: {last['acc_top1_t']:.6f}\n")
            f.write(f"final_acc_top3_t: {last['acc_top3_t']:.6f}\n")
            f.write(f"final_bwt_t: {last['bwt_t']:.6f}\n")
            f.write(f"final_top1: {last['top1']:.6f}\n")
            f.write(f"final_top3: {last['top3']:.6f}\n")
            f.write(f"final_top5: {last['top5']:.6f}\n")
            f.write(f"final_mfr: {last['mfr']:.6f}\n")
            f.write(f"final_mar: {last['mar']:.6f}\n")

    print(f"Saved metrics csv: {csv_path}")
    print(f"Saved model: {model_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
