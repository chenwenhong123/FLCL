import copy
import csv
import os
import pickle
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

# 持续学习运行默认关闭 Dataset 词表打印，避免刷屏
os.environ.setdefault("GRACE_SHOW_VOCAB", "0")
from Dataset import SumDataset
from Model import NlEncoder
from ScheduledOptim import ScheduledOptim


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "GraceDate")
RESULT_ROOT = os.path.join(BASE_DIR, "result")

NlLen_map = {
    "Time": 3900,
    "Math": 4500,
    "Lang": 280,
    "Chart": 2350,
    "Mockito": 1780,
    "Closure": 3000,
    "unknown": 2200,
}
CodeLen_map = {
    "Time": 1300,
    "Math": 2700,
    "Lang": 300,
    "Chart": 5250,
    "Mockito": 1176,
    "Closure": 6000,
    "unknown": 2800,
}


def _resolve_data_path(filename: str) -> str:
    if os.path.exists(filename):
        return filename
    return os.path.join(DEFAULT_DATA_DIR, filename)


def _result_dir(project: str) -> str:
    d = os.path.join(RESULT_ROOT, project)
    os.makedirs(d, exist_ok=True)
    return d


def _run_out_dir(project: str, run_id: str) -> str:
    d = os.path.join(_result_dir(project), run_id)
    os.makedirs(d, exist_ok=True)
    return d


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


def gVar(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = gVar(data[i])
        tensor = data
    else:
        assert isinstance(tensor, torch.Tensor)
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.to(DEVICE)
    return tensor


def init_args(project: str, lr: float, seed: int, batch_size: int):
    return dotdict(
        {
            "NlLen": NlLen_map[project],
            "CodeLen": CodeLen_map[project],
            "SentenceLen": 10,
            "batch_size": batch_size,
            "embedding_size": 32,
            "WoLen": 15,
            "Vocsize": 100,
            "Nl_Vocsize": 100,
            "max_step": 3,
            "margin": 0.5,
            "poolsize": 50,
            "Code_Vocsize": 100,
            "seed": seed,
            "lr": lr,
        }
    )


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 追求吞吐：固定形状输入下 benchmark 往往更快
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_one_id_dataset(args, project: str, bug_id: int):
    ds = SumDataset(args, "val", proj=project, testid=bug_id)
    args.Code_Vocsize = len(ds.Code_Voc)
    args.Nl_Vocsize = len(ds.Nl_Voc)
    args.Vocsize = len(ds.Char_Voc)
    return ds


def get_single_batch(ds: SumDataset):
    batch = next(ds.Get_Train(len(ds)))
    for i in range(len(batch)):
        batch[i] = gVar(batch[i])
    return batch


def get_batch_for_id(
    args,
    project: str,
    bug_id: int,
    ds_cache: Dict[int, SumDataset],
    batch_cache: Dict[int, List[torch.Tensor]],
    cache_on_device: bool,
):
    """Lazy cache by bug id to reduce repeated Dataset construction and IO."""
    if bug_id not in ds_cache:
        ds_cache[bug_id] = build_one_id_dataset(args, project, bug_id)
    if bug_id not in batch_cache:
        raw_batch = next(ds_cache[bug_id].Get_Train(len(ds_cache[bug_id])))
        if cache_on_device:
            raw_batch = [gVar(x) for x in raw_batch]
        batch_cache[bug_id] = raw_batch
    cached = batch_cache[bug_id]
    if cache_on_device:
        return cached
    return [gVar(x) for x in cached]


def rank_metrics_from_batch(model: NlEncoder, batch, ans_list: List[int]) -> Tuple[int, float]:
    model.eval()
    with torch.no_grad():
        _, pre, _ = model(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7])
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


def train_on_one_id(
    model: NlEncoder,
    optimizer: ScheduledOptim,
    args,
    project: str,
    bug_id: int,
    epochs: int,
    ds_cache: Dict[int, SumDataset],
    batch_cache: Dict[int, List[torch.Tensor]],
    cache_on_device: bool,
):
    model.train()
    for _ in range(epochs):
        batch = get_batch_for_id(args, project, bug_id, ds_cache, batch_cache, cache_on_device)
        loss, _, _ = model(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7])
        optimizer.zero_grad()
        loss = loss.mean()
        loss.backward()
        optimizer.step_and_update_lr()


def calc_summary(ranks: List[int], mars: List[float]) -> Dict[str, float]:
    n = len(ranks)
    if n == 0:
        return {
            "n": 0,
            "top1": 0.0,
            "top3": 0.0,
            "top5": 0.0,
            "top1_count": 0,
            "top3_count": 0,
            "top5_count": 0,
            "mfr": 0.0,
            "mar": 0.0,
        }
    top1_count = sum(1 for r in ranks if r == 0)
    top3_count = sum(1 for r in ranks if r < 3)
    top5_count = sum(1 for r in ranks if r < 5)
    return {
        "n": n,
        "top1": float(top1_count / n),
        "top3": float(top3_count / n),
        "top5": float(top5_count / n),
        "top1_count": top1_count,
        "top3_count": top3_count,
        "top5_count": top5_count,
        "mfr": float(np.mean(ranks)),
        "mar": float(np.mean(mars)) if len(mars) > 0 else 0.0,
    }


def main():
    if len(sys.argv) < 5:
        print(
            "Usage: python continual_run.py <Project> <lr> <seed> <batch_size> [warmup_ids=10] [base_epochs=5] [inc_epochs=1]"
        )
        sys.exit(1)

    project = sys.argv[1]
    lr = float(sys.argv[2])
    seed = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    warmup_ids = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    base_epochs = int(sys.argv[6]) if len(sys.argv) > 6 else 15
    inc_epochs = int(sys.argv[7]) if len(sys.argv) > 7 else 2
    run_t0 = time.perf_counter()
    run_ts = datetime.now().strftime("%m%d%H%M")
    run_id = f"continual_base{base_epochs}_inc{inc_epochs}_lr{lr}_bs{batch_size}_warm{warmup_ids}_ts{run_ts}"

    set_seed(seed)
    args = init_args(project, lr, seed, batch_size)
    print(f"using device: {DEVICE}")
    # 小图任务（如 Lang）缓存到设备端可明显减少拷贝开销；大图默认只做CPU侧缓存防止OOM
    cache_on_device = DEVICE.type == "cuda" and (args.NlLen + args.CodeLen) <= 1200
    ds_cache: Dict[int, SumDataset] = {}
    batch_cache: Dict[int, List[torch.Tensor]] = {}
    print(f"batch_cache_on_device: {cache_on_device}")

    data = pickle.load(open(_resolve_data_path(project + ".pkl"), "rb"))
    total_ids = len(data)
    if total_ids <= warmup_ids:
        raise ValueError(f"{project} only has {total_ids} ids, warmup_ids={warmup_ids} is too large.")

    warmup_range = list(range(warmup_ids))
    stream_range = list(range(warmup_ids, total_ids))

    best_base = None
    best_state = None
    best_score = (-1, -1, float("inf"))  # top1, top3, mfr

    # Step1: 前10个ID逐ID重训，选择最优基础模型
    for bid in tqdm(warmup_range, desc="Warmup Base Selection", ncols=100):
        set_seed(seed + bid)
        # 先构建数据集以刷新 args.{Code/Nl/Voc}size，再初始化模型，避免 state_dict 尺寸不一致
        ds = build_one_id_dataset(args, project, bid)
        ds_cache[bid] = ds
        model = NlEncoder(args).to(DEVICE)
        optimizer = ScheduledOptim(optim.Adam(model.parameters(), lr=args.lr), args.embedding_size, 4000)
        train_on_one_id(
            model, optimizer, args, project, bid, base_epochs, ds_cache, batch_cache, cache_on_device
        )
        batch = get_batch_for_id(args, project, bid, ds_cache, batch_cache, cache_on_device)
        rank, mar = rank_metrics_from_batch(model, batch, data[bid]["ans"])
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

    # Step2: 加载最佳基础模型，做增量学习
    model = NlEncoder(args).to(DEVICE)
    model.load_state_dict(best_state)
    optimizer = ScheduledOptim(optim.Adam(model.parameters(), lr=args.lr), args.embedding_size, 4000)

    # 记录每个任务初次学完时的性能，用于 BWT
    first_learn_top1: Dict[int, float] = {}
    first_learn_top3: Dict[int, float] = {}

    # 先补齐 warmup 任务的“初次性能”
    for bid in warmup_range:
        batch = get_batch_for_id(args, project, bid, ds_cache, batch_cache, cache_on_device)
        rank, _ = rank_metrics_from_batch(model, batch, data[bid]["ans"])
        first_learn_top1[bid] = 1.0 if rank == 0 else 0.0
        first_learn_top3[bid] = 1.0 if rank < 3 else 0.0

    rows = []
    for t in tqdm(stream_range, desc="Incremental Stream", ncols=100):
        # 在线指标：更新前后都可以算，这里按“更新后的当前任务命中”定义 Online-Top1@t
        train_on_one_id(
            model, optimizer, args, project, t, inc_epochs, ds_cache, batch_cache, cache_on_device
        )
        cur_batch = get_batch_for_id(args, project, t, ds_cache, batch_cache, cache_on_device)
        cur_rank, cur_mar = rank_metrics_from_batch(model, cur_batch, data[t]["ans"])
        online_top1 = 1.0 if cur_rank == 0 else 0.0

        # 当前任务初次学习后的性能（作为 BWT 参照点）
        first_learn_top1[t] = 1.0 if cur_rank == 0 else 0.0
        first_learn_top3[t] = 1.0 if cur_rank < 3 else 0.0

        # ACC@t：截至 t 所有已见任务平均质量
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

        # BWT@t：对旧任务(0..t-1)比较“当前性能 - 初次学习性能”
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
