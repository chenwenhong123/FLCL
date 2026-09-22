import torch
from torch import optim
from Dataset import SumDataset
import os
from tqdm import tqdm
from Model import *
import numpy as np
#from annoy import AnnoyIndex
from nltk import word_tokenize
import pickle
from ScheduledOptim import ScheduledOptim
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
import random
import sys
import time
from datetime import datetime
# Prefer CWD, then fall back to Code/Default/GraceDate/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "GraceDate")
RESULT_ROOT = os.path.join(BASE_DIR, "result")

def _resolve_data_path(filename: str) -> str:
    if os.path.exists(filename):
        return filename
    return os.path.join(DEFAULT_DATA_DIR, filename)

def _result_dir(project: str) -> str:
    d = os.path.join(RESULT_ROOT, project)
    os.makedirs(d, exist_ok=True)
    return d
#import wandb
#wandb.init(project="codesum")
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

NlLen_map = {"Time":3900, "Math":4500, "Lang":280, "Chart": 2350, "Mockito":1780, "unknown":2200, "Closure":6000}
CodeLen_map = {"Time":1300, "Math":2700, "Lang":300, "Chart":5250, "Mockito":1176, "unknown":2800, "Closure":3000}
args = dotdict({
    'NlLen':NlLen_map[sys.argv[2]],
    'CodeLen':CodeLen_map[sys.argv[2]],
    'SentenceLen':10,
    'batch_size':60,
    'embedding_size':32,
    'WoLen':15,
    'Vocsize':100,
    'Nl_Vocsize':100,
    'max_step':3,
    'margin':0.5,
    'poolsize':50,
    'Code_Vocsize':100,
    'seed':0,
    'lr':1e-3
})
os.environ['PYTHONHASHSEED'] = str(args.seed)

def save_model(model, dirs = "checkpointcodeSearch"):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    torch.save(model.state_dict(), dirs + '/best_model.ckpt')


def load_model(model, dirs="checkpointcodeSearch"):
    assert os.path.exists(dirs + '/best_model.ckpt'), 'Weights for saved model not found'
    model.load_state_dict(torch.load(dirs + '/best_model.ckpt'))

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
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
        # MPS 对稀疏张量支持不完整
        # 其形状通常为 [batch, N, N]（3D）。因此仅对这种 3D sparse 做 densify，。
        if DEVICE.type == "mps" and getattr(tensor, "is_sparse", False):
            if tensor.dim() == 3:
                tensor = tensor.to_dense()
            else:
                tensor = tensor.to_dense()
        tensor = tensor.to(DEVICE)
    return tensor

def train(t = 5, p='Math'):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  
    random.seed(args.seed + t)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dev_set = SumDataset(args, "test", p, testid=t)
    val_set = SumDataset(args, "val", p, testid=t)
    data = pickle.load(open(_resolve_data_path(p + '.pkl'), 'rb'))
    dev_data = pickle.load(open(_resolve_data_path(p + '.pkl'), 'rb'))
    train_set = SumDataset(args, "train", testid=t, proj=p, lst=dev_set.ids + val_set.ids)
    numt = len(train_set.data[0])
    args.Code_Vocsize = len(train_set.Code_Voc)
    args.Nl_Vocsize = len(train_set.Nl_Voc)
    args.Vocsize = len(train_set.Char_Voc)

    print(dev_set.ids)
    model = NlEncoder(args)
    print(f'using device: {DEVICE}')
    model = model.to(DEVICE)
    maxl = 1e9
    optimizer = ScheduledOptim(optim.Adam(model.parameters(), lr=args.lr), args.embedding_size, 4000)
    maxAcc = 0
    minloss = 1e9
    rdic = {}
    brest = []
    bans = []
    batchn = []
    each_epoch_pred = {}
    for x in dev_set.Nl_Voc:
      rdic[dev_set.Nl_Voc[x]] = x
    for epoch in range(3):
        index = 0
        for dBatch in tqdm(train_set.Get_Train(args.batch_size)):
            if index == 0:
                accs = []
                loss = []
                model = model.eval()
                
                score2 = []
                for k, devBatch in tqdm(enumerate(val_set.Get_Train(len(val_set)))):
                        for i in range(len(devBatch)):
                            devBatch[i] = gVar(devBatch[i])
                        with torch.no_grad():
                            l, pre, _ = model(devBatch[0], devBatch[1], devBatch[2], devBatch[3], devBatch[4], devBatch[5], devBatch[6], devBatch[7])
                            resmask = torch.eq(devBatch[0], 2)
                            s = -pre#-pre[:, :, 1]
                            s = s.masked_fill(resmask == 0, 1e9)
                            pred = s.argsort(dim=-1)
                            pred = pred.data.cpu().numpy()
                            alst = []

                            for k in range(len(pred)): 
                                datat = data[val_set.ids[k]]
                                maxn = 1e9
                                lst = pred[k].tolist()[:resmask.sum(dim=-1)[k].item()]#score = np.sum(loss) / numt
                                #bans = lst
                                for x in datat['ans']:
                                    i = lst.index(x)
                                    maxn = min(maxn, i)
                                score2.append(maxn)

                each_epoch_pred[epoch] = lst
                score = score2[0]
                print('curr accuracy is ' + str(score) + "," + str(score2))
                if score2[0] == 0:
                    batchn.append(epoch)
                    

                if  maxl >= score:
                    brest = score2
                    bans = lst
                    maxl = score
                    print("find better score " + str(score) + "," + str(score2))
                    #save_model(model)
                    #torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
                model = model.train()
            for i in range(len(dBatch)):
                dBatch[i] = gVar(dBatch[i])
            loss, _, _ = model(dBatch[0], dBatch[1], dBatch[2], dBatch[3], dBatch[4], dBatch[5], dBatch[6], dBatch[7])
            print(loss.mean().item())
            optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()

            optimizer.step_and_update_lr()
            index += 1
    return brest, bans, batchn, each_epoch_pred



if __name__ == "__main__":
    args.lr = float(sys.argv[3])
    args.seed = int(sys.argv[4])
    args.batch_size = int(sys.argv[5])
    np.set_printoptions(threshold=sys.maxsize)
    res = {}    
    p = sys.argv[2]
    bug_id = int(sys.argv[1])
    run_t0 = time.perf_counter()
    res[bug_id] = train(bug_id, p)
    runtime_seconds = time.perf_counter() - run_t0
    run_ts = datetime.now().strftime("%m%d%H%M")
    out_dir = _result_dir(p)
    out_path = os.path.join(
        out_dir,
        '%sres%d_%d_%s_%s.pkl' % (p, bug_id, args.seed, args.lr, args.batch_size),
    )
    open(out_path, 'wb').write(pickle.dumps(res))

    print(f"runtime_seconds: {runtime_seconds:.3f}")
    rt_path = out_path[:-4] + ".runtime.txt" if out_path.endswith(".pkl") else out_path + ".runtime.txt"
    with open(rt_path, "w") as f:
        f.write(f"{runtime_seconds:.3f}\n")

    suffix = "_%d_%s_%s.runtime.txt" % (args.seed, args.lr, args.batch_size)
    prefix = "%sres" % p
    bug_times = []
    for name in sorted(os.listdir(out_dir)):
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        mid = name[len(prefix):-len(suffix)]
        if not mid.isdigit():
            continue
        try:
            with open(os.path.join(out_dir, name), "r") as f:
                bug_times.append((int(mid), float(f.read().strip())))
        except (OSError, ValueError):
            continue
    total_rt = sum(t for _, t in bug_times)
    summary_path = os.path.join(
        out_dir,
        "oneshot_runtime_summary_%d_%s_%s.txt" % (args.seed, args.lr, args.batch_size),
    )
    with open(summary_path, "w") as f:
        f.write("project: %s\n" % p)
        f.write("seed: %s\n" % args.seed)
        f.write("lr: %s\n" % args.lr)
        f.write("batch_size: %s\n" % args.batch_size)
        f.write("n_finished: %d\n" % len(bug_times))
        f.write("runtime_seconds: %.3f\n" % total_rt)
        f.write("runtime_hours: %.3f\n" % (total_rt / 3600.0))
        for bid, t in bug_times:
            f.write("bug_id_%d: %.3f\n" % (bid, t))
    print(f"Saved runtime: {rt_path}")
    print(f"Saved oneshot runtime summary: {summary_path}")



