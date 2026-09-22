import subprocess
from tqdm import tqdm
import time
import os, sys
import pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "GraceDate")

def _resolve_data_path(filename: str) -> str:
    if os.path.exists(filename):
        return filename
    return os.path.join(DEFAULT_DATA_DIR, filename)
project = sys.argv[1]
card = [0]
lst = list(range(len(pickle.load(open(_resolve_data_path(project + '.pkl'), 'rb')))))
singlenums = {'Time':5, 'Math':2, "Lang":10, "Chart":3, "Mockito":4, "Closure":1}
singlenum = singlenums[project]
totalnum = len(card) * singlenum
lr = 1e-2
seed = 0
batch_size = 60
for i in tqdm(range(int(len(lst) / totalnum) + 1)):
    jobs = []
    for j in range(totalnum):
        if totalnum * i + j >= len(lst):
            continue
        p = subprocess.Popen("python run.py %d %s %f %d %d"%(lst[totalnum * i + j], project, lr, seed, batch_size), shell=True)
        jobs.append(p)
        time.sleep(10)
    for p in jobs:
        p.wait()
p = subprocess.Popen("python sum.py %s %d %f %d"%(project, seed, lr, batch_size), shell=True)
p.wait()
subprocess.Popen("python watch.py %s %d %f %d"%(project, seed, lr, batch_size),shell=True)            