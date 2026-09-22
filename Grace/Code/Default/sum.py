import pickle
import os
import sys
versionNum = {'Lang': 65, 'Time': 27, 'Chart': 26, 'Math': 106, 'Closure': 133, 'Mockito': 38,
                   'CommonsCli': 24, 'CommonsCodec': 22, 'CommonsCsv': 12, 'CommonsJXPath': 14,
                   'JacksonCore': 13, 'JacksonDatabind': 39, 'JacksonXml': 5, 'Jsoup': 63}
proj = sys.argv[1]
seed = int(sys.argv[2])
lr = float(sys.argv[3])
batch_size = int(sys.argv[4])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "result", proj)
os.makedirs(RESULT_DIR, exist_ok=True)

t = {}
for i in range(0, versionNum[proj]):
    in_path = os.path.join(RESULT_DIR, proj + 'res%d_%d_%s_%s.pkl'%(i, seed, lr, batch_size))
    if not os.path.exists(in_path):
        continue
    p = pickle.load(open(in_path, 'rb'))
    for x in p:
        t[x] = p[x]


    
out_path = os.path.join(RESULT_DIR, proj + 'res_%d_%s_%s.pkl'%(seed,lr, batch_size))
open(out_path, 'wb').write(pickle.dumps(t))
print(len(t))
