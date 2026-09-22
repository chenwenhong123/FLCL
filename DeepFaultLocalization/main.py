import bidirectional_rnn as birnn
import recurrent_network as myrnn
import multilayer_perceptron_one_hidden_layer as mlp
import multilayer_perceptron_two_hidden_layer as mlp2
import fc_based_1 as mlp_dfl_1
import fc_based_2 as mlp_dfl_2
import os
import time
import numpy as np
from config import *

  
# main run driver
def main():
    print(sub + '-' + v)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    train_path = os.path.join(dir,tech,sub,v,train_file)
    train_label_path = os.path.join(dir,tech,sub,v,train_label_file)
    if tech == "CrossDeepFL":
         train_path = os.path.join(dir,tech,sub+train_file)
         train_label_path = os.path.join(dir,tech,sub+train_label_file)
    test_path = os.path.join(dir,tech,sub,v,test_file)
    test_label_path = os.path.join(dir,tech,sub,v,test_label_file)
    group_path = os.path.join(dir,tech,group_dir,sub,v,group_file)
    susp_dir = os.path.join(out_dir,sub,v,tech)
    if not os.path.exists(susp_dir):
        os.makedirs(susp_dir)

    l = losses.index(loss)   #get index of loss function
    start_time = time.time()
    susp_path = os.path.join(susp_dir, model + '-' + losses[l])
    if model == "rnn":
        myrnn.run(train_path,train_label_path, test_path,test_label_path, group_path, susp_path, featureDistribution, l)
    elif model == "birnn":
        birnn.run(train_path,train_label_path, test_path,test_label_path, group_path, susp_path, featureDistribution, l)
    elif model == "mlp":
        mlp.run(train_path,train_label_path, test_path,test_label_path, group_path ,susp_path, l, featureNum=feature,nodeNum=feature)
    elif model == "mlp2":
        mlp2.run(train_path, train_label_path, test_path, test_label_path, group_path, susp_path, l, featureNum=feature,nodeNum=feature)
    elif model == "mlp_dfl_1":
        mlp_dfl_1.run(train_path,train_label_path, test_path,test_label_path, group_path ,susp_path, l, featureNum=feature,nodeNum=feature)
    elif model == "mlp_dfl_2":
        mlp_dfl_2.run(train_path,train_label_path, test_path,test_label_path, group_path ,susp_path, l, featureNum=feature,nodeNum=feature)
        #mlp2.run(train_path,train_label_path, test_path,test_label_path, group_path ,susp_path, l, featureNum=feature,nodeNum=feature)
    end_time = time.time()
    elapsed = end_time - start_time
    sec = max(0.0, float(elapsed))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - h * 3600 - m * 60
    hms = "%dh%02dm%06.3fs" % (h, m, s)
    rt_path = os.path.join(susp_dir, model + "-" + losses[l] + "-runtime.txt")
    with open(rt_path, "w") as rf:
        rf.write("runtime_sec=%.3f\nruntime=%s\n" % (sec, hms))
    print("runtime_sec=%.3f runtime=%s" % (sec, hms))
#main function execution
if __name__=='__main__':
    main()
