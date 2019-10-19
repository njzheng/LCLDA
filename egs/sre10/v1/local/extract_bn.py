import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# import torch.optim as optim
import numpy as np
# import net
import dataset
import os
import sys

if sys.argv.__len__() <= 5 :
    print("The usage: local/extract_bn.py gpu_device_indx dnn_name test_path\
                         sub_sre_path \
                         sub_sre_test_path ...")
    print(sys.argv[:])
    exit()


# (Hyper parameters)
batchSize = 128  # batchsize的大小

gpu_device_indx = sys.argv[1]
device = torch.device("cuda:"+str(gpu_device_indx))
# device = torch.device("cuda:0")

# load model
dnn_path = sys.argv[2]
model = torch.load(dnn_path)

# training on the first cuda device
model.to(device)

# load test_dist class
test_path = sys.argv[3]
test_dist_path = os.path.join(test_path,"subivector_dist.ark")

# load train_dist class
train_path = sys.argv[4]
prob_trans_path = os.path.join(train_path,"prob_trans.mat")

csvector = []
# define the hook function
def forward_hook(module, input, output):
    csvector.append(output.data.cpu().numpy())


dataset_list = sys.argv[5:]
dataset_num = len(dataset_list)



dis_phone_num = 4

# load data
for dataset_indx in range(dataset_num):
    # intial the dist class for different training datasets
    train_dist_path = os.path.join(dataset_list[dataset_indx],"subivector_dist.ark")
    dist_trans= dataset.Input_transform(train_dist_path,test_dist_path,dis_phone_num,prob_trans_path)


    ds_path = os.path.join(dataset_list[dataset_indx],"subivector.scp")
    ds = dataset.SubivectorDataset(ds_path)

    utt_num = ds.__len__()
    print("The number of utters in %s is  %d." %(ds_path, utt_num))

    subivector_dim = np.size(ds.data_list[0],1)
    phone_num = np.size(ds.data_list[0],0)

    # read into training data loader
    ds_loader = DataLoader(ds, batch_size=batchSize,  # batch training
                          shuffle=False, num_workers=int(1))



    # set hook and store output in csvector
    csvector = []
    # Model.Evaluate where stop batch normalization upadate
    model.eval() 
    # add a forward hook to get l2 output
    handle = model.l2.register_forward_hook(forward_hook)
    # handle = model.tanh1.register_forward_hook(forward_hook)

    utt_list = []
    for i, data in enumerate(ds_loader, 0):
        uttid_list, real_cpu = data['utt'], data['subivector']

        utt_list.append(uttid_list)

        # do some transform of real_cpu to match the data in test dataset
        is_prob = False;
        # real_cpu_disp = dist_trans.transform_input(real_cpu, is_prob)
        real_cpu_disp = dist_trans.transform_norm_input(real_cpu,is_prob)

        # vectorize the subivector mat
        # input = real_cpu_disp.view(-1,(phone_num - dis_phone_num) * subivector_dim).to(device)
        input = real_cpu_disp.to(device)

        # forward + backward + optimize
        output = model(input)

        # print('batch: %d, test_eval' %(i + 1))

    dataset.write_vectors(utt_list, csvector, os.path.dirname(ds_path)+'/csvector.ark')
    # dataset.write_vectors(utt_list, output.cpu().detach().numpy(), os.path.dirname(sre_test_path)+'/output.ark')
    
    handle.remove()

print("Bottolnet feature extraction Finished! ")

