import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# import torch.optim as optim
import numpy as np
import net
import dataset
import os
import sys

if sys.argv.__len__() <= 5 :
    print("The usage: local/extract_emb.py gpu_device_indx dnn_name test_path\
                         sub_sre_path \
                         sub_sre_test_path ...")
    print(sys.argv[:])
    exit()


# (Hyper parameters)
batchSize = 256  # batchsize的大小

gpu_device_indx = sys.argv[1]
device = torch.device("cuda:"+str(gpu_device_indx))
# device = torch.device("cuda:0")

subivector_dim = 80
# construct network
subnet_list_name = sys.argv[2]+".list"
# subnet_state_list = torch.load(subnet_list_name)
# subnet_list = []
# for i in range(len(subnet_state_list)):
#     subnet_temp = net.TriSubnet(subivector_dim, 50, 50, 50)
#     subnet_temp.load_state_dict(subnet_state_list[i])
#     subnet_temp.to(device)
#     subnet_list.append(subnet_temp)

# load model
dnn_name = sys.argv[2]+".tri_tot"
out_dim = 400
# model = net.TriTotnet(subivector_dim, subnet_list, 600, 400, out_dim)
# model.load_state_dict(torch.load(dnn_name))
model = torch.load(dnn_name)
# Model.Evaluate where stop batch normalization upadate
model.eval() 

# training on the first cuda device
model.to(device)


# csvector = []
# # define the hook function
# def forward_hook(module, input, output):
#     csvector.append(output.data.cpu().numpy())


dataset_list = sys.argv[3:]
dataset_num = len(dataset_list)

dis_phone_num = 4


# load data
for dataset_indx in range(dataset_num):
    ds_path = os.path.join(dataset_list[dataset_indx],"subivector.scp")
    ds = dataset.SubivectorDataset(ds_path)

    utt_num = ds.__len__()
    print("The number of utters in %s is  %d." %(ds_path, utt_num))

    subivector_dim = np.size(ds.data_list[0],1)
    phone_num = np.size(ds.data_list[0],0)

    # read into training data loader
    ds_loader = DataLoader(ds, batch_size=batchSize,  # batch training
                          shuffle=False, num_workers=int(1))

    utt_list = []
    csvector = []
    for i, data in enumerate(ds_loader, 0):
        uttid_list, real_cpu = data['utt'], data['subivector']

        utt_list.append(uttid_list)

        # vectorize the subivector mat
        input = real_cpu.to(device)

        # forward + backward + optimize
        output = model.forward_single(input)

        csvector.append(output.data.cpu().numpy())
        # print('batch: %d, test_eval' %(i + 1))

    # dataset.write_vectors(utt_list, csvector, os.path.dirname(ds_path)+'/csvector.ark')
    dataset.write_vectors(utt_list, csvector, os.path.dirname(ds_path)+'/csvector.ark')
    
print("Triplet feature extraction Finished! ")

