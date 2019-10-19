import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np
import os
import sys

import net
import dataset


if sys.argv.__len__() != 5 :
    print("The usage: python3 local/train_dnn.py gpu_device_indx sub_train_path dnn_name")
    print(sys.argv[:])
    exit()


# (Hyper parameters)
batchSize = 128  # batchsize的大小
niter = 500  # epoch的最大值


gpu_device_indx = sys.argv[1]
device = torch.device("cuda:"+str(gpu_device_indx))
# device = torch.device("cuda:1")

dnn_name = sys.argv[4] # $sub_train_path/net.pkl
 
# load data
train_path = sys.argv[2] #'exp/subivectors_train_dnn$ivector_dim'
test_path = sys.argv[3] #'exp/subivectors_train_dnn$ivector_dim'

# discard the un voiced part
dis_phone_num = 4
print('The dis_phone_num is %d' %(dis_phone_num))

# initial the dist class
train_dist_path = os.path.join(train_path,"subivector_dist.ark")
test_dist_path = os.path.join(test_path,"subivector_dist.ark")
prob_trans_path = os.path.join(train_path,"prob_trans.mat")
dist_trans= dataset.Input_transform(train_dist_path,test_dist_path,dis_phone_num,prob_trans_path)


# train_scp_path = os.path.join(train_path,"subivector.scp")
train_scp_path = os.path.join(train_path,"subivector.train.scp")
valid_scp_path = os.path.join(train_path,"subivector.valid.scp")



train_dataset = dataset.SubivectorDataset(train_scp_path)
valid_dataset = dataset.SubivectorDataset(valid_scp_path)

subivector_dim = np.size(train_dataset.data_list[0],1)
phone_num = np.size(train_dataset.data_list[0],0)

train_utt_num = train_dataset.__len__()
valid_utt_num = valid_dataset.__len__()
print('The number of training and valid utters are  %d  %d' %(train_utt_num, valid_utt_num))
print('The subivector_dim is %d, phone_num is %d' %(subivector_dim, phone_num))

# train_len = int(0.9*train_utt_num)
# valid_len = train_utt_num - train_len
# train, valid = torch.utils.data.random_split(train_dataset, lengths=[train_len, valid_len])


# load spk2int list
# train_utt2int_path = os.path.join(train_path,"utt2int")
train_utt2int_path = os.path.join(train_path,"utt2int.train")
valid_utt2int_path = os.path.join(train_path,"utt2int.valid")

train_utt2int = dataset.Uttid2int(train_utt2int_path)
valid_utt2int = dataset.Uttid2int(valid_utt2int_path)
spk_num = max(list(train_utt2int.uttid_dic.values())) + 1
spk_valid_num = max(list(valid_utt2int.uttid_dic.values())) + 1

assert spk_valid_num <= spk_num, 'valid spk id is larger than train spk id!'
print('The number of training spkers are  %d' %(spk_num))

# read into training data loader
train_loader = DataLoader(train_dataset, batch_size=batchSize,  # batch training
                          shuffle=True, num_workers=int(2))

valid_loader = DataLoader(valid_dataset, batch_size=batchSize,  # batch training
                          shuffle=False, num_workers=int(2))


# construct network
# model = net.Net(phone_num-dis_phone_num, subivector_dim, 1024, 400, 1024, spk_num)
model = net.CNN1d(phone_num-dis_phone_num, subivector_dim, 8, 16, 400, 400, spk_num)
print(model)
optimizer = optim.Adam(list(model.parameters()), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.004)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)
criterion = nn.CrossEntropyLoss()  # loss function


# training on the first cuda device
model.to(device)
criterion.to(device)


# Model.Train
vloss_recording = np.zeros(niter)
for epoch in range(niter):
    running_loss = 0.0
    model.train() 
    for i, data in enumerate(train_loader, 0):
        model.zero_grad()
        uttid_list, real_cpu = data['utt'], data['subivector']
        # convert uttid into int
        label_int = [train_utt2int.get_spkint(uttid) for uttid in uttid_list]

        is_prob = True
        # do some transform of real_cpu to match the data in test dataset
        # real_cpu_disp = dist_trans.transform_input(real_cpu,is_prob)
        real_cpu_disp = dist_trans.transform_norm_input(real_cpu,is_prob)

        # vectorize the subivector mat
        # input = real_cpu_disp.view(-1,(phone_num-dis_phone_num) * subivector_dim).to(device)
        input = real_cpu_disp.to(device)

        # do not need to construct the one hot vectors
        label_cpu = torch.LongTensor(label_int)
        label =label_cpu.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    print('epoch: %d, run loss: %.6f' %(epoch + 1, running_loss/train_utt_num))

    # model.validate()
    vad_loss = 0.0
    model.eval()
    for i, data in enumerate(valid_loader, 0):
        uttid_list, real_cpu = data['utt'], data['subivector']
        # convert uttid into int
        label_int = [valid_utt2int.get_spkint(uttid) for uttid in uttid_list]

        # do some transform of real_cpu to match the data in test dataset
        is_prob = True
        # real_cpu_disp = dist_trans.transform_input(real_cpu, is_prob)
        real_cpu_disp = dist_trans.transform_GMM_input(real_cpu,is_prob)

        # vectorize the subivector mat
        # input = real_cpu_disp.view(-1,(phone_num-dis_phone_num) * subivector_dim).to(device)
        input = real_cpu_disp.to(device)

        # do not need to construct the one hot vectors
        label_cpu = torch.LongTensor(label_int)
        label = label_cpu.to(device)

        # forward + backward + optimize
        output = model(input)
        loss = criterion(output, label)

        # print statistics
        vad_loss += loss.item()

    print('epoch: %d, vad loss: %.6f' %(epoch + 1, vad_loss))
    # print('epoch: %d, vad loss: %.6f' %(epoch + 1, vad_loss/valid_utt_num))
    vloss_recording[epoch] = vad_loss

    if (epoch == 50):
        torch.save(model, dnn_name+'.50')

    if (epoch == 100):
        torch.save(model, dnn_name+'.100')

    if (epoch == 150):
        torch.save(model, dnn_name+'.150')

    # whether to stop early
    if (epoch > 200):
        if (vloss_recording[epoch] > vloss_recording[epoch-20]):
            print('Break at epoch %d compare now loss %.6f with early loss %.6f ' %(epoch,vloss_recording[epoch],vloss_recording[epoch-20] ) )
            break


# save the model with dnn_name
print("Save the Model's state_dict:")
# Print model's state_dict

# for param_tensor in model.state_dict():
#     print(param_tensor,"\t",model.state_dict()[param_tensor].size())

print("optimizer's state_dict:")
# Print optimizer's state_dict
# for var_name in optimizer.state_dict():
#     print(var_name,"\t",optimizer.state_dict()[var_name])


torch.save(model, dnn_name)
# model = torch.load('model.pkl')