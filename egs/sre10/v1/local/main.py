import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np
import net
import dataset
import os
import sys

if sys.argv.__len__() != 7 :
    print("The usage: python3 main.py train_path sre_test_path sre_train_path sre_path uttid2int_dic_path gpu_device_indx")
    print(sys.argv[:])
    exit()


# (Hyper parameters)
batchSize = 128  # batchsize的大小
niter = 500  # epoch的最大值

# load data
train_path = sys.argv[1] #'/scratch/njzheng/myprogram/phone_vectors5/subivector.scp'
sre_test_path = sys.argv[2] #'/scratch/njzheng/myprogram/sre_test/subivector_test.scp'
sre_train_path = sys.argv[3] #'/scratch/njzheng/myprogram/sre_train/subivector_test.scp'
sre_path = sys.argv[4] #'/scratch/njzheng/myprogram/sre/subivector_test.scp'

train_dataset = dataset.SubivectorDataset(train_path)
sre_test_dataset = dataset.SubivectorDataset(sre_test_path)
sre_train_dataset = dataset.SubivectorDataset(sre_train_path)
sre_dataset = dataset.SubivectorDataset(sre_path)

utt_num = train_dataset.__len__()
train_len = int(0.9*utt_num)
valid_len = utt_num - train_len
train, valid = torch.utils.data.random_split(train_dataset, lengths=[train_len, valid_len])

subivector_dim = np.size(train_dataset.data_list[0],0)
phone_num = np.size(train_dataset.data_list[0],1)


# load spk2int list
uttid2int_dic_path = sys.argv[5] #'/scratch/njzheng/myprogram/phone_vectors5/utt2int'
uttid2int_dic = dataset.Uttid2int(uttid2int_dic_path)
spk_num = max(list(uttid2int_dic.uttid_dic.values())) + 1

# read into training data loader
train_loader = DataLoader(train, batch_size=batchSize,  # batch training
                          shuffle=True, num_workers=int(2))

valid_loader = DataLoader(valid, batch_size=batchSize,  # batch training
                          shuffle=False, num_workers=int(2))

sre_test_loader = DataLoader(sre_test_dataset, batch_size=batchSize,  # batch training
                          shuffle=False, num_workers=int(1))

sre_train_loader = DataLoader(sre_train_dataset, batch_size=batchSize,  # batch training
                          shuffle=False, num_workers=int(1))

sre_loader = DataLoader(sre_dataset, batch_size=batchSize,  # batch training
                          shuffle=False, num_workers=int(1))


# set hook and store output in csvector
csvector = []

def forward_hook(module, input, output):
    csvector.append(output.data.cpu().numpy())

# construct network
model = net.Net(phone_num * subivector_dim, 2048, 600, 600, spk_num)
optimizer = optim.Adam(list(model.parameters()), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.004)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)
criterion = nn.CrossEntropyLoss()  # loss function


gpu_device_indx = sys.argv[6]
device = torch.device("cuda:"+str(gpu_device_indx))
# device = torch.device("cuda:0")

# training on the first cuda device
model.to(device)
criterion.to(device)


# Model.Train
for epoch in range(niter):
    running_loss = 0.0
    ERROR_Train = []
    # model.train()
    for i, data in enumerate(train_loader, 0):
        model.zero_grad()  # 首先提取清零
        uttid_list, real_cpu = data['utt'], data['subivector']
        # convert uttid into int
        label_int = [uttid2int_dic.get_spkint(uttid) for uttid in uttid_list]

        # vectorize the subivector mat
        input = real_cpu.view(-1,phone_num * subivector_dim).to(device)

        # do not need to construct the one hot vectors
        label_cpu = torch.LongTensor(label_int)
        label =label_cpu.to(device)

        # inputv = Variable(input)
        # labelv = Variable(label)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        # print statistics
        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # print every 200 mini-batches
            print('[%d, %5d] loss: %.5f' %(epoch + 1, i + 1, running_loss / 100))
            # print(label[:10])
            # est_label = torch.max(F.softmax(output),1)[1]
            # print(est_label[:10])

        running_loss = 0.0

    # model.validate()
    vad_loss = 0.0
    for i, data in enumerate(valid_loader, 0):
        uttid_list, real_cpu = data['utt'], data['subivector']
        # convert uttid into int
        label_int = [uttid2int_dic.get_spkint(uttid) for uttid in uttid_list]

        # vectorize the subivector mat
        input = real_cpu.view(-1, phone_num * subivector_dim).to(device)

        # do not need to construct the one hot vectors
        label_cpu = torch.LongTensor(label_int)
        label = label_cpu.to(device)

        # forward + backward + optimize
        output = model(input)
        loss = criterion(output, label)

        # print statistics
        # print statistics
        vad_loss += loss.item()

    print('epoch: %d, vad loss: %.5f' %(epoch + 1, vad_loss))



sys.exit(1)


# Model.Evaluate for sre_test
# stop batch normalization upadate
model.eval() 
# add a forward hook to get l2 output
handle = model.l2_bn.register_forward_hook(forward_hook)
utt_list = []
for i, data in enumerate(sre_test_loader, 0):
    uttid_list, real_cpu = data['utt'], data['subivector']

    utt_list.append(uttid_list)
    # vectorize the subivector mat
    input = real_cpu.view(-1, phone_num * subivector_dim).to(device)

    # forward + backward + optimize
    output = model(input)

    print('batch: %d, test_eval' %(i + 1))



dataset.write_vectors(utt_list, csvector, os.path.dirname(sre_test_path)+'/csvector.ark')
# dataset.write_vectors(utt_list, output.cpu().detach().numpy(), os.path.dirname(sre_test_path)+'/output.ark')


# Model.Evaluate for sre sre_train  sre_test
csvector = []
utt_list = []
for i, data in enumerate(sre_train_loader, 0):
    uttid_list, real_cpu = data['utt'], data['subivector']

    utt_list.append(uttid_list)
    # vectorize the subivector mat
    input = real_cpu.view(-1, phone_num * subivector_dim).to(device)

    # forward + backward + optimize
    output = model(input)

    print('batch: %d, train_eval' %(i + 1))


dataset.write_vectors(utt_list, csvector, os.path.dirname(sre_train_path)+'/csvector.ark')
# dataset.write_vectors(utt_list, output.cpu().detach().numpy(), os.path.dirname(sre_train_path)+'/output.ark')


# Model.Evaluate for sre
csvector = []
utt_list = []
for i, data in enumerate(sre_loader, 0):
    uttid_list, real_cpu = data['utt'], data['subivector']

    utt_list.append(uttid_list)
    # vectorize the subivector mat
    input = real_cpu.view(-1, phone_num * subivector_dim).to(device)

    # forward + backward + optimize
    output = model(input)

    print('batch: %d, train_eval' %(i + 1))


handle.remove()

dataset.write_vectors(utt_list, csvector, os.path.dirname(sre_path)+'/csvector.ark')
# dataset.write_vectors(utt_list, output.cpu().detach().numpy(), os.path.dirname(sre_path)+'/output.ark')

print("Bottolnet feature extraction Finished! ")

