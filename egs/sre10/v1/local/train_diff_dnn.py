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

import loss
import net
import dataset


if sys.argv.__len__() != 4 :
    print("The usage: python3 local/train_dnn.py gpu_device_indx train_path dnn_name")
    print(sys.argv[:])
    exit()


# (Hyper parameters)
batchSize = 1024  # batchsize的大小
niter = 400  # epoch的最大值

gpu_device_indx = sys.argv[1]
device = torch.device("cuda:"+str(gpu_device_indx))
cpu_device = torch.device("cpu")
# device = torch.device("cuda:1")

dnn_name = sys.argv[3] # $sub_train_path/net.pkl
 
# load data
train_path = sys.argv[2] #'exp/subivectors_train_dnn$ivector_dim'

loss_name = "batch_hard_triplet_loss"


if(loss_name == "cross_entropy"):
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

    # train_scp_path = os.path.join(train_path,"subivector.scp")
    train_scp_path = os.path.join(train_path,"ivector.train.scp")
    valid_scp_path = os.path.join(train_path,"ivector.valid.scp")

    train_dataset = dataset.IvectorDataset(train_scp_path)
    valid_dataset = dataset.IvectorDataset(valid_scp_path)

    ivector_dim = np.size(train_dataset.data_list[0])

    train_utt_num = train_dataset.__len__()
    valid_utt_num = valid_dataset.__len__()
    print('The number of training and valid utters are  %d  %d' %(train_utt_num, valid_utt_num))
    print('The ivector_dim is %d' %(ivector_dim))

elif(loss_name == "triplet_loss"):
    # load spk2int list
    train_utt2int_path = os.path.join(train_path,"utt2int")

    train_utt2int = dataset.Uttid2int(train_utt2int_path)
    spk_num = max(list(train_utt2int.uttid_dic.values())) + 1

    print('The number of training spkers are  %d' %(spk_num))

    scp_path = os.path.join(train_path,"ivector.scp")

    # utt for select the same class and different class triplet samples
    uttid_list = list(train_utt2int.uttid_dic.values())
    # k_times*utt_in_class = triplet_number for each class
    all_dataset = dataset.Ivector_Triplet_Dataset(scp_path, uttid_list, spk_num, 6)
    ivector_dim = np.size(all_dataset.data_list[0])

    all_utt_num = all_dataset.__len__()

    train_utt_num = int(0.9*all_utt_num)
    valid_utt_num = all_utt_num - train_utt_num
    train_dataset, valid_dataset = torch.utils.data.random_split(all_dataset, lengths=[train_utt_num, valid_utt_num])

    valid_utt_num = valid_dataset.__len__()
    print('The number of training and valid triplet samples are  %d  %d' %(train_utt_num, valid_utt_num))
    print('The ivector_dim is %d' %(ivector_dim))

elif(loss_name == "batch_hard_triplet_loss"):
    # load spk2int list
    train_utt2int_path = os.path.join(train_path,"utt2int")

    train_utt2int = dataset.Uttid2int(train_utt2int_path)
    spk_num = max(list(train_utt2int.uttid_dic.values())) + 1

    print('The number of training spkers are  %d' %(spk_num))

    scp_path = os.path.join(train_path,"ivector.scp")

    # utt for select the same class and different class triplet samples
    uttid_list = list(train_utt2int.uttid_dic.values())
    # k_times*utt_in_class = triplet_number for each class
    all_dataset = dataset.Ivector_Batch_Hard_Triplet_Dataset(scp_path, uttid_list, spk_num)
    ivector_dim = np.size(all_dataset.data_list[0])

    all_utt_num = all_dataset.__len__()

    train_utt_num = int(0.9*all_utt_num)
    valid_utt_num = all_utt_num - train_utt_num
    train_dataset, valid_dataset = torch.utils.data.random_split(all_dataset, lengths=[train_utt_num, valid_utt_num])

    valid_utt_num = valid_dataset.__len__()
    print('The number of training and valid utt are  %d  %d' %(train_utt_num, valid_utt_num))
    print('The ivector_dim is %d' %(ivector_dim))


else:
    print('Err: Have no this loss type %s' %(loss_name))
    exit




# read into training data loader
train_loader = DataLoader(train_dataset, batch_size=batchSize,  # batch training
                          shuffle=True, num_workers=int(2))

valid_loader = DataLoader(valid_dataset, batch_size=batchSize,  # batch training
                          shuffle=False, num_workers=int(2))


# construct network
model = net.Diff_DNN(ivector_dim, 400, 200)
# model = net.Net(ivector_dim, 200)
print(model)
optimizer = optim.Adam(list(model.parameters()), lr=0.001, betas=(0.9, 0.999), weight_decay=0.004)
# optimizer = optim.ASGD(list(model.parameters()), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)

if(loss_name == "cross_entropy"):
    criterion = nn.CrossEntropyLoss()  # loss function
    criterion.to(device)

elif(loss_name == "triplet_loss"):
    ivector_dim_sqrt = torch.tensor(ivector_dim).float().sqrt()
    criterion = nn.TripletMarginLoss(margin=0.2*ivector_dim_sqrt, p=2, reduction='none') 
    criterion.to(device)

elif(loss_name == "batch_hard_triplet_loss"):
    ivector_dim_sqrt = torch.tensor(ivector_dim).float().sqrt()
    criterion = loss.Batch_hard_TripletMarginLoss(margin=0.2*ivector_dim_sqrt, p=2, k=1)
    criterion.to(device)
# torch.nn.functional.pairwise_distance(x1, x2, p=2.0, eps=1e-06, keepdim=False)



# training on the first cuda device
model.to(device)


# Model.Train
vloss_recording = np.zeros(niter)
if(loss_name == "cross_entropy"):
    for epoch in range(niter):
        running_loss = 0.0
        model.train() 
        for i, data in enumerate(train_loader):
            model.zero_grad()
            uttid_list, real_cpu = data['utt'], data['ivector']
            # convert uttid into int
            label_int = [train_utt2int.get_spkint(uttid) for uttid in uttid_list]


            input = real_cpu.to(device)

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
        for i, data in enumerate(valid_loader):
            uttid_list, real_cpu = data['utt'], data['ivector']
            # convert uttid into int
            label_int = [valid_utt2int.get_spkint(uttid) for uttid in uttid_list]

            # vectorize the subivector mat
            # input = real_cpu_disp.view(-1,(phone_num-dis_phone_num) * ivector_dim).to(device)
            input = real_cpu.to(device)

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


        # whether to stop early
        if (epoch > 300):
            if (vloss_recording[epoch] > vloss_recording[epoch-20]):
                print('Break at epoch %d compare now loss %.6f with early loss %.6f ' %(epoch,vloss_recording[epoch],vloss_recording[epoch-20] ) )
                break

elif(loss_name == "triplet_loss"):
    for epoch in range(niter):
        running_loss = 0.0
        model.train()

        if(epoch%10==1):
            all_dataset.renew()
            train_dataset, valid_dataset = torch.utils.data.random_split(all_dataset, lengths=[train_utt_num, valid_utt_num])
            train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=int(2))
        
        for i, (data1, data2, data3) in enumerate(train_loader):
            model.zero_grad()

            input1 = data1.to(device)
            input2 = data2.to(device)
            input3 = data3.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output1 = model(input1)
            output2 = model(input2)
            output3 = model(input3)

            loss_noreduce = criterion(output1, output2, output3)
            loss = torch.sum(loss_noreduce)/max(1.0,torch.sum(loss_noreduce>0))
            loss_noreduce += criterion(output2, output1, output3)
            loss += torch.sum(loss_noreduce)/max(1.0,torch.sum(loss_noreduce>0))
            loss /=2.0;
            loss.backward()

            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print('epoch: %d, run loss: %.6f' %(epoch + 1, running_loss/train_utt_num))

        # model.validate()
        vad_loss = 0.0
        model.eval()
        if(epoch%10==1):
            valid_loader = DataLoader(valid_dataset, batch_size=batchSize,shuffle=False, num_workers=int(2))

        for i, (data1, data2, data3) in enumerate(valid_loader):

            input1 = data1.to(device)
            input2 = data2.to(device)
            input3 = data3.to(device)

            # forward + backward + optimize
            output1 = model(input1)
            output2 = model(input2)
            output3 = model(input3)

            loss_noreduce = criterion(output1, output2, output3)
            loss = torch.sum(loss_noreduce)/max(1.0,torch.sum(loss_noreduce>0))
            loss_noreduce += criterion(output2, output1, output3)
            loss += torch.sum(loss_noreduce)/max(1.0,torch.sum(loss_noreduce>0))
            loss /=2.0;

            # print statistics
            vad_loss += loss.item()

        print('epoch: %d, vad loss: %.6f' %(epoch + 1, vad_loss))
        # print('epoch: %d, vad loss: %.6f' %(epoch + 1, vad_loss/valid_utt_num))
        vloss_recording[epoch] = vad_loss

        if (epoch == 50):
            torch.save(model, dnn_name+'.50')


        # whether to stop early
        if (epoch > 100):
            if (vloss_recording[epoch] > vloss_recording[epoch-20]):
                print('Break at epoch %d compare now loss %.6f with early loss %.6f ' %(epoch,vloss_recording[epoch],vloss_recording[epoch-20] ) )
                break

elif(loss_name == "batch_hard_triplet_loss"):
    for epoch in range(niter):
        running_loss = 0.0
        model.train()   
        for i, (data, spk) in enumerate(train_loader):
            model.zero_grad()

            input = data.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(input)

            loss = criterion(output, spk)
            loss.backward()

            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print('epoch: %d, run loss: %.6f' %(epoch + 1, running_loss/train_utt_num))

        # model.validate()
        vad_loss = 0.0
        model.eval()
        for i, (data, spk) in enumerate(valid_loader):

            input = data.to(device)

            # forward + backward + optimize
            output = model(input)

            loss = criterion(output, spk)

            # print statistics
            vad_loss += loss.item()

        print('epoch: %d, vad loss: %.6f' %(epoch + 1, vad_loss))
        # print('epoch: %d, vad loss: %.6f' %(epoch + 1, vad_loss/valid_utt_num))
        vloss_recording[epoch] = vad_loss

        if (epoch == 50):
            torch.save(model, dnn_name+'.50')


        # whether to stop early
        if (epoch > 100):
            if (vloss_recording[epoch] > vloss_recording[epoch-20]):
                print('Break at epoch %d compare now loss %.6f with early loss %.6f ' %(epoch,vloss_recording[epoch],vloss_recording[epoch-20] ) )
                break


print("Save the Model's state_dict:")
# Save:
# torch.save(model.state_dict(), dnn_name)
torch.save(model, dnn_name)

