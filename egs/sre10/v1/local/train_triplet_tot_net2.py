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


if sys.argv.__len__() != 4 :
    print("The usage: python3 local/train_triplet_tot_net.py gpu_device_indx sub_train_path dnn_name")
    print(sys.argv[:])
    exit()


# (Hyper parameters)
batchSize = 512  # batchsize的大小
niter = 300  # epoch的最大值


gpu_device_indx = sys.argv[1]
device = torch.device("cuda:"+str(gpu_device_indx))
# device = torch.device("cuda:1")

# load data
train_path = sys.argv[2] 

# discard the un voiced part
dis_phone_num = 4
print('The dis_phone_num is %d' %(dis_phone_num))

phone_num = 43

# train_scp_path = os.path.join(train_path,"subivector.scp")
train_scp_path = os.path.join(train_path,"phone_vector",
    "triplet_data_all.scp")

print('scp file is: %s' %(train_scp_path))


train_dataset = dataset.SubivectorDataset(train_scp_path)

# the original is 43*80
subivector_dim = np.size(train_dataset.data_list[0],1)
subivector_dim = int(subivector_dim/phone_num);

train_utt_num = train_dataset.__len__()

train_len = int(0.9*train_utt_num)
valid_len = train_utt_num - train_len
train, valid = torch.utils.data.random_split(train_dataset, lengths=[train_len, valid_len])


print('The number of training and valid utters are  %d  %d' %(train_len, valid_len))
print('The subivector_dim is %d' %(subivector_dim))


# read into training data loader
train_loader = DataLoader(train, batch_size=batchSize,  # batch training
                          shuffle=True, num_workers=int(2))

valid_loader = DataLoader(valid, batch_size=batchSize,  # batch training
                          shuffle=False, num_workers=int(2))


in_dim_sqrt = torch.tensor(subivector_dim*phone_num).float().sqrt()
out_dim = 400
out_dim_sqrt = torch.tensor(out_dim).float().sqrt()
model = net.TriTotnet2(subivector_dim*phone_num, 1024, 400, out_dim)

print(model)
optimizer = optim.Adam(list(model.parameters()), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.004)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)

margin = torch.tensor(0.2 * out_dim).sqrt();
criterion = nn.TripletMarginLoss(margin=margin, p = 2.0)  # loss function (margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')


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

        # is 3-rows mat
        input = real_cpu.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output1, output2, output3 = model(input)
        loss = criterion(output1, output2, output3)
        loss = (loss + criterion(output2, output1, output3))/2

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

        input = real_cpu.to(device)
        in_dis = torch.norm(real_cpu[:,0,:]-real_cpu[:,2,:],dim=1) - torch.norm(real_cpu[:,0,:]-real_cpu[:,1,:],dim=1)

        # forward + backward + optimize
        output1, output2, output3 = model(input)
        loss = criterion(output1, output2, output3)
        loss = (loss + criterion(output2, output1, output3))/2
        
        dis = torch.norm(output1-output3,dim=1) - torch.norm(output1-output2,dim=1)

        # print statistics
        vad_loss += loss.item()

    print('-----epoch: %d, vad loss: %.6f, input_mean_dis: %.5f, output mean dis: %.5f, min dis: %.5f'
        %( epoch + 1, vad_loss, in_dis.mean()/in_dim_sqrt, dis.mean()/out_dim_sqrt, dis.min()/out_dim_sqrt ))
    # print('epoch: %d, vad loss: %.6f' %(epoch + 1, vad_loss/valid_utt_num))
    vloss_recording[epoch] = vad_loss

    # if (epoch == 50):
    #     torch.save(model, dnn_name+'.50')

    # if (epoch == 100):
    #     torch.save(model, dnn_name+'.100')

    # whether to stop early
    if (epoch > 20):
        if (vloss_recording[epoch] > vloss_recording[epoch-10]):
            print('Break at epoch %d compare now loss %.6f with early loss %.6f ' %(epoch,vloss_recording[epoch],vloss_recording[epoch-10] ) )
            break


# save the model with dnn_name
print("Save the Model's state_dict:")
# Save:
dnn_name = sys.argv[3]+".tri_tot"
# save total model
torch.save(model, dnn_name) 
# torch.save(model.state_dict(), dnn_name)

