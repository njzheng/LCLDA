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
    print("The usage: python3 local/train_triplet_net.py gpu_device_indx sub_train_path dnn_name")
    print(sys.argv[:])
    exit()


# (Hyper parameters)
batchSize = 256  # batchsize的大小
niter = 250  # epoch的最大值


gpu_device_indx = sys.argv[1]
device = torch.device("cuda:"+str(gpu_device_indx))
# device = torch.device("cuda:1")

 
# load data
train_path = sys.argv[2] #'exp/subivectors_train_dnn$ivector_dim'

# discard the un voiced part
dis_phone_num = 4
print('The dis_phone_num is %d' %(dis_phone_num))


phone_num = 43
subnet_out_dim = 50
out_dim_sqrt = torch.tensor(subnet_out_dim).float().sqrt()


subnet_list = []
# Need loop ==============================================================
for phone_indx in range(34, phone_num + 1):
    dnn_name = sys.argv[3]+"."+str(phone_indx) # $sub_train_path/net.pkl

    # train_scp_path = os.path.join(train_path,"subivector.scp")
    train_scp_path = os.path.join(train_path,"phone_vector",
        "triplet_data."+str(phone_indx)+".scp")

    print('scp file is: %s' %(train_scp_path))


    train_dataset = dataset.SubivectorDataset(train_scp_path)
    subivector_dim = np.size(train_dataset.data_list[0],1)
    in_dim_sqrt = torch.tensor(subivector_dim).float().sqrt()

    train_utt_num = train_dataset.__len__()

    train_len = int(0.9*train_utt_num)
    valid_len = train_utt_num - train_len
    train, valid = torch.utils.data.random_split(train_dataset, lengths=[train_len, valid_len])


    print('The number of training and valid utters are  %d  %d' %(train_len, valid_len))
    print('The subivector_dim is %d, phone_indx is %d' %(subivector_dim, phone_indx))


    # read into training data loader
    train_loader = DataLoader(train, batch_size=batchSize,  # batch training
                              shuffle=True, num_workers=int(2))

    valid_loader = DataLoader(valid, batch_size=batchSize,  # batch training
                              shuffle=False, num_workers=int(2))

    # construct network
    model = net.TriSubnet(subivector_dim, 50, 50, subnet_out_dim)

    print(model)
    optimizer = optim.Adam(list(model.parameters()), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.004)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)

    margin = torch.tensor(0.2 * subnet_out_dim).sqrt();
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


            # forward + backward + optimize
            output1, output2, output3 = model(input)
            loss = criterion(output1, output2, output3)

            # print statistics
            vad_loss += loss.item()

            # compare the input and output
            in_dis = torch.norm(real_cpu[:,0,:]-real_cpu[:,2,:],dim=1) - torch.norm(real_cpu[:,0,:]-real_cpu[:,1,:],dim=1)    
            out_dis = torch.norm(output1-output3,dim=1) - torch.norm(output1-output2,dim=1)


        print('-----phone: %d epoch: %d, vad loss: %.6f, in_dis: %5f, out_dis: %5f' 
            %(phone_indx, epoch + 1, vad_loss, in_dis.mean()/in_dim_sqrt, out_dis.mean()/out_dim_sqrt))
        # print('epoch: %d, vad loss: %.6f' %(epoch + 1, vad_loss/valid_utt_num))
        vloss_recording[epoch] = vad_loss

        # if (epoch == 50):
        #     torch.save(model, dnn_name+'.50')


        # whether to stop early
        if (epoch > 100):
            if (vloss_recording[epoch] > vloss_recording[epoch-10]):
                print('Break at epoch %d compare now loss %.6f with early loss %.6f ' %(epoch,vloss_recording[epoch],vloss_recording[epoch-10] ) )
                break


    # save the model with dnn_name
    print("Save the Model's state_dict:")
    # Save:
    torch.save(model.state_dict(), dnn_name)
    subnet_list.append(model.state_dict())

# save the list total
list_name = sys.argv[3]+".list"
torch.save(subnet_list, list_name)

    # Load:
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()


    # torch.save({
    #             'modelA_state_dict': model.state_dict(),
    #             'modelB_state_dict': modelB.state_dict(),
                # 'optimizerA_state_dict': optimizerA.state_dict(),
    #             'optimizerB_state_dict': optimizerB.state_dict(),
    #             ...
    #             }, PATH)


    # modelA = TheModelAClass(*args, **kwargs)
    # modelB = TheModelBClass(*args, **kwargs)
    # optimizerA = TheOptimizerAClass(*args, **kwargs)
    # optimizerB = TheOptimizerBClass(*args, **kwargs)

    # checkpoint = torch.load(PATH)
    # modelA.load_state_dict(checkpoint['modelA_state_dict'])
    # modelB.load_state_dict(checkpoint['modelB_state_dict'])
    # optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
    # optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

subnet_list = []
for phone_indx in range(1, phone_num + 1):
    dnn_name = sys.argv[3]+"."+str(phone_indx) # $sub_train_path/net.pkl

    subnet_list.append(torch.load(dnn_name))

# save the list total
list_name = sys.argv[3]+".list"
torch.save(subnet_list, list_name)