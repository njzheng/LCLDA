import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================

class Subnet(nn.Module):
    def __init__(self, in_dim, c_hidden_1, c_hidden_2, out_dim):
        super(Subnet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, c_hidden_1),
                        nn.ReLU(),
                        # nn.BatchNorm1d(c_hidden_1),

                        nn.Linear(c_hidden_1, c_hidden_2),
                        nn.ReLU(),
                        # nn.BatchNorm1d(c_hidden_2),

                        nn.Linear(c_hidden_2, out_dim)
                        )
        self.out_dim_sqrt = torch.tensor(out_dim).float().sqrt()
    def forward(self, x):
        # one part: anchor, positive, negative
        output = self.fc(x)
        output = F.normalize(output, p=2)*self.out_dim_sqrt
        return output

class TriSubnet(nn.Module):
    def __init__(self, in_dim, c_hidden_1, c_hidden_2, out_dim):
        super(TriSubnet, self).__init__()
        # self.fc = Subnet(in_dim, c_hidden_1, c_hidden_2, out_dim)
        self.fc = nn.Sequential(nn.Linear(in_dim, c_hidden_1),
                        nn.ReLU(),
                        # nn.BatchNorm1d(c_hidden_1),

                        nn.Linear(c_hidden_1, c_hidden_2),
                        nn.ReLU(),
                        # nn.BatchNorm1d(c_hidden_2),

                        nn.Linear(c_hidden_2, out_dim)
                        )
        self.out_dim_sqrt = torch.tensor(out_dim).float().sqrt()

    def forward(self, x):
        # three part: anchor, positive, negative
        output1 = self.fc(x[:,0,:])
        output1 = F.normalize(output1, p=2)*self.out_dim_sqrt
        output2 = self.fc(x[:,1,:])
        output2 = F.normalize(output2, p=2)*self.out_dim_sqrt
        output3 = self.fc(x[:,2,:])
        output3 = F.normalize(output3, p=2)*self.out_dim_sqrt
        return output1, output2, output3

    def forward_single(self, x):
        # three part: anchor, positive, negative
        output = self.fc(x)
        output = F.normalize(output, p=2)*self.out_dim_sqrt
        return output
# ==============================================================================

class TriTotnet(nn.Module):
    def __init__(self, in_dim, subnet_list, t_hidden_1, t_hidden_2, out_dim, device):
        super(TriTotnet, self).__init__()

        self.subnet = subnet_list
        self.phone_num = len(subnet_list)
        self.subnet_in_dim = in_dim
        self.subnet_out_dim = subnet_list[0].fc[-1].out_features
        # print(self.subnet_out_dim)
        # print(self.phone_num)
        self.fc = nn.Sequential(
            nn.Linear(self.subnet_out_dim*self.phone_num, t_hidden_1),#50*43
            nn.ReLU(),
            # nn.BatchNorm1d(c_hidden_1),

            nn.Linear(t_hidden_1, t_hidden_2),
            nn.ReLU(),
            # nn.BatchNorm1d(c_hidden_2),

            nn.Linear(t_hidden_2, out_dim)
            )
        self.out_dim_sqrt = torch.tensor(out_dim).float().sqrt()
        
        for i in range(self.phone_num):
            self.subnet[i].to(device)
            self.subnet[i].train()

    def forward(self, x, device):
        # three part: anchor, positive, negative
        batch_size = x.shape[0]
        ivector_dim = self.subnet_in_dim

        output_subnet1 = torch.zeros(batch_size, self.phone_num, self.subnet_out_dim,device=device)
        output_subnet2 = torch.zeros(batch_size, self.phone_num, self.subnet_out_dim,device=device)
        output_subnet3 = torch.zeros(batch_size, self.phone_num, self.subnet_out_dim,device=device)

        for i in range(self.phone_num):
            output_subnet1[:,i,:] = self.subnet[i].forward_single(x[:,0,i*ivector_dim:(i+1)*ivector_dim]) 
            output_subnet2[:,i,:] = self.subnet[i].forward_single(x[:,1,i*ivector_dim:(i+1)*ivector_dim]) 
            output_subnet3[:,i,:] = self.subnet[i].forward_single(x[:,2,i*ivector_dim:(i+1)*ivector_dim]) 
        
        output1 = self.fc(output_subnet1.view(-1,self.subnet_out_dim*self.phone_num))
        output2 = self.fc(output_subnet2.view(-1,self.subnet_out_dim*self.phone_num))
        output3 = self.fc(output_subnet3.view(-1,self.subnet_out_dim*self.phone_num))

        output1 = F.normalize(output1, p=2)*self.out_dim_sqrt
        output2 = F.normalize(output2, p=2)*self.out_dim_sqrt
        output3 = F.normalize(output3, p=2)*self.out_dim_sqrt
            
        return output1, output2, output3

    def forward_single(self, x, device):
        # only one input for evaluation
        batch_size = x.shape[0]
        phone_num = x.shape[1]
        ivector_dim_sqrt = torch.tensor(x.shape[2]).float().sqrt()
        # input normalization
        x = F.normalize(x, p=2, dim=2)*ivector_dim_sqrt
        
        output_subnet = torch.zeros(batch_size, self.phone_num, self.subnet_out_dim,device=device)
        for i in range(self.phone_num):
            output_subnet[:,i,:] = self.subnet[i].forward_single(x[:,i,:]) #??
        
        output = self.fc(output_subnet.view(-1,self.subnet_out_dim*self.phone_num))      
        output = F.normalize(output, p=2)*self.out_dim_sqrt

        return output
# ==============================================================================

class TriTotnet2(nn.Module):
    def __init__(self, in_dim, t_hidden_1, t_hidden_2, out_dim):
        super(TriTotnet2, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_dim, t_hidden_1),#50*43
            # nn.BatchNorm1d(t_hidden_1),
            nn.Tanh(),

            nn.Linear(t_hidden_1, t_hidden_2),
            nn.Tanh(),
            # nn.ReLU(),
            # nn.BatchNorm1d(c_hidden_2),

            # nn.Linear(t_hidden_2, t_hidden_3),
            # nn.ReLU(),
            # # nn.BatchNorm1d(c_hidden_2),

            nn.Linear(t_hidden_2, out_dim)
            )
        self.out_dim_sqrt = torch.tensor(out_dim).float().sqrt()


    def forward(self, x):
        # three part: anchor, positive, negative
        batch_size = x.shape[0]
      
        output1 = self.fc(x[:,0,:])
        output2 = self.fc(x[:,1,:])
        output3 = self.fc(x[:,2,:])

        output1 = F.normalize(output1, p=2)*self.out_dim_sqrt
        output2 = F.normalize(output2, p=2)*self.out_dim_sqrt
        output3 = F.normalize(output3, p=2)*self.out_dim_sqrt
            
        return output1, output2, output3

    def forward_single(self, x):
        # only one input for evaluation
        batch_size = x.shape[0]
        phone_num = x.shape[1]
        ivector_dim_sqrt = torch.tensor(x.shape[2]).float().sqrt()
        # input normalization
        x = F.normalize(x, p=2, dim=2)*ivector_dim_sqrt
             
        output = self.fc(x.view(batch_size,-1))      
        output = F.normalize(output, p=2)*self.out_dim_sqrt

        return output
# ==============================================================================

class CNN1d(nn.Module):
    def __init__(self, phone_num, in_dim, c_hidden_1, c_hidden_2,n_hidden_1, n_hidden_2,out_dim):
        super(CNN1d, self).__init__()
        self.kernel_size = 8
        self.stride = 1
        self.conv1 = nn.Conv1d(phone_num, c_hidden_1, kernel_size=self.kernel_size, stride=1, padding=0)
        self.cbn1 = nn.BatchNorm1d(c_hidden_1)
        self.ctanh1 = nn.Tanh()

        self.conv2 = nn.Conv1d(c_hidden_1, c_hidden_2, kernel_size=self.kernel_size, stride=1, padding=0)
        self.cbn2 = nn.BatchNorm1d(c_hidden_2)
        self.ctanh2 = nn.Tanh()

        #  compute dim for dnn input int(W-F+2P/S)+1
        dnn_dim = int(math.floor((in_dim-self.kernel_size)/self.stride + 1))
        dnn_dim = int(math.floor((dnn_dim-self.kernel_size)/self.stride + 1))
        self.dnn_dim = int(dnn_dim * c_hidden_2)


        # self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.l1 = nn.Linear(self.dnn_dim, n_hidden_1)
        self.l1_bn = nn.BatchNorm1d(n_hidden_1)
        self.tanh1 = nn.Tanh()
        # self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.l2_bn = nn.BatchNorm1d(n_hidden_2)
        self.tanh2 = nn.Tanh()
        # self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(n_hidden_2, out_dim)

        self.dropout = nn.Dropout(p=0.2)
        self.in_dim = in_dim
        self.phone_num = phone_num


    def forward(self, x):
        x = self.ctanh1(self.cbn1(self.conv1(x)))
        x = self.ctanh2(self.cbn2(self.conv2(x)))

        x = x.view(-1, self.dnn_dim)

        x = self.l1(x)
        x = self.l1_bn(x)
        x = self.tanh1(x)

        x = self.l2(x)
        x = self.l2_bn(x)
        x = self.tanh2(x)
        # x = self.relu2(x)
        x = self.l3(x)

        return x

# ==============================================================================

class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()
        self.l1 = nn.Linear(in_dim, out_dim,bias=False)
        # self.l1_bn = nn.BatchNorm1d(n_hidden_1)
        # self.tanh1 = nn.Tanh()
        # self.relu1 = nn.ReLU()
        
        # self.l2 = nn.Linear(n_hidden_1, n_hidden_2,bias=False)
        # # self.l2_bn = nn.BatchNorm1d(n_hidden_2)
        # # self.tanh2 = nn.Tanh()
        # self.relu2 = nn.ReLU()
        
        # self.l3 = nn.Linear(n_hidden_2, n_hidden_3,bias=False)
        # # self.l3_bn = nn.BatchNorm1d(n_hidden_3)
        # # self.tanh3 = nn.Tanh()
        # self.relu3 = nn.ReLU()
        # self.l4 = nn.Linear(n_hidden_1, out_dim,bias=False)

        self.in_dim = in_dim

    def forward(self, x):
        x = self.l1(x)
        # x = self.relu1(x)
        # x = self.dropout(x)

        # x = self.l2(x)
        # x = self.relu2(x)
        # x = self.dropout(x)

        # x = self.l3(x)
        # x = self.relu3(x)
        # x = self.dropout(x)

        # x = self.l4(x)
        ivector_dim_sqrt = torch.tensor(x.shape[1]).float().sqrt()
        x = F.normalize(x, p=2, dim=1)*ivector_dim_sqrt

        return x

    def hookl2(self, x):
        x = self.l1(x)
        # x = self.relu1(x)
        # x = self.dropout(x)

        # x = self.l2(x)
        # x = self.l3(x)
        ivector_dim_sqrt = torch.tensor(x.shape[1]).float().sqrt()
        x = F.normalize(x, p=2, dim=1)*ivector_dim_sqrt

        return x

# ==============================================================================
# to avoid the informaiton loss in relu(), we adopt relu(x) and relu(-x) together
class Diff_DNN(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2):
        super(Diff_DNN, self).__init__()
        self.l1 = nn.Linear(in_dim, n_hidden_1,bias=False)
        # self.l1_bn = nn.BatchNorm1d(n_hidden_1)
        # self.tanh1 = nn.Tanh()
        self.relu1 = nn.ReLU()

        self.l2_a = nn.Linear(n_hidden_1, n_hidden_2)
        self.l2_b = nn.Linear(n_hidden_1, n_hidden_2)
        
        # self.l2_bn = nn.BatchNorm1d(n_hidden_2)
        # self.tanh2 = nn.Tanh()
        # self.relu2 = nn.ReLU()
        
        # self.l3 = nn.Linear(4*n_hidden_2, n_hidden_3)
        # # self.l3_bn = nn.BatchNorm1d(n_hidden_3)
        # # self.tanh3 = nn.Tanh()
        # self.relu3 = nn.ReLU()
        
        # self.l4 = nn.Linear(n_hidden_3, out_dim)
        
        # self.dropout = nn.Dropout(p=0.2)
        self.in_dim = in_dim

    def forward(self, x):
        x_1 = self.l1(x)
        x_1_a = self.relu1(x_1)
        # x_1_b = self.relu1(-x_1)
        

        x_2_a = self.l2_a(x_1_a)
        # x_2_aa = self.relu2(x_2_a)
        # x_2_ab = self.relu2(-x_2_a)


        # x_2_b = self.l2_b(x_1_b)
        # x_2_ba = self.relu2(x_2_b)
        # x_2_bb = self.relu2(-x_2_b)

        # 4*n_hidden_2
        # x2 = torch.cat([x_2_aa, x_2_ab, x_2_ba, x_2_bb], dim=1)


        # x3 = self.l3(x2)
        # x3 = self.relu3(x3)

        # x = self.l4(x3)
        ivector_dim_sqrt = torch.tensor(x.shape[1]).float().sqrt()
        x = x_2_a
        x = F.normalize(x, p=2, dim=1)*ivector_dim_sqrt
        return x


    def hookl2(self, x):
        x_1 = self.l1(x)
        x_1_a = self.relu1(x_1)
        # x_1_b = self.relu1(-x_1)
    
        x_2_a = self.l2_a(x_1_a)

        # x_2_b = self.l2_b(x_1_b)

        # 2*n_hidden_2
        ivector_dim_sqrt = torch.tensor(x.shape[1]).float().sqrt()
        x = x_2_a
        x = F.normalize(x, p=2, dim=1)*ivector_dim_sqrt
        return x

# ==============================================================================
# add self attention to give weight
class Att_Diff_DNN(nn.Module):
    def __init__(self, ivector_dim, phone_num, n_hidden_1, n_hidden_2):
        super(Att_Diff_DNN, self).__init__()
        # only use linear transform to get the final combined output


        # attention extract network
        self.al1 = nn.Linear(ivector_dim*phone_num, n_hidden_1, bias=True)
        # self.tanh1 = nn.Tanh()
        self.arelu1 = nn.PReLU()
        self.al1_bn = nn.BatchNorm1d(n_hidden_1)

        self.al2 = nn.Linear(n_hidden_1, n_hidden_1, bias=True)
        # self.tanh1 = nn.Tanh()
        self.arelu2 = nn.PReLU()
        self.al2_bn = nn.BatchNorm1d(n_hidden_1)


        self.al21 = nn.Linear(n_hidden_1, n_hidden_1, bias=True)
        # self.tanh1 = nn.Tanh()
        self.arelu21 = nn.PReLU()
        self.al21_bn = nn.BatchNorm1d(n_hidden_1)

        self.al3 = nn.Linear(n_hidden_1, phone_num)
        self.asoft = nn.Softmax(dim=1)
        
        # self.dropout = nn.Dropout(p=0.2)
        self.ivector_dim = ivector_dim
        self.phone_num = phone_num


    def forward(self, x):
        ivector_dim_sqrt = torch.tensor(x.shape[1]).float().sqrt()
        # x = F.normalize(x, p=2, dim=1)*ivector_dim_sqrt
        # get the attention weight
        wx = self.arelu1(self.al1(x.view(-1,self.ivector_dim*self.phone_num)))
        # wx = self.al1_bn(wx)
        
        wx = self.arelu2(self.al2(wx))
        # wx = self.al2_bn(wx)
        # wx = self.arelu21(self.al21(wx))

        wx = self.al3(wx)
        att_weight = self.asoft(wx)

        # x is the batch*ivector array
        x = torch.bmm(att_weight.unsqueeze(1), x).squeeze(1)

        
        x = F.normalize(x, p=2, dim=1)*ivector_dim_sqrt
        return x


    def hookl2(self, x):
        ivector_dim_sqrt = torch.tensor(x.shape[1]).float().sqrt()
        # x = F.normalize(x, p=2, dim=1)*ivector_dim_sqrt
        # get the attention weight

        # wx = self.arelu1(self.al1(x.view(-1,self.ivector_dim*self.phone_num)))
        # # wx = self.al1_bn(wx)
        
        # wx = self.arelu2(self.al2(wx))
        # # wx = self.al2_bn(wx)

        # wx = self.al3(wx)
        # att_weight = self.asoft(wx)

        # # x is the batch*ivector array
        # x = torch.bmm(att_weight.unsqueeze(1), x).squeeze(1)
        # print(x.shape);
        x = torch.mean(x,1).squeeze(1);
        x = F.normalize(x, p=2, dim=1)*ivector_dim_sqrt
        return x

# ==============================================================================
# add self attention to give weight
class Att_Diff_DNN2(nn.Module):
    def __init__(self, ivector_dim, phone_num, n_hidden_1, n_hidden_2):
        super(Att_Diff_DNN2, self).__init__()
        # only use linear transform to get the final combined output
        self.al1 = nn.Linear(ivector_dim*phone_num, n_hidden_1, bias=True)
        # self.tanh1 = nn.Tanh()
        self.arelu1 = nn.PReLU()
        self.al3 = nn.Linear(n_hidden_1, phone_num)
        self.asoft = nn.Softmax(dim=1)

        self.bl = nn.Bilinear(ivector_dim,ivector_dim,phone_num)
        
        
        # self.dropout = nn.Dropout(p=0.2)
        self.ivector_dim = ivector_dim
        self.phone_num = phone_num


    def forward(self, x):

        ivector_dim_sqrt = torch.tensor(x.shape[1]).float().sqrt()
        # x = F.normalize(x, p=2, dim=1)*ivector_dim_sqrt
        # get the attention weight
        # wx = self.arelu1(self.al1(x.view(-1,self.ivector_dim*self.phone_num)))
        # wx = self.al3(wx)
        # att_weight = self.asoft(wx)


        B = self.bl.weight
        x2 = torch.zeros(x.size(), device = x.device)

        # for each batch
        for i in range(x.shape[0]):
            x2[i] = torch.matmul(x[i].unsqueeze(1),B).squeeze(1)

        C = torch.inverse(B.sum(0) - (self.phone_num-1)*torch.eye(self.ivector_dim, device = x.device))
        x3 = torch.matmul(x2,C)

        x3 = x3.mean(1)
        x3 = x3.squeeze(1)  
        # x3 = torch.bmm(att_weight.unsqueeze(1), x3).squeeze(1)

        x3 = F.normalize(x3, p=2, dim=1)*ivector_dim_sqrt
        return x3


    def hookl2(self, x):

        ivector_dim_sqrt = torch.tensor(x.shape[1]).float().sqrt()
        # x = F.normalize(x, p=2, dim=1)*ivector_dim_sqrt
        # get the attention weight
        # wx = self.arelu1(self.al1(x.view(-1,self.ivector_dim*self.phone_num)))
        # wx = self.al3(wx)
        # att_weight = self.asoft(wx)


        B = self.bl.weight
        x2 = torch.zeros(x.size(), device = x.device)

        # for each batch
        for i in range(x.shape[0]):
            x2[i] = torch.matmul(x[i].unsqueeze(1),B).squeeze(1)

        C = torch.inverse(B.sum(0) - (self.phone_num-1)*torch.eye(self.ivector_dim, device = x.device))
        x3 = torch.matmul(x2,C)
        
        x3 = x3.mean(1)
        x3 = x3.squeeze(1)  
        # x3 = torch.bmm(att_weight.unsqueeze(1), x3).squeeze(1)

        x3 = F.normalize(x3, p=2, dim=1)*ivector_dim_sqrt
        return x3
