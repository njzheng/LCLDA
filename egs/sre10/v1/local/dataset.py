import kaldi_io
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as f
import numpy as np
import torch
# import os
import scipy.io as sio
import csv

class SubivectorDataset(Dataset):
    def __init__(self, data_path):
        # for key,mat in kaldi_io.read_mat_scp(file):
        data_dic = { k:m for k,m in kaldi_io.read_mat_scp(data_path) } 
        # data_dic = {}
        # # avoid to have same key
        # indx = 0
        # for k,m in kaldi_io.read_mat_scp(data_path):
        #     data_dic[k+'-'+str(indx)]=m
        #     indx = indx + 1

        self.utt_list = list(data_dic.keys())
        self.data_list = list(data_dic.values())

    def __len__(self):
        return len(self.utt_list)

    def __getitem__(self, idx):
        utt = self.utt_list[idx]
        subivector = self.data_list[idx]
        """Convert ndarrays to Tensors."""
        return {'utt': utt,
                'subivector': torch.from_numpy(subivector).float()
                }


# ==================================================================================

class SubIvector_Batch_Hard_Triplet_Dataset(Dataset):
    def __init__(self, data_path, uttid_list, class_num):
        data_dic = { k:m for k,m in kaldi_io.read_mat_scp(data_path) } 
        self.utt_list = list(data_dic.keys())
        self.data_list = list(data_dic.values())

        self.uttid_list = uttid_list
        self.class_num = class_num

        assert len(uttid_list)==len(self.data_list), "The lengths of uttid_list and data unmatch!"

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x = self.data_list[idx]
        spk = self.uttid_list[idx]

        return x, spk 

# ==================================================================================

class IvectorDataset(Dataset):
    def __init__(self, data_path):
        data_dic = { k:m for k,m in kaldi_io.read_vec_flt_scp(data_path) } 
        self.utt_list = list(data_dic.keys())
        self.data_list = list(data_dic.values())

    def __len__(self):
        return len(self.utt_list)

    def __getitem__(self, idx):
        utt = self.utt_list[idx]
        ivector = self.data_list[idx]
        """Convert ndarrays to Tensors."""
        return {'utt': utt,
                'ivector': torch.from_numpy(ivector).float()
                }

class Ivector_Triplet_Dataset(Dataset):
    def __init__(self, data_path, uttid_list, class_num, k_times):
        data_dic = { k:m for k,m in kaldi_io.read_vec_flt_scp(data_path) } 
        self.utt_list = list(data_dic.keys())
        self.data_list = list(data_dic.values())

        self.uttid_list = uttid_list
        self.class_num = class_num
        self.k_times = k_times
        

        self.triplet_list = self.make_triplet_list(class_num,k_times)
        

    def __len__(self):
        return len(self.triplet_list)

    def __getitem__(self, idx):

        idx1, idx2, idx3 = self.triplet_list[idx]

        x1, x2, x3 = self.data_list[idx1], self.data_list[idx2], self.data_list[idx3]

        return x1, x2, x3


    def make_triplet_list(self, class_num, k_times=5):

        uttid_list = np.array(self.uttid_list)

        print('Processing Triplet Generation ...')

        triplet_list = []
        for class_idx in range(class_num):

            # If an ndarray, a random sample is generated from its elements. 
            within_class_num = len(np.where(uttid_list==class_idx)[0])
            # avoid only one sample
            ntriplet = 0
            if(within_class_num > 1):
                ntriplet = within_class_num*self.k_times
                within_class_set = np.where(uttid_list==class_idx)[0]
                a = np.random.choice(within_class_set, int(ntriplet), replace=True)
                b = np.random.choice(within_class_set, int(ntriplet), replace=True)
                while np.any((a-b)==0):
                    b = np.random.choice(within_class_set, int(ntriplet), replace=True)

                c = np.random.choice(np.where(uttid_list!=class_idx)[0], int(ntriplet), replace=True)

                for i in range(a.shape[0]):
                    triplet_list.append([int(a[i]), int(b[i]), int(c[i])])           

            # if(class_idx%500 == 1):
            #     print("class_indx: %d, ntriplet: %d" %(class_idx, ntriplet))


        # with open('/scratch/njzheng/kaldi/egs/sre10/v3/triplet_list.csv', "w") as f:
        #     writer = csv.writer(f, delimiter=' ')
        #     writer.writerows(triplet_list)
        # print('Done!')

        return triplet_list


    def renew(self):
        self.triplet_list = self.make_triplet_list(self.class_num,self.k_times)

# ==================================================================================

class Ivector_Batch_Hard_Triplet_Dataset(Dataset):
    def __init__(self, data_path, uttid_list, class_num):
        data_dic = { k:m for k,m in kaldi_io.read_vec_flt_scp(data_path) } 
        self.utt_list = list(data_dic.keys())
        self.data_list = list(data_dic.values())

        self.uttid_list = uttid_list
        self.class_num = class_num

        assert len(uttid_list)==len(self.data_list), "The lengths of uttid_list and data unmatch!"

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x = self.data_list[idx]
        spk = self.uttid_list[idx]

        return x, spk 

# ==================================================================================

class Uttid2int:
    #  construct the list to convert uttid into int as the label
    def __init__(self,list_path):
        f = open(list_path,'r')
        self.uttid_dic = {} #initialization
        for ln in f.readlines():
            temp_str = ln.split(' ',1)
            self.uttid_dic[temp_str[0]] = int(temp_str[1])

        f.close()

    def get_spkint(self, uttid):
        return self.uttid_dic[uttid]


#  write csvectors
def write_vectors(utt_list, csvector_list, data_path):
    size = len(utt_list)
    csvector_dict = {}
    for i in range(size):
        sub_list = utt_list[i]
        sub_vector = csvector_list[i]
        for j in range(len(sub_list)):
            csvector_dict[sub_list[j]] = sub_vector[j]

    with kaldi_io.open_or_fd(data_path,'wb') as f:
        for k,v in csvector_dict.items():
            kaldi_io.write_vec_flt(f, v, k)




class Input_transform:
    def __init__(self,train_dist_path, test_dist_path,dis_phone_num,prob_trans_path):
        f = open(train_dist_path,'r')

        # train_prob
        ln=f.readline()
        temp_prob_str = ln.split(' ')[3:-1]
        self.train_prob = torch.FloatTensor(list(map(float, temp_prob_str)))
        # var
        ln=f.readline()
        temp_var_str = ln.split(' ')[3:-1]
        self.train_var = torch.FloatTensor(list(map(float, temp_var_str)))
        f.close()

        # test_prob
        f = open(test_dist_path,'r')
        ln=f.readline()
        temp_prob_str = ln.split(' ')[3:-1]
        self.test_prob = torch.FloatTensor(list(map(float, temp_prob_str)))
        # var
        ln=f.readline()
        temp_var_str = ln.split(' ')[3:-1]
        self.test_var = torch.FloatTensor(list(map(float, temp_var_str)))
        f.close()

        # ratio
        length=len(self.test_prob)
        self.prob_ratio = torch.min(self.test_prob/self.train_prob, torch.ones(length))
        print('self.prob_ratio:')
        print(self.prob_ratio)

        assert max(self.prob_ratio) <=1 , 'The prob ratio is larger than 1!?'
        assert min(self.test_var) > 0, 'The min of test var is <= 0!?'
        self.std_var_ratio = torch.sqrt(self.test_var/self.train_var)
        

        print('self.std_var_ratio:')    
        print(self.std_var_ratio)
        print('self.test_var.sum():')    
        print(self.test_var.sum())

        # The first 4 are zero unvoiced parts
        self.phone_prob = torch.FloatTensor([0,0,0,0, 0.0202, 0.0522, 0.0917, 0.0153, 0.0088, 0.0483, 0.0130, 0.0048, 0.0290, 0.0212, 0.0249, 0.0177, 0.0240, 0.0146, 0.0093, 0.0194, 0.0490, 0.0457, 0.0050, 0.0296, 0.0367, 0.0407, 0.0530, 0.0114, 0.0416, 0.0011, 0.0124, 0.0302, 0.0457, 0.0073, 0.0571, 0.0064, 0.0047, 0.0249, 0.0123, 0.0191, 0.0287, 0.0230, 0.0002])
        self.prob_sum = self.phone_prob.sum()
        phone_num = len(self.phone_prob)

        assert dis_phone_num >=0 , 'The dis_phone_num need to be non negtive!'
        self.dis_phone_num = dis_phone_num 

        threshold_prob = self.prob_sum/((phone_num-dis_phone_num)*0.8)
        self.prob_ratio_upper = torch.max(self.phone_prob/threshold_prob, 0.2*torch.ones(phone_num))

        # load the GMM params
        temp_mat = sio.loadmat(prob_trans_path)
        self.mu_ratio = torch.from_numpy(temp_mat['mu_ratio']).float()
        # weight cumulation sum
        self.comp_wcum = torch.from_numpy(temp_mat['comp_wcum']).float()




    # transform input data 
    def transform_input(self,inputs, is_prob):
        batch_size = inputs.shape[0]
        phone_num = inputs.shape[1]
        ivector_dim = inputs.shape[2]

        dis_phone_num = self.dis_phone_num

        # to make the sum(||input||^2)=phone_num*ivector_dim
        tot_ratio = torch.sqrt(( (phone_num-dis_phone_num)*ivector_dim)/self.train_var[dis_phone_num-1:-1].sum())
        # tot_ratio = 1.0

        # respective ratio
        # resp_ratio = (ivector_dim)/self.train_var
        # tot_ratio = resp_ratio.sqrt()
        
        # discard some unvoiced part and store them in inputs_disphone
        inputs_disphone = torch.FloatTensor(batch_size,phone_num-dis_phone_num,ivector_dim)
        
        for i in range(batch_size):
            if(is_prob): 
                # prob_ratio_cut = torch.min(self.prob_ratio, 1.0*torch.ones(phone_num))
                # prob_exit = torch.rand(phone_num) < self.prob_ratio
                # prob_exit = torch.rand(phone_num) < prob_ratio_cut   

                prob_exit = torch.rand(phone_num) < self.prob_ratio_upper
                # add noise
                inputs[i,:,:] = torch.matmul(torch.diag(prob_exit.float()),inputs[i,:,:]) + torch.randn(phone_num,ivector_dim)*0.05

            inputs_disphone[i,:,:] = inputs[i,dis_phone_num-1:-1,:]*tot_ratio

        return inputs_disphone

    # transform input data with length normalization
    def transform_norm_input(self,inputs, is_prob):
        batch_size = inputs.shape[0]
        phone_num = inputs.shape[1]
        ivector_dim = inputs.shape[2]
        ivector_dim_sqrt = torch.tensor(ivector_dim).float().sqrt()

        dis_phone_num = self.dis_phone_num

        inputs_disphone = torch.FloatTensor(batch_size,phone_num-dis_phone_num,ivector_dim)

        for i in range(batch_size):
            inputs[i,:,:] = f.normalize(inputs[i,:,:], p=2, dim=1)*ivector_dim_sqrt
            if(is_prob):
                prob_exit = torch.rand(phone_num) < self.prob_ratio_upper

                inputs[i,:,:] = torch.matmul(torch.diag(prob_exit.float()),inputs[i,:,:]) + torch.randn(phone_num,ivector_dim)*0.01
            
            inputs_disphone[i,:,:] = inputs[i,dis_phone_num-1:-1,:]
        
        return inputs_disphone


    # transform input data with GMM transform
    def transform_GMM_input(self,inputs, is_prob):
        batch_size = inputs.shape[0]
        phone_num = inputs.shape[1]
        ivector_dim = inputs.shape[2]
        # ivector_dim_sqrt = torch.tensor(ivector_dim).float().sqrt()

        dis_phone_num = self.dis_phone_num

        inputs_disphone = torch.FloatTensor(batch_size,phone_num-dis_phone_num,ivector_dim)

        for i in range(batch_size):

            if(is_prob):
                temp = torch.rand(phone_num,1) < self.comp_wcum
                temp2 = torch.cat((torch.zeros(phone_num,1),temp.float()),1)
                comp_select = temp2[:,1:] - temp2[:,:-1]
                # The ratio for each phone vector
                mu_ratio_vec = torch.sum(self.mu_ratio*comp_select,1)

                inputs[i,:,:] = torch.matmul(torch.diag(mu_ratio_vec),inputs[i,:,:]) + torch.randn(phone_num,ivector_dim)*0.01
            
            inputs_disphone[i,:,:] = inputs[i,dis_phone_num-1:-1,:]
        
        return inputs_disphone