import kaldi_io
import numpy as np
import torch
import os
import sys
import scipy.io as sio
import csv
import tsne


if sys.argv.__len__() != 6 :
    print("The usage: python3 local/LDA_tsne.py LCLDAset.ark LCPLDAset.ark spk_mean.ark t_sne.mat ./local/tsne/")
    print(sys.argv[:])
    exit()

LCLDA_filename = sys.argv[1]
LCPLDA_filename = sys.argv[2]

spkmean_filename = sys.argv[3]
tsne_filename = sys.argv[4] 
save_path = sys.argv[5] 

# read LCLDA mat into lda_vec
col_indx = 0
lclda_vec = {'Description':' This vec set store the columns of the LDA matrix from all speakers.'} 
lclda_vec2 = {'Description':' This vec set store the lastcolumns of the LDA matrix from all speakers.'} 

dim = 0
for k,m in kaldi_io.read_mat_ark(LCLDA_filename):
	dim = m.shape[1] #numcol
	vec = m[col_indx,:]
	lclda_vec[k] = vec
	lclda_vec2[k] = m[-1,:]


# read LCPDA mat into lda_vec
lcplda_vec = {'Description':' This vec set store the columns of the LDA matrix from all speakers.'} 
lcplda_vec2 = {'Description':' This vec set store the lastcolumns of the LDA matrix from all speakers.'} 

dim = 0
for k,m in kaldi_io.read_mat_ark(LCPLDA_filename):
	dim = m.shape[1] #numcol
	vec = m[col_indx,:]
	lcplda_vec[k] = vec
	lcplda_vec2[k] = m[-1,:]


# ignore discription item
print("lclda_vec size is ",len(lclda_vec)-1)
print("lcplda_vec size is ",len(lcplda_vec)-1)


# read speaker mean into spk_mean
spk_mean = {k:m for k,m in kaldi_io.read_vec_flt_ark(spkmean_filename)}
print("spk_mean size is ",len(spk_mean))


# delete the empty item in spk_mean
# key_list = spk_mean.keys()
# for k in key_list:
# 	if not k in lda_vec: 
# 		del spk_mean[k] 
# print("spk_mean size after deletion is ",len(spk_mean))

num_spk_ = len(spk_mean)
# convert dict to array
lclda_mat = np.empty([num_spk_, dim], dtype = float) 
lclda_mat2 = np.empty([num_spk_, dim], dtype = float) 
lcplda_mat = np.empty([num_spk_, dim], dtype = float) 
lcplda_mat2 = np.empty([num_spk_, dim], dtype = float) 

spk_mean_mat = np.empty([num_spk_, dim], dtype = float) 

spk_indx=0
for k in spk_mean:
	if k in lclda_vec:
		lclda_mat[spk_indx,:] = lclda_vec[k]
		lclda_mat2[spk_indx,:] = lclda_vec2[k]
		lcplda_mat[spk_indx,:] = lcplda_vec[k]
		lcplda_mat2[spk_indx,:] = lcplda_vec2[k]

		spk_mean_mat[spk_indx,:] = spk_mean[k]
		spk_indx = spk_indx+1



# spk_mean_tsne= tsne.tsne(spk_mean_mat, 2, 50, 20.0)
# lclda_mat_tsne= tsne.tsne(lclda_mat, 2, 50, 20.0)
# lclda_mat_tsne2= tsne.tsne(lclda_mat2, 2, 50, 20.0)
lcplda_mat_tsne= tsne.tsne(lcplda_mat, 2, 50, 20.0)
# lcplda_mat_tsne2= tsne.tsne(lcplda_mat2, 2, 50, 20.0)

# np.save(os.path.join(save_path,"spk_mean_tsne.npy"), spk_mean_tsne)
# np.save(os.path.join(save_path,"lclda_mat_tsne.npy"), lclda_mat_tsne)
# np.save(os.path.join(save_path,"lclda_mat_tsne2.npy"), lclda_mat_tsne2)
np.save(os.path.join(save_path,"lcplda_mat_tsne.npy"), lcplda_mat_tsne)
# np.save(os.path.join(save_path,"lcplda_mat_tsne2.npy"), lcplda_mat_tsne2)

# np.savetxt(os.path.join(save_path,"spk_mean_tsne.txt"), spk_mean_tsne, fmt="%d", delimiter=" ")
# np.savetxt(os.path.join(save_path,"lclda_mat_tsne.txt"), lclda_mat_tsne, fmt="%d", delimiter=" ")
# np.savetxt(os.path.join(save_path,"lclda_mat_tsne2.txt"), lclda_mat_tsne2, fmt="%d", delimiter=" ")
np.savetxt(os.path.join(save_path,"lcplda_mat_tsne.txt"), lcplda_mat_tsne, fmt="%d", delimiter=" ")
# np.savetxt(os.path.join(save_path,"lcplda_mat_tsne2.txt"), lcplda_mat_tsne2, fmt="%d", delimiter=" ")





# python3 local/LDA_tsne.py exp/ivectors_sre_dnn600/transform_lc.ark  exp/ivectors_sre_dnn600/transform_lcp.ark exp/ivectors_sre_dnn600/spk_mean_ln.ark t_sne.mat ./exp/tsne/