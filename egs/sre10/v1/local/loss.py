import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# class TripletMarginLoss(_Loss):
    
#     __constants__ = ['margin', 'p', 'eps', 'swap', 'reduction']

#     def __init__(self, margin=1.0, p=2., eps=1e-6, swap=False, size_average=None,
#                  reduce=None, reduction='mean'):
#         super(TripletMarginLoss, self).__init__(size_average, reduce, reduction)
#         self.margin = margin
#         self.p = p
#         self.eps = eps
#         self.swap = swap

#     @weak_script_method
#     def forward(self, anchor, positive, negative):
#         return F.triplet_margin_loss(anchor, positive, negative, margin=self.margin, p=self.p, eps=self.eps, swap=self.swap, reduction=self.reduction)

class Batch_hard_TripletMarginLoss(nn.Module):
    
    __constants__ = ['margin', 'p', 'eps']

    def __init__(self, margin=1.0, p=2.0, eps=1e-6, k=1):
        super(Batch_hard_TripletMarginLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.eps = eps
        self.k = k

    def forward(self, embedding, spk):
        # k is the topk para
        # k = self.k

        spk = torch.LongTensor(spk).view(-1, 1)
        spk = spk.cuda()

        dist_mat = pairwise_distances(embedding)

        spkt = torch.transpose(spk, 0, 1)

        post_mask = (spk == spkt)
        nega_mask = 1 - post_mask


        post_dist = torch.max(dist_mat*post_mask.float(), dim=1)[0]
        nega_dist = torch.min(dist_mat*nega_mask.float(), dim=1)[0]

        # post_dist = torch.sum(dist_mat*post_mask.float(), dim=1)
        # nega_dist = torch.sum(dist_mat*nega_mask.float(), dim=1)/(1+F.relu( torch.sum(nega_mask.float(), dim =1 ) -1 ))


        loss_a = F.relu(self.margin.cuda() + post_dist - nega_dist)

        return loss_a.mean()


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)