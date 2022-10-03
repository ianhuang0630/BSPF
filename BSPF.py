import torch
import torch.nn as nn
import torch.nn.functional as F


class bspf_network(nn.Module):
    def __init__(self, embedding_size, num_planes):
        super().__init__()

        self.embedding_size = embedding_size
        self.num_planes = num_planes


        # the below is going to be inertible
        self.plane_feat_size = 4
        self.output_size = self.num_planes * 4
        self.mlp = nn.Linear(embedding_size * 2, self.output_size)

        # initializing both to be 0
        nn.init.constant_(self.mlp.weight, 0)
        nn.init.constant_(self.mlp.bias, 0)

    def forward(self, tgt_z, src_z, src_planes):
        plane_delta = self.mlp(torch.cat([tgt_z, src_z], dim=1)) # batchsize x 4 
        plane_delta = plane_delta.reshape(plane_delta.shape[0], -1, self.num_planes)
        return src_planes + plane_delta
        # needs to be output 
        # batchsize x num_planes x 4

