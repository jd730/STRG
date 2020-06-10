import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import pdb
import time

from module.gcn import GCN, GraphConvolution
from roi_graph import get_st_graph


class RGCN(torch.nn.Module):
    def __init__(self, in_channel=512, out_channel=512, test_mode=False,
                 dropout=0.5,
                 separate_fb=True):
        super(RGCN, self).__init__()

        # 1 by 1 conv -> 512  wang: 2048 -> 512
        self.out_channel = out_channel
        in_channel = in_channel # 512
        dropout = dropout
        self.separate_fb = separate_fb


        # wang2018video differentiates forward graph and backward graph,
        # but in this implementation we ignore this.

        self.sim_embed1 = nn.Linear(in_channel, in_channel, bias=False)
        self.sim_embed2 = nn.Linear(in_channel, in_channel, bias=False)

        self.st_gc1 = GraphConvolution(in_channel, in_channel, bias=False)
        self.st_gc2 = GraphConvolution(in_channel, in_channel, bias=False)
        self.st_gc3 = GraphConvolution(in_channel, self.out_channel, bias=False)
        if self.separate_fb:
            self.st_gc1_back = GraphConvolution(in_channel, in_channel, bias=False)
            self.st_gc2_back = GraphConvolution(in_channel, in_channel, bias=False)
            self.st_gc3_back = GraphConvolution(in_channel, self.out_channel, bias=False)

        self.sim_gc1 = GraphConvolution(in_channel, in_channel, bias=False)
        self.sim_gc2 = GraphConvolution(in_channel, in_channel, bias=False)
        self.sim_gc3 = GraphConvolution(in_channel, self.out_channel, bias=False)

#        self.dropout = nn.Dropout(dropout)
        self.init_weight()

    def st_GCN(self, input, adj):
        input = input.squeeze(2)
        out = F.relu(self.st_gc1(input,adj))
        if self.separate_fb:
            adj_back = adj.transpose(1,0)
            out += F.relu(self.st_gc1_back(input,adj_back))
#        out = self.dropout(out)

        out2 = F.relu(self.st_gc2(out,adj))
        if self.separate_fb:
            out2 += F.relu(self.st_gc2_back(out,adj_back))
        out = out2
#        out = self.dropout(out2)

        out2 = F.relu(self.st_gc3(out,adj))
        if self.separate_fb:
            out2 += F.relu(self.st_gc3_back(out,adj_back))
        return out2


    def sim_GCN(self, input, adj):
        input = input.squeeze(2)
        out = F.relu(self.sim_gc1(input,adj))
#        out = self.dropout(out)
        out = F.relu(self.sim_gc2(out,adj))
#        out = self.dropout(out)
        out = F.relu(self.sim_gc3(out,adj))
        return out



    def init_weight(self):
#        nn.init.constant_(self.sim_gc1.bias.data, 0)
#        nn.init.constant_(self.sim_gc2.bias.data, 0)
#        nn.init.constant_(self.sim_gc3.bias.data, 0)
#
#        nn.init.constant_(self.st_gc1.bias.data, 0)
#        nn.init.constant_(self.st_gc2.bias.data, 0)
#        nn.init.constant_(self.st_gc3.bias.data, 0)

        nn.init.normal_(self.sim_gc1.weight.data, 0, 0.001)
        nn.init.normal_(self.sim_gc2.weight.data, 0, 0.001)
        nn.init.normal_(self.st_gc1.weight.data, 0, 0.001)
        nn.init.normal_(self.st_gc2.weight.data, 0, 0.001)

        nn.init.normal_(self.st_gc3.weight.data, 0, 0.001)
        nn.init.normal_(self.sim_gc3.weight.data, 0, 0.001)
#        nn.init.constant_(self.sim_gc3.weight.data, 0)
#        nn.init.constant_(self.st_gc3.weight.data, 0)

        if self.separate_fb:
            nn.init.normal_(self.st_gc1_back.weight.data, 0, 0.001)
            nn.init.normal_(self.st_gc2_back.weight.data, 0, 0.001)
            nn.init.constant_(self.st_gc3_back.weight.data, 0)




    def generate_st_graphs(self, rois, connection, return_dict, st=0):
        for i, (r, c) in enumerate(zip(rois, connection)):
            return_dict[i+st] = get_st_graph(r,c)



    def forward(self, rois_info):
        if rois_info is None:
            raise Exception("WRONG")

        rois_features, rois, rois_st_graph = rois_info
        N = len(rois)


#        N_rois = rois_features.shape[2]
#        rois_features = rois_features.view(-1,N_rois, rois_features.shape[3])
        N_rois = rois_features.shape[-2]
        if N_rois == 0 :
            print(rois_features.shape)
        rois_features = rois_features.view(-1,N_rois, rois_features.shape[-1])
        rois = rois.view(-1,N_rois, 5)
        rois_st_graph = rois_st_graph.view(len(rois), N_rois, N_rois)

        ret_val = []
        count = 0

        for i, (f,r,s) in enumerate(zip(rois_features, rois, rois_st_graph)):
            # why should we set this?
            idx = torch.nonzero(r[:,0] >= 0).view(-1)
            if len(idx) == 0:
#                if self.fusion_type == 'mult':
#                    ret_val.append(torch.ones((1,1024)).cuda())
#                else:
                ret_val.append(torch.zeros((1,512)).cuda())
                count +=1
                continue

            r = r[idx]
            f = f[idx]
            st_graph = s[:len(idx),:len(idx)].detach()
            encoded_features = f.unsqueeze(-1)
            sim_graph = self.sim_graph(encoded_features.squeeze(-1))

            st_gcn = self.st_GCN(encoded_features, st_graph)
            sim_gcn = self.sim_GCN(encoded_features, sim_graph)

            gcn_out = st_gcn + sim_gcn
#            if not self.two_stream:
#                gcn_out = gcn_out.mean(dim=0, keepdim=True)
            ret_val.append(gcn_out)

        ret_val = torch.stack(ret_val, 0)
        ret_val = ret_val.view(N, -1, ret_val.shape[-1])
#        print("ZERO", count)
        return ret_val


    def sim_graph(self, features):
        sim1 = self.sim_embed1(features)
        sim2 = self.sim_embed2(features)
        sim_features = torch.matmul(sim1, sim2.transpose(1,0)) # d x d mat.
        sim_graph = F.softmax(sim_features, dim=1)
        return sim_graph

    def get_optim_policies(self):

        normal_weight = []
        normal_bias = []

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif isinstance(m, GraphConvolution):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif 'Conv' in str(type(m)):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    pdb.set_trace()
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
        ]



