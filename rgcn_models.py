import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import pdb
import time

from module.gcn import GCN, GraphConvolution
from module.roi_graph import get_st_graph


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

        self.st_gc1 = GraphConvolution(in_channel, in_channel, bias=False, batch=True)
        self.st_gc2 = GraphConvolution(in_channel, in_channel, bias=False, batch=True)
        self.st_gc3 = GraphConvolution(in_channel, self.out_channel, bias=False, batch=True)
        if self.separate_fb:
            self.st_gc1_back = GraphConvolution(in_channel, in_channel, bias=False, batch=True)
            self.st_gc2_back = GraphConvolution(in_channel, in_channel, bias=False, batch=True)
            self.st_gc3_back = GraphConvolution(in_channel, self.out_channel, bias=False, batch=True)

        self.sim_gc1 = GraphConvolution(in_channel, in_channel, bias=False, batch=True)
        self.sim_gc2 = GraphConvolution(in_channel, in_channel, bias=False, batch=True)
        self.sim_gc3 = GraphConvolution(in_channel, self.out_channel, bias=False, batch=True)

        self.dropout = nn.Dropout(dropout)
        self.init_weight()


    def st_GCN(self, input, front_graph, back_graph=None):
        input = input.squeeze(2)
        out = F.relu(self.st_gc1(input,front_graph))
        if self.separate_fb:
            out += F.relu(self.st_gc1_back(input,back_graph))
#        out = self.dropout(out)

        out2 = F.relu(self.st_gc2(out,front_graph))
        if self.separate_fb:
            out2 += F.relu(self.st_gc2_back(out,back_graph))
        out = out2
#        out = self.dropout(out2)

        out2 = F.relu(self.st_gc3(out,front_graph))
        if self.separate_fb:
            out2 += F.relu(self.st_gc3_back(out,back_graph))
        return out2


    def sim_GCN(self, input, adj):
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



    def forward(self, rois_features, rois):
        front_graph, back_graph = get_st_graph(rois)

        front_graph = front_graph.to(rois.device).detach()
        back_graph = back_graph.to(rois.device).detach()

        B, T, N, C = rois_features.size()
        N_rois = T*N
        rois_features = rois_features.view(B, N_rois, -1)
        sim_graph = self.sim_graph(rois_features).detach()
        sim_gcn = self.sim_GCN(rois_features, sim_graph)
        st_gcn = self.st_GCN(rois_features, front_graph, back_graph)
        gcn_out = sim_gcn + st_gcn
        gcn_out = gcn_out.mean(1)
        gcn_out = self.dropout(gcn_out)
        return gcn_out



    def sim_graph(self, features):
        sim1 = self.sim_embed1(features)
        sim2 = self.sim_embed2(features)
        sim_features = torch.matmul(sim1, sim2.transpose(1,2)) # d x d mat.
        sim_graph = F.softmax(sim_features, dim=-1)
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
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
        ]


if __name__ == '__main__':
    rois = torch.rand((4,8,10,4))
    rois_features = torch.rand((4,8,10,512))
    rgcn = RGCN()
    out = rgcn(rois_features, rois)

    pdb.set_trace()



