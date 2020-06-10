from collections import OrderedDict
import pdb
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign

from rgcn_models import RGCN
from roi_graph import extract_rois

# (output_size, spatial_scale, sampling_ratio, aligned=False)


class STRG(nn.Module):
    def __init__(self, base_model, in_channel=2048, out_channel=512,
                 nclass=174, dropout=0.3):
        super(STRG,self).__init__()
        self.base_model = base_model
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.nclass = nclass

        self.base_model.fc = nn.Identity()
        self.base_model.avgpool = nn.Identity()
        self.base_model.maxpool.stride = (1,2,2)
        self.base_model.layer3[0].conv2.stride=(1,2,2)
        self.base_model.layer3[0].downsample[0].stride=(1,2,2)
        self.base_model.layer4[0].conv2.stride=(1,1,1)
        self.base_model.layer4[0].downsample[0].stride=(1,1,1)

        self.reducer = nn.Conv3d(self.in_channel, self.out_channel,1)
        self.classifier = nn.Linear(2*self.in_channel, nclass)
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Dropout(p=dropout)
        )
        self.node_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(p=dropout)
        )

        self.strg_gcn = nn.Identity()

        self.rgcn = RGCN()
        self.roi_align = RoIAlign((7,7), 1/8, -1, aligned=True)

    def extract_feature(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        if not self.base_model.no_max_pool:
            x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        return x



    def forward(self, inputs, rois=None):
        features = self.extract_feature(inputs)
        features = self.reducer(features) # N C T H W
        pooled_features = self.avg_pool(features).squeeze(-1).squeeze(-1).squeeze(-1)
        N, C, T, H, W = features.shape


        gcn_features = features.transpose(1,2).contiguous().view(N*T,C,H,W)
        rois = rois.view(-1, 10, 4)
        rois = [r for r in rois]
        rois_features = self.roi_align(gcn_features, rois).view(N,T,C,H,W)
        rois_info = extract_rois(rois_features, rois)


        gcn_features = self.strg_gcn(features)
        gcn_features = self.node_pool(gcn_features).squeeze(-1)

        features = torch.cat((pooled_features, gcn_features), dim=-1)
        outputs = self.classifier(features)

        return outputs
