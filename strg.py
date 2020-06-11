from collections import OrderedDict
import pdb
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign

from rgcn_models import RGCN

from model import (generate_model, load_pretrained_model, make_data_parallel,
                   get_fine_tuning_parameters)

from models import resnet, resnet2p1d, pre_act_resnet, wide_resnet, resnext, densenet

class STRG(nn.Module):
    def __init__(self, base_model, in_channel=2048, out_channel=512,
                 nclass=174, dropout=0.3, nrois=10):
        super(STRG,self).__init__()
        self.base_model = base_model
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.nclass = nclass
        self.nrois = nrois

        self.base_model.fc = nn.Identity()
        self.base_model.avgpool = nn.Identity()
        self.base_model.maxpool.stride = (1,2,2)
        self.base_model.layer3[0].conv2.stride=(1,2,2)
        self.base_model.layer3[0].downsample[0].stride=(1,2,2)
        self.base_model.layer4[0].conv2.stride=(1,1,1)
        self.base_model.layer4[0].downsample[0].stride=(1,1,1)

        self.reducer = nn.Conv3d(self.in_channel, self.out_channel,1)
        self.classifier = nn.Linear(2*self.out_channel, nclass)
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Dropout(p=dropout)
        )
        self.max_pool = nn.AdaptiveAvgPool2d(1)

        self.strg_gcn = RGCN()
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

        rois_list = rois.view(-1, self.nrois, 4)
        rois_list = [r for r in rois_list]

        features = features.transpose(1,2).contiguous().view(N*T,C,H,W)
        rois_features = self.roi_align(features, rois_list)
        rois_features = self.max_pool(rois_features)
        rois_features = rois_features.view(N,T,self.nrois,C)
        gcn_features = self.strg_gcn(rois_features, rois)

        features = torch.cat((pooled_features, gcn_features), dim=-1)
        outputs = self.classifier(features)

        return outputs


if __name__ == '__main__':

    model = resnet.generate_model(model_depth=50,
                                    n_classes=174,
                                    n_input_channels=3,
                                    shortcut_type='B',
                                    conv1_t_size=7,
                                    conv1_t_stride=1,
                                    no_max_pool=False,
                                    widen_factor=1.0)

    rois = torch.rand((4,8,10,4))
    inputs = torch.rand((4,3,16,224,224))
    strg = STRG(model)
    out = strg(inputs, rois)

    pdb.set_trace()
    print(out.shape)



