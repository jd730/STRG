import pdb
from collections import OrderedDict
import copy

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.image_list import ImageList

from torch.jit.annotations import Tuple, List, Dict, Optional
from torch import Tensor
import warnings
from transform import STRGTransform

class RPN(nn.Module):
    def __init__(self, pretrained=True, nrois=10):
        super(RPN,self).__init__()
        model = fasterrcnn_resnet50_fpn(pretrained=True).eval()
        self.transform = STRGTransform(model.transform.min_size,
                                       model.transform.max_size,
                                       0,0) #copy.deepcopy(model.transform)
        self.backbone = copy.deepcopy(model.backbone)
        self.rpn = copy.deepcopy(model.rpn)
#        self.eaget_outputs = copy.deepcopy(model.eaget_outputs)
        self.roi_heads = copy.deepcopy(model.roi_heads)
        self.rpn._pre_nms_top_n = {'training':3*nrois, 'testing':3*nrois}
        self.rpn._post_nms_top_n = {'training':nrois, 'testing':nrois}
        self.rpn.fg_bg_sampler.positive_fraction = 1.0
        del model

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
               like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        bs = len(images)
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenrate box
                    bb_idx = degenerate_boxes.any(dim=1).nonzero().view(-1)[0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invaid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        proposals = self.transform.rpn_postprocess(proposals, images.image_sizes, original_image_sizes)
        if False:
            for i in range(len(proposals)):
                delta = self.rpn._post_nms_top_n['testing'] - len(proposals[i])
                if delta != 0:
                    print("RPN finds only {} among {}".format(len(proposals[i]),
                                                        len(proposals[i])+delta))
                    dummy = -torch.ones((delta, 4)).to(proposals[i].device())
                    proposals[i] = torch.cat((proposals[i], dummy))
        return torch.cat(proposals).view(bs, -1, 4)


if __name__ == '__main__':
    rpn = RPN().eval()
#    rpn = nn.DataParallel(rpn, device_ids=None).cuda()
    inputs = torch.rand((5,3,224,224))
    out = rpn(inputs)
    pdb.set_trace()



