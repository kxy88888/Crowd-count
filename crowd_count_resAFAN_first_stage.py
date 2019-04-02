import torch
import torch.nn as nn
import torch.nn.functional as F

import src.network as network
from src.flow_attention_models_resAFAN_first_stage import resFAN_first_stage

class CrowdCounter1(nn.Module):
    def __init__(self):
        super(CrowdCounter1, self).__init__()
        self.xfnet = resFAN_first_stage()
    @property
    def loss(self):
        return 1
    
    def forward(self,  im_data, gt_data=None, gt_mask=None):
        #im_data = network.np_to_variable(im_data, is_cuda=True, is_training=self.training)
        x, f= self.xfnet(im_data)
        #density_cls_prob = F.softmax(density_cls_score
        return x ,f


