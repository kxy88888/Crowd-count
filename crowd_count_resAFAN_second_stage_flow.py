import torch
import torch.nn as nn
import torch.nn.functional as F

import src.network as network
from src.flow_attention_models_resAFAN_second_stage_flow import resFAN

class CrowdCounter2(nn.Module):
    def __init__(self):
        super(CrowdCounter2, self).__init__()
        self.CCN = resFAN()
        #if ce_weights is not None:
            #ce_weights = torch.Tensor(ce_weights)
            #ce_weights = ce_weights.cuda()
        self.loss_mse_fn = nn.MSELoss()
        #self.loss_bce_fn = nn.BCELoss(weight=ce_weights)
        
    @property
    def loss(self):
        return self.loss_mask
    
    def forward(self,  in_data, gt_mask=None):
        #im_data = network.np_to_variable(im_data, is_cuda=True, is_training=self.training)
        network.set_trainable(self.CCN, True)
        mask_info,mask= self.CCN(in_data)
        #density_cls_prob = F.softmax(density_cls_score)
        
        if self.training:
            gt_mask = network.np_to_variable(gt_mask, is_cuda=True, is_training=self.training)
            self.loss_mask  = self.build_loss(mask,gt_mask)
        return mask_info,mask
    
    def build_loss(self,  mask, gt_mask):
        #ce_weights = torch.Tensor(ce_weights)
        #ce_weights = ce_weights.cuda()
        loss_mask=self.loss_mse_fn(mask,gt_mask)
        #loss_mse2=self.lose_mse_fn(density_map2,gt_data)
        #cross_entropy = self.loss_bce_fn(density_cls_score, gt_cls_label)
        return loss_mask #,#loss_mse2


class CrowdCounter2_flow(nn.Module):
    def __init__(self):
        super(CrowdCounter2_flow, self).__init__()
        self.CCN = resFAN()
        # if ce_weights is not None:
        # ce_weights = torch.Tensor(ce_weights)
        # ce_weights = ce_weights.cuda()

        # self.loss_bce_fn = nn.BCELoss(weight=ce_weights)

    def forward(self, in_data, gt_mask=None):
        # im_data = network.np_to_variable(im_data, is_cuda=True, is_training=self.training)
        mask_info, mask = self.CCN(in_data)
        # density_cls_prob = F.softmax(density_cls_score)
        return mask_info, mask
