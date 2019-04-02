import torch
import torch.nn as nn
import torch.nn.functional as F

import src.network as network
from src.flow_attention_models_resAFAN_second_stage import resFAN

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
        return self.loss_mse + 0.003*self.loss_mask
    
    def forward(self,  in_data, gt_data=None, gt_mask=None):
        #im_data = network.np_to_variable(im_data, is_cuda=True, is_training=self.training)
        network.set_trainable(self.CCN, True)
        density_map, mask= self.CCN(in_data)
        #density_cls_prob = F.softmax(density_cls_score)
        
        if self.training:                        
            gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=self.training)            
            gt_mask = network.np_to_variable(gt_mask, is_cuda=True, is_training=self.training,dtype=torch.FloatTensor)
            self.loss_mse, self.loss_mask  = self.build_loss(density_map, mask, gt_data, gt_mask)
        return density_map ,mask
    
    def build_loss(self, density_map, mask, gt_data, gt_mask):
        loss_mse = self.loss_mse_fn(density_map, gt_data)        
        #ce_weights = torch.Tensor(ce_weights)
        #ce_weights = ce_weights.cuda()
        loss_mask=self.loss_mse_fn(mask,gt_mask)
        #loss_mse2=self.lose_mse_fn(density_map2,gt_data)
        #cross_entropy = self.loss_bce_fn(density_cls_score, gt_cls_label)
        return loss_mse, loss_mask #,#loss_mse2

