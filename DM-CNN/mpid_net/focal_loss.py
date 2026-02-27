import torchvision
import torch, sys, os, pdb
import torch.nn as nn
import torch.nn.functional as F

# https://programtalk.com/vs4/python/socom20/facebook-image-similarity-challenge-2021/ensemble_training_scripts/smp_test19/Facebook_model_v20.py/
# https://catalyst-team.github.io/catalyst/v20.10/_modules/catalyst/metrics/focal.html

def sigmoid_focal_loss(inputs, targets, alpha=-1, gamma=0, reduction="mean"):
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=-1, gamma=0):
        super(FocalLoss, self).__init__()
        self.alpha = -1
        self.gamma = gamma
        return None

    def forward(self, inputs, targets, reduction='mean'):
        focal_loss = sigmoid_focal_loss(
            inputs,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=reduction,
        )
        
        return focal_loss
