import torch
from torch import nn, Tensor
import numpy as np
import torch.nn.functional as F

# TODO systematic naming system for class weights
class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, reduction: str = 'mean'):
        # Initialize the superclass (nn.CrossEntropyLoss) with the weight, ignore_index, and reduction parameters        
        # Define class weights
        # alpha = torch.tensor([1., 20.0, 20.], dtype=torch.float).cuda()
        # if weight is not None:
        #     if isinstance(weight, list):
        #         self.weight = torch.tensor(weight, device='cuda')
        #     else:
        #         self.weight = weight
        # else:
        #     self.weight = None

        print("RobustCrossEntropyLoss with class weights (weight) ", str(weight))
        super().__init__(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, alpha=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(alpha)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()

# Per Claude 3

class MulticlassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(MulticlassFocalLoss, self).__init__()
        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = torch.tensor(alpha)
            else:
                self.alpha = alpha
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        
        # Apply softmax to get class probabilities
        prob = F.softmax(inputs, dim=1)
        
        # Create a mask for the target class probabilities
        class_mask = torch.zeros_like(prob)
        targets_int64 = targets.to(torch.int64)
        class_mask.scatter_(1, targets_int64.unsqueeze(1), 1)
        
        # Compute the focal loss
        probs = (prob * class_mask).sum(1)
        if self.alpha is not None:
            if self.alpha.dtype != probs.dtype:
                self.alpha = self.alpha.to(probs.dtype)
            if self.alpha.shape[0] != num_classes:
                raise ValueError(f"The number of class weights ({self.alpha.shape[0]}) does not match the number of classes ({num_classes})")
            self.alpha = self.alpha.to(targets_int64.device)  # Move alpha to the same device as targets
            alpha = self.alpha[targets_int64]
        else:
            alpha = 1.
            
        focal_loss = -alpha * (1 - probs) ** self.gamma * torch.log(probs + 1e-12)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
# Per GPT4:
# class MulticlassFocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
#         super().__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction
#         if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)

#     def forward(self, inputs, targets):
#         # Ensure class probabilities using softmax
#         inputs = F.softmax(inputs, dim=1)
#         # Create the targets in one-hot encoded format
#         # target_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).float()
        
#         # Calculate log probabilities
#         logpt = torch.log(inputs)
#         # Gather the log probabilities by the target class
#         logpt = logpt.gather(1, targets.long().unsqueeze(1))
#         logpt = logpt.view(-1)
#         pt = logpt.exp()

#         if self.alpha is not None:
#             if self.alpha.type()!=inputs.data.type():
#                 self.alpha = self.alpha.type_as(inputs.data)
#             at = self.alpha.gather(0, targets.data.view(-1).long())
#             logpt = logpt * at

#         # Compute the focal loss
#         loss = -1 * (1 - pt) ** self.gamma * logpt

#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             return loss

# Focal loss variant of cross entropy loss, designed to emphasize hard to classify pixels.
# Alternate implementations:
# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
# https://gist.github.com/f1recracker/0f564fd48f15a58f4b92b3eb3879149b
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class FocalLoss(RobustCrossEntropyLoss):
#     ''' Focal loss for classification tasks on imbalanced datasets '''

#     def __init__(self, gamma=2., weight=None, ignore_index=-100, reduction='none'):
#         super().__init__(weight=weight, ignore_index=ignore_index, reduction='none')
#         self.reduction = reduction
#         self.gamma = gamma

#     def forward(self, input_, target):
#         cross_entropy = super().forward(input_, target)
#         # Temporarily mask out ignore index to '0' for valid gather-indices input.
#         # This won't contribute final loss as the cross_entropy contribution
#         # for these would be zero.
#         target = target * (target != self.ignore_index).long()
#         # input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
#         # Per chatgpt to fix a runtime error
#         input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1).long())
#         loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
#         match self.reduction:
#             case 'mean':
#                 return torch.mean(loss)
#             case 'sum':
#                 return torch.sum(loss)
#             case _:
#                 return loss
            
            # Other focal loss options
            # https://github.com/ailias/Focal-Loss-implement-on-TensorFlow/blob/master/focal_loss.py
# 

