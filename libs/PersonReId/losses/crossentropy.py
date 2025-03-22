import torch
import torch.nn as nn


class CrossEntropyLabelSmooth(nn.Module):
    """
    Cross entropy loss with label smoothing regularizer.
    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    
    Args:
        epsilon (float): weight.
    """
    def __init__(self, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        _, num_classes = inputs.shape
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: 
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

