import torch
import torch.nn as nn
from torchvision import models

from .blocks import ClassBlock


# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, circle=False, linear_num=512):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        if stride == 1:
            model_ft.features.transition3.pool.stride = 1
        self.model = model_ft
        self.circle = circle
        # For DenseNet, the feature dim is 1024 
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f=circle)

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

