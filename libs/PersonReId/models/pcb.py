import torch
import torch.nn as nn
from torchvision import models

from .blocks import ClassBlock


# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):

    def __init__(self, class_num ):
        super(PCB, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.part = 6 # cut the pool5 to 6 parts
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(2048, class_num, droprate=0.5, linear=256, relu=False, bnorm=True))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
    
        part = {}
        predict = {}
        # get 6-part features with shape (B, 2048, 6)
        for i in range(self.part):
            part[i] = x[:,:,i].view(x.size(0), x.size(1))
            name = 'classifier' + str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])

        # prediction
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y


class PCB_test(nn.Module):

    def __init__(self,model):
        super(PCB_test, self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0),x.size(1),x.size(2))
        return y
