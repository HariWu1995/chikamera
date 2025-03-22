import timm
import torch
import torch.nn as nn

from .blocks import ClassBlock
from .utils import load_state_dict_mute


# Define the swin_base_patch4_window7_224 Model
class ft_net_swin(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, circle=False, linear_num=512):
        super(ft_net_swin, self).__init__()
        model_ft = timm.create_model('swin_base_patch4_window7_224', pretrained=True, drop_path_rate=0.2)
        model_ft.head = nn.Sequential() # save memory
        self.model = model_ft
        self.circle = circle
        self.avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.avgpool2d = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f=circle)
        # print('Make sure timm > 0.6.0'
        #       'You can install latest timm version by pip install git+https://github.com/rwightman/pytorch-image-models.git')

    def forward(self, x):
        x = self.model.forward_features(x)
        # swin is update in latest timm > 0.6.0, so I add the following 2 lines.
        if x.dim() == 3:
            x = self.avgpool1d(x.permute((0,2,1)))
        else: 
            x = self.avgpool2d(x.permute((0,3,1,2)))
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class ft_net_swinv2(nn.Module):

    def __init__(self, class_num, input_size=(256, 128), droprate=0.5, stride=2, circle=False, linear_num=512):
        super(ft_net_swinv2, self).__init__()
        # model_ft = timm.create_model('swinv2_cr_small_224', pretrained=True, img_size=input_size, drop_path_rate=0.2)
        model_ft = timm.create_model('swinv2_base_window8_256', pretrained=False, img_size=input_size, drop_path_rate=0.2)
        model_full = timm.create_model('swinv2_base_window8_256', pretrained=True)
        load_state_dict_mute(model_ft, model_full.state_dict(), strict=False)
        model_ft.head = nn.Sequential() # save memory
        self.model = model_ft
        self.circle = circle
        self.avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.avgpool2d = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f=circle)
        # print('Make sure timm > 0.6.0'
        #       'You can install latest timm version by pip install git+https://github.com/rwightman/pytorch-image-models.git')

    def forward(self, x):
        x = self.model.forward_features(x)
        if x.dim() == 3:
            x = self.avgpool1d(x.permute((0,2,1)))
        else:
            x = self.avgpool2d(x.permute((0,3,1,2)))
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

