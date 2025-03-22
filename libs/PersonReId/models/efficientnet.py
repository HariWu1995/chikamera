import torch
import torch.nn as nn

from .blocks import ClassBlock


# Define the Efficient-b4-based Model
class ft_net_efficient(nn.Module):

    def __init__(self, class_num, droprate=0.5, circle=False, linear_num=512):
        super().__init__()
        try:
            from efficientnet_pytorch import EfficientNet
        except ImportError:
            print('Please install efficientnet_pytorch')
        model_ft = EfficientNet.from_pretrained('efficientnet-b4')
        # model_ft = timm.create_model('tf_efficientnet_b4', pretrained=True)
        # avg pooling to global pooling
        # model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.head = nn.Sequential() # save memory
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.classifier = nn.Sequential()
        self.model = model_ft
        self.circle = circle
        # For EfficientNet, the feature dim is not fixed
        #   for efficientnet_b2: 1408
        #   for efficientnet_b4: 1792
        self.classifier = ClassBlock(1792, class_num, droprate, linear=linear_num, return_f=circle)

    def forward(self, x):
        #x = self.model.forward_features(x)
        x = self.model.extract_features(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

