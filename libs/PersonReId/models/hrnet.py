import timm
import torch
import torch.nn as nn

from .blocks import ClassBlock


# Define the HRNet18-based Model
class ft_net_hr(nn.Module):

    def __init__(self, class_num, droprate=0.5, circle=False, linear_num=512):
        super().__init__()
        model_ft = timm.create_model('hrnet_w18', pretrained=True)
        # avg pooling to global pooling
        # model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.classifier = nn.Sequential() # save memory
        self.model = model_ft
        self.circle = circle
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, return_f=circle)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return 


if __name__ == '__main__':
    from torch.autograd import Variable

    net = ft_net_hr(751)
    net.classifier = nn.Sequential()
    print(net)

    input = Variable(torch.FloatTensor(8, 3, 224, 224))
    output = net(input)
    print('net output size:')
    print(output.shape)


