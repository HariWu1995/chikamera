import torch
import torch.nn as nn
from torchvision import models

from efficientnet_pytorch import EfficientNet

from .base import freeze_bn, UpsamplingConcat


class Encoder_swin_t(nn.Module):

    def __init__(self, C):
        super().__init__()
        self.C = C

        swin_t = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        freeze_bn(swin_t)

        self.layer0 = swin_t.features[0]
        self.layer1 = swin_t.features[1:3]
        self.layer2 = swin_t.features[3:5]
        # self.layer3 = swin_t.features[5:7]

        self.upsampling_layer1 = UpsamplingConcat(384 + 192, 384)
        self.upsampling_layer2 = UpsamplingConcat(384 + 96, 384)
        self.depth_layer = nn.Conv2d(384, self.C, kernel_size=1, bias=False)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        # x3 = self.layer3(x2)

        x = self.upsampling_layer1(x2.permute(0, 3, 1, 2), x1.permute(0, 3, 1, 2))
        x = self.upsampling_layer2(x, x0.permute(0, 3, 1, 2))
        x = self.depth_layer(x)

        return x


class Encoder_res101(nn.Module):

    def __init__(self, C):
        super().__init__()
        self.C = C

        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        freeze_bn(resnet)

        self.layer0 = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.upsampling_layer1 = UpsamplingConcat(1024 + 512, 512)
        self.upsampling_layer2 = UpsamplingConcat(512 + 256, 512)
        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, bias=False)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.upsampling_layer1(x3, x2)
        x = self.upsampling_layer2(x, x1)
        x = self.depth_layer(x)

        return x


class Encoder_res50(nn.Module):

    def __init__(self, C):
        super().__init__()
        self.C = C

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        freeze_bn(resnet)

        self.layer0 = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.upsampling_layer1 = UpsamplingConcat(1024 + 512, 512)
        self.upsampling_layer2 = UpsamplingConcat(512 + 256,  512)
        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, bias=False)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.upsampling_layer1(x3, x2)
        x = self.upsampling_layer2(x, x1)
        x = self.depth_layer(x)

        return x


class Encoder_res34(nn.Module):

    def __init__(self, C):
        super().__init__()
        self.C = C

        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        freeze_bn(resnet)

        self.layer0 = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.upsampling_layer1 = UpsamplingConcat(256 + 128, 256)
        self.upsampling_layer2 = UpsamplingConcat(256 + 64, 256)
        self.depth_layer = nn.Conv2d(256, self.C, kernel_size=1, bias=False)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.upsampling_layer1(x3, x2)
        x = self.upsampling_layer2(x, x1)
        x = self.depth_layer(x)

        return x


class Encoder_res18(nn.Module):

    def __init__(self, C):
        super().__init__()
        self.C = C

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        freeze_bn(resnet)

        self.layer0 = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.upsampling_layer1 = UpsamplingConcat(256 + 128, 256)
        self.upsampling_layer2 = UpsamplingConcat(256 + 64, 256)
        self.depth_layer = nn.Conv2d(256, self.C, kernel_size=1, bias=False)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.upsampling_layer1(x3, x2)
        x = self.upsampling_layer2(x, x1)
        x = self.depth_layer(x)

        return x


class Encoder_eff(nn.Module):

    def __init__(self, C, version='b4'):
        super().__init__()
        self.C = C
        self.downsample = 8

        self.version = version
        assert self.version in ['b0','b4'], \
            f'EfficientNet-{self.version} is not supported! Please choose `b0` or `b4`'
        self.backbone = EfficientNet.from_pretrained(f'efficientnet-{self.version}')
        self.delete_unused_layers()

        if self.downsample == 16:
            if self.version == 'b0':
                upsampling_in_channels = 320 + 112
            elif self.version == 'b4':
                upsampling_in_channels = 448 + 160
            upsampling_out_channels = 512

        elif self.downsample == 8:
            if self.version == 'b0':
                upsampling_in_channels = 112 + 40
            elif self.version == 'b4':
                upsampling_in_channels = 160 + 56
            upsampling_out_channels = 128

        else:
            raise ValueError(f'Downsample factor {self.downsample} not handled.')

        self.upsampling_layer = UpsamplingConcat(upsampling_in_channels, upsampling_out_channels)
        self.depth_layer = nn.Conv2d(upsampling_out_channels, self.C, kernel_size=1, padding=0)

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features(self, x):
        # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)

            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        if self.downsample == 16:
            input_1 = endpoints['reduction_5']
            input_2 = endpoints['reduction_4']

        elif self.downsample == 8:
            input_1 = endpoints['reduction_4']
            input_2 = endpoints['reduction_3']

        x = self.upsampling_layer(input_1, input_2)
        return x

    def forward(self, x):
        x = self.get_features(x)    # get feature vector
        x = self.depth_layer(x)     # feature and depth head
        return x

