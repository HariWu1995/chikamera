import torch
import torch.nn as nn

from timm.utils.model import freeze_batch_norm_2d


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            m.momentum = momentum


def freeze_bn(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            freeze_bn(module)
        if isinstance(module, nn.BatchNorm2d):
            setattr(model, n, freeze_batch_norm_2d(module))


class UpsamplingConcat(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)

