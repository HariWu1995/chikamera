import os

import torch
import torch.nn as nn
import copy

# from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from .backbones.resnet import ResNet, Bottleneck
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID
from .backbones.swin_transformer import swin_base_patch4_window7_224, swin_small_patch4_window7_224, swin_tiny_patch4_window7_224
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a
from torchreid.constants import GLOBAL, BACKGROUND, FOREGROUND, CONCAT_PARTS, PARTS, \
                            BN_GLOBAL, BN_BACKGROUND, BN_FOREGROUND, BN_CONCAT_PARTS, BN_PARTS


def shuffle_unit(features, shift, group, begin=1):

    batch_size = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batch_size, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batch_size, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, dim)

    return x

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


# class Backbone(nn.Module):
#     def __init__(self, num_classes, cfg):
#         super(Backbone, self).__init__()
#         last_stride = cfg.MODEL.LAST_STRIDE
#         model_path = cfg.MODEL.PRETRAIN_PATH
#         model_name = cfg.MODEL.NAME
#         pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
#         self.cos_layer = cfg.MODEL.COS_LAYER
#         self.neck = cfg.MODEL.NECK
#         self.neck_feat = cfg.TEST.NECK_FEAT
#         self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
#         self.feat_dim = cfg.MODEL.FEAT_DIM
#         self.dropout_rate = cfg.MODEL.DROPOUT_RATE
#
#         if model_name == 'resnet50':
#             self.in_planes = 2048
#             self.base = ResNet(last_stride=last_stride,
#                                block=Bottleneck,
#                                layers=[3, 4, 6, 3])
#             print('using resnet50 as a backbone')
#         elif model_name == 'resnet50_ibn_a':
#             self.in_planes = 2048
#             self.base = resnet50_ibn_a(last_stride)
#             print('using resnet50_ibn_a as a backbone')
#         else:
#             print('unsupported backbone! but got {}'.format(model_name))
#
#         if pretrain_choice == 'imagenet':
#             self.base.load_param(model_path)
#             print('Loading pretrained ImageNet model......from {}'.format(model_path))
#
#
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.num_classes = num_classes
#         if self.reduce_feat_dim:
#             self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
#             self.fcneck.apply(weights_init_xavier)
#             self.in_planes = cfg.MODEL.FEAT_DIM
#
#         self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
#         self.classifier.apply(weights_init_classifier)
#
#         self.bottleneck = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck.bias.requires_grad_(False)
#         self.bottleneck.apply(weights_init_kaiming)
#
#         if self.dropout_rate > 0:
#             self.dropout = nn.Dropout(self.dropout_rate)
#
#         if pretrain_choice == 'self':
#             self.load_param(model_path)
#
#
#     def forward(self, x, label=None, **kwargs):  # label is unused if self.cos_layer == 'no'
#         x = self.base(x)
#         global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
#         global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
#         if self.reduce_feat_dim:
#             global_feat = self.fcneck(global_feat)
#
#         if self.neck == 'no':
#             feat = global_feat
#         elif self.neck == 'bnneck':
#             feat = self.bottleneck(global_feat)
#         if self.dropout_rate > 0:
#             feat = self.dropout(feat)
#
#         if self.training:
#             if self.cos_layer:
#                 cls_score = self.arcface(feat, label)
#             else:
#                 cls_score = self.classifier(feat)
#             return cls_score, global_feat
#         else:
#             if self.neck_feat == 'after':
#                 return feat
#             else:
#                 return global_feat
#
#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         if 'state_dict' in param_dict:
#             param_dict = param_dict['state_dict']
#         for i in param_dict:
#             if 'classifier' in i:
#                 continue
#             elif 'module' in i:
#                 self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
#             else:
#                 self.state_dict()[i].copy_(param_dict[i])
#         print('Loading pretrained model from {}'.format(trained_path))
#
#     #  def load_param(self, trained_path):
#         #  param_dict = torch.load(trained_path, map_location = 'cpu')
#         #  for i in param_dict:
#             #  try:
#                 #  self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
#             #  except:
#                 #  continue
#         #  print('Loading pretrained model from {}'.format(trained_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, semantic_weight, model_filename):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        # model_path = cfg.MODEL.PRETRAIN_PATH
        model_path = os.path.join(cfg.MODEL.PRETRAIN_PATH, model_filename[cfg.MODEL.TRANSFORMER_TYPE])
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE
        self.feature_dim = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        convert_weights = True if pretrain_choice == 'imagenet' else False
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate= cfg.MODEL.DROP_OUT,attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, pretrained=model_path, convert_weights=convert_weights, semantic_weight=semantic_weight)
        if model_path != '':
            self.base.init_weights(model_path)
        self.in_planes = self.base.num_features[-1]

        self.num_classes = num_classes
        # self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        # if self.ID_LOSS_TYPE == 'arcface':
        #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
        #     self.classifier = Arcface(self.in_planes, self.num_classes,
        #                               s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        # elif self.ID_LOSS_TYPE == 'cosface':
        #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
        #     self.classifier = Cosface(self.in_planes, self.num_classes,
        #                               s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        # elif self.ID_LOSS_TYPE == 'amsoftmax':
        #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
        #     self.classifier = AMSoftmax(self.in_planes, self.num_classes,
        #                                 s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        # elif self.ID_LOSS_TYPE == 'circle':
        #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
        #     self.classifier = CircleLoss(self.in_planes, self.num_classes,
        #                                 s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        # else:
        if self.reduce_feat_dim:
            self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
            self.fcneck.apply(weights_init_xavier)
            self.in_planes = cfg.MODEL.FEAT_DIM
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.dropout = nn.Dropout(self.dropout_rate)

        #if pretrain_choice == 'self':
        #    self.load_param(model_path)
        self.spatial_feature_shape = [cfg.INPUT.SIZE_TRAIN[0]/16, cfg.INPUT.SIZE_TRAIN[1]/16, self.base.num_features[-1]]

    def forward(self, x, label=None, cam_label= None, view_label=None, **kwargs):
        global_feat, featmaps = self.base(x)
        if self.reduce_feat_dim:
            global_feat = self.fcneck(global_feat)
        feat = self.bottleneck(global_feat)
        feat_cls = self.dropout(feat)

        # if self.training:
        #     if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
        #         cls_score = self.classifier(feat_cls, label)
        #     else:
        #         cls_score = self.classifier(feat_cls)
        #
        #     return cls_score, global_feat, featmaps  # global feature for triplet loss
        # else:
        #     if self.neck_feat == 'after':
        #         # print("Test with feature after BN")
        #         return feat, featmaps
        #     else:
        #         # print("Test with feature before BN")
        #         return global_feat, featmaps

        if self.training:
            cls_score = self.classifier(feat)
            # Outputs
            embeddings = {
                GLOBAL: global_feat,  # [N, D]
                BACKGROUND: None,  # [N, D]
                FOREGROUND: None,  # [N, D]
                CONCAT_PARTS: None,  # [N, K*D]
                PARTS: None,  # [N, K, D]
                BN_GLOBAL: feat,  # [N, D]
                BN_BACKGROUND: None,  # [N, D]
                BN_FOREGROUND: None,  # [N, D]
                BN_CONCAT_PARTS: None,  # [N, K*D]
                BN_PARTS: None,  # [N, K, D]
            }

            visibility_scores = {
                GLOBAL: torch.ones(global_feat.shape[0], device=global_feat.device, dtype=torch.bool),  # [N]
                BACKGROUND: None,  # [N]
                FOREGROUND: None,  # [N]
                CONCAT_PARTS: None,  # [N]
                PARTS: None,  # [N, K]
            }

            id_cls_scores = {
                GLOBAL: cls_score,  # [N, num_classes]
                BACKGROUND: None,  # [N, num_classes]
                FOREGROUND: None,  # [N, num_classes]
                CONCAT_PARTS: None,  # [N, num_classes]
                PARTS: None,  # [N, K, num_classes]
            }

            masks = {
                GLOBAL: None,
                BACKGROUND: None,
                FOREGROUND: None,
                CONCAT_PARTS: None,
                PARTS: None,
            }

            pixels_cls_scores = None
            spatial_features = None

            return embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, masks
        else:
            # Outputs
            embeddings = {
                GLOBAL: global_feat,  # [N, D]
                BACKGROUND: None,  # [N, D]
                FOREGROUND: None,  # [N, D]
                CONCAT_PARTS: None,  # [N, K*D]
                PARTS: None,  # [N, K, D]
                BN_GLOBAL: feat,  # [N, D]
                BN_BACKGROUND: None,  # [N, D]
                BN_FOREGROUND: None,  # [N, D]
                BN_CONCAT_PARTS: None,  # [N, K*D]
                BN_PARTS: None,
                # [N, K, D]
            }

            visibility_scores = {
                GLOBAL: torch.ones(global_feat.shape[0], device=global_feat.device, dtype=torch.bool),  # [N]
                BACKGROUND: None,  # [N]
                FOREGROUND: None,  # [N]
                CONCAT_PARTS: None,  # [N]
                PARTS: None,  # [N, K]
            }

            id_cls_scores = {
                GLOBAL: None,  # [N, num_classes]
                BACKGROUND: None,  # [N, num_classes]
                FOREGROUND: None,  # [N, num_classes]
                CONCAT_PARTS: None,  # [N, num_classes]
                PARTS: None,  # [N, K, num_classes]
            }

            masks = {
                GLOBAL: torch.ones((global_feat.shape[0], 32, 16), device=global_feat.device),  # [N, Hf, Wf]
                BACKGROUND: None,  # [N, Hf, Wf]
                FOREGROUND: None,  # [N, Hf, Wf]
                CONCAT_PARTS: None,  # [N, Hf, Wf]
                PARTS: None,  # [N, K, Hf, Wf]
            }

            pixels_cls_scores = None
            spatial_features = None

            return embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, masks


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location = 'cpu')
        for i in param_dict:
            try:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            except:
                continue
        print('Loading pretrained model from {}'.format(trained_path))


# class build_transformer_local(nn.Module):
#     def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
#         super(build_transformer_local, self).__init__()
#         model_path = cfg.MODEL.PRETRAIN_PATH
#         pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
#         self.cos_layer = cfg.MODEL.COS_LAYER
#         self.neck = cfg.MODEL.NECK
#         self.neck_feat = cfg.TEST.NECK_FEAT
#
#         print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
#
#         if cfg.MODEL.SIE_CAMERA:
#             camera_num = camera_num
#         else:
#             camera_num = 0
#
#         if cfg.MODEL.SIE_VIEW:
#             view_num = view_num
#         else:
#             view_num = 0
#
#         self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
#         self.in_planes = self.base.in_planes
#         if pretrain_choice == 'imagenet':
#             self.base.load_param(model_path,hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
#             print('Loading pretrained ImageNet model......from {}'.format(model_path))
#
#         block = self.base.blocks[-1]
#         layer_norm = self.base.norm
#         self.b1 = nn.Sequential(
#             copy.deepcopy(block),
#             copy.deepcopy(layer_norm)
#         )
#         self.b2 = nn.Sequential(
#             copy.deepcopy(block),
#             copy.deepcopy(layer_norm)
#         )
#
#         self.num_classes = num_classes
#         # self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
#         # if self.ID_LOSS_TYPE == 'arcface':
#         #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#         #     self.classifier = Arcface(self.in_planes, self.num_classes,
#         #                               s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         # elif self.ID_LOSS_TYPE == 'cosface':
#         #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#         #     self.classifier = Cosface(self.in_planes, self.num_classes,
#         #                               s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         # elif self.ID_LOSS_TYPE == 'amsoftmax':
#         #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#         #     self.classifier = AMSoftmax(self.in_planes, self.num_classes,
#         #                                 s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         # elif self.ID_LOSS_TYPE == 'circle':
#         #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
#         #     self.classifier = CircleLoss(self.in_planes, self.num_classes,
#         #                                 s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         # else:
#         self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
#         self.classifier.apply(weights_init_classifier)
#         self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
#         self.classifier_1.apply(weights_init_classifier)
#         self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
#         self.classifier_2.apply(weights_init_classifier)
#         self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
#         self.classifier_3.apply(weights_init_classifier)
#         self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
#         self.classifier_4.apply(weights_init_classifier)
#
#         self.bottleneck = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck.bias.requires_grad_(False)
#         self.bottleneck.apply(weights_init_kaiming)
#         self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck_1.bias.requires_grad_(False)
#         self.bottleneck_1.apply(weights_init_kaiming)
#         self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck_2.bias.requires_grad_(False)
#         self.bottleneck_2.apply(weights_init_kaiming)
#         self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck_3.bias.requires_grad_(False)
#         self.bottleneck_3.apply(weights_init_kaiming)
#         self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck_4.bias.requires_grad_(False)
#         self.bottleneck_4.apply(weights_init_kaiming)
#
#         self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
#         print('using shuffle_groups size:{}'.format(self.shuffle_groups))
#         self.shift_num = cfg.MODEL.SHIFT_NUM
#         print('using shift_num size:{}'.format(self.shift_num))
#         self.divide_length = cfg.MODEL.DEVIDE_LENGTH
#         print('using divide_length size:{}'.format(self.divide_length))
#         self.rearrange = rearrange
#
#     def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
#
#         features = self.base(x, cam_label=cam_label, view_label=view_label)
#
#         # global branch
#         b1_feat = self.b1(features) # [64, 129, 768]
#         global_feat = b1_feat[:, 0]
#
#         # JPM branch
#         feature_length = features.size(1) - 1
#         patch_length = feature_length // self.divide_length
#         token = features[:, 0:1]
#
#         if self.rearrange:
#             x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
#         else:
#             x = features[:, 1:]
#         # lf_1
#         b1_local_feat = x[:, :patch_length]
#         b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
#         local_feat_1 = b1_local_feat[:, 0]
#
#         # lf_2
#         b2_local_feat = x[:, patch_length:patch_length*2]
#         b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
#         local_feat_2 = b2_local_feat[:, 0]
#
#         # lf_3
#         b3_local_feat = x[:, patch_length*2:patch_length*3]
#         b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
#         local_feat_3 = b3_local_feat[:, 0]
#
#         # lf_4
#         b4_local_feat = x[:, patch_length*3:patch_length*4]
#         b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
#         local_feat_4 = b4_local_feat[:, 0]
#
#         feat = self.bottleneck(global_feat)
#
#         local_feat_1_bn = self.bottleneck_1(local_feat_1)
#         local_feat_2_bn = self.bottleneck_2(local_feat_2)
#         local_feat_3_bn = self.bottleneck_3(local_feat_3)
#         local_feat_4_bn = self.bottleneck_4(local_feat_4)
#
#         if self.training:
#             if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
#                 cls_score = self.classifier(feat, label)
#             else:
#                 cls_score = self.classifier(feat)
#                 cls_score_1 = self.classifier_1(local_feat_1_bn)
#                 cls_score_2 = self.classifier_2(local_feat_2_bn)
#                 cls_score_3 = self.classifier_3(local_feat_3_bn)
#                 cls_score_4 = self.classifier_4(local_feat_4_bn)
#             return [cls_score, cls_score_1, cls_score_2, cls_score_3,
#                         cls_score_4
#                         ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
#                             local_feat_4]  # global feature for triplet loss
#         else:
#             if self.neck_feat == 'after':
#                 return torch.cat(
#                     [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
#             else:
#                 return torch.cat(
#                     [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)
#
#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         for i in param_dict:
#             self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
#         print('Loading pretrained model from {}'.format(trained_path))



__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'swin_base_patch4_window7_224': swin_base_patch4_window7_224,
    'swin_small_patch4_window7_224': swin_small_patch4_window7_224,
    'swin_tiny_patch4_window7_224': swin_tiny_patch4_window7_224,
}


__model_filename = {
    'swin_base_patch4_window7_224': "SOLIDER/swin_base_reid.pth",
}

def make_model(cfg, num_class, camera_num, view_num, semantic_weight):
    # if cfg.MODEL.NAME == 'transformer':
    #     if cfg.MODEL.JPM:
    #         model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
    #         print('===========building transformer with JPM module ===========')
    #     else:
    #         model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type, semantic_weight, __model_filename)  # always here
    #         print('===========building transformer===========')
    # else:
    #     model = Backbone(num_class, cfg)
    #     print('===========building ResNet===========')
    # return model

    return build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type, semantic_weight, __model_filename)  # always here
