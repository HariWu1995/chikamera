# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import time
import yaml
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import datasets, transforms

import math
import numpy as np
import scipy.io

import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models import *

# try:
#     from apex.fp16_utils import *   # will be 3.x series
# except ImportError:
#     print('If you want to use low precision, i.e., fp16, '
#           'please install the apex with CUDA support (https://github.com/NVIDIA/apex) '
#           'and update pytorch to 1.0')


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def load_model(model_structure, opt):
    model_path = os.path.join(opt.ckpt_dir, opt.ckpt_name, 'net_%s.pth' % opt.which_epoch)
    network = load_network(model_structure, model_path)
    return network


def extract_feature(model, dataloaders, subset, opt):
    print(f'\n\nExtracting features for `{subset}` ...')
    if opt.linear_num <= 0:
        if opt.model_arch in ['swin','swin_v2','densenet','convnext']:
            opt.linear_num = 1024
        elif opt.model_arch in ['efficientnet']:
            opt.linear_num = 1792
        elif opt.model_arch in ['nasnet']:
            opt.linear_num = 4032
        else:
            opt.linear_num = 2048

    dataloader = dataloaders[subset]
    features = []

    count = 0
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        img, label = data
        n, c, h, w = img.size()
        count += n
        # print(count)
        
        if opt.model_arch == "pcb":
            ff = torch.FloatTensor(n, 2048, 6).zero_().cuda() # we have 6 parts
        else:
            ff = torch.FloatTensor(n, opt.linear_num).zero_().cuda()

        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in opt.ms:
                if scale != 1:
                    # bicubic is only available in pytorch >= 1.1
                    input_img = F.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img) 
                ff += outputs
        ff /= 2

        # norm feature
        if opt.model_arch == 'pcb':
            # feature size (n, 2048, 6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        # if i == 0:
        #     features = torch.FloatTensor(len(dataloader.dataset), ff.shape[1])
        # features = torch.cat((features, ff.data.cpu()), 0)
        features.append(ff)
                
    features = torch.cat(features, dim=0)
    features = features.detach().cpu().numpy()
    return features


def get_id_market(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = os.path.basename(path)
        label = filename[:4]
        camera = filename.split('c')[1]
        if label[:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


def get_id_mtmc(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = os.path.basename(path)
        label = filename[:4]
        camera = filename.split('_')[2][:2]
        if label[:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera))
    return camera_id, labels


def get_id_duke(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = os.path.basename(path)
        label = filename[:4]
        camera = filename.split('_')[1][1]
        if label[:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


def run(opt):
    print(opt)

    if opt.data_name == 'market':
        get_id = get_id_market
    elif opt.data_name == 'duke':
        get_id = get_id_duke    # FIXME
    elif opt.data_name == 'mtmc':
        get_id = get_id_mtmc
    else:
        raise ValueError(f'`data_name` = {opt.data_name} is not supported!')

    str_ids = opt.gpu_ids.split(',')
    data_dir = opt.test_dir

    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)

    str_ms = opt.ms.split(',')
    ms = []
    for s in str_ms:
        s_f = float(s)
        ms.append(math.sqrt(s_f))
    opt.ms = ms
    print('\nMultiple scales:', opt.ms)

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True
    use_gpu = torch.cuda.is_available()

    # Load Data
    print('\n\nLoading data ...')
    if opt.model_arch == 'swin':
        h, w = 224, 224
    else:
        h, w = 256, 128

    data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        # Ten Crop        
        # transforms.TenCrop(224),
        # transforms.Lambda(lambda crops: torch.stack(
        #       [transforms.ToTensor()(crop) for crop in crops]
        # )),
        # transforms.Lambda(lambda crops: torch.stack(
        #       [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
        #       for crop in crops]
        # )),
    ])

    if opt.model_arch == 'pcb':
        data_transforms = transforms.Compose([
            transforms.Resize((384, 192), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])
        h, w = 384, 192

    subsets = ['gallery','query']
    if opt.multi:
        subsets.append('multi-query')

    image_datasets = {
            x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) 
        for x in subsets
    }
    dataloaders = {
            x: DataLoader(image_datasets[x], batch_size=opt.batch_size, shuffle=False, num_workers=16) 
        for x in subsets
    }

    class_names = image_datasets['query'].classes

    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs

    gallery_cam, gallery_label = get_id(gallery_path)
    query_cam, query_label = get_id(query_path)

    if opt.multi:
        mquery_path = image_datasets['multi-query'].imgs
        mquery_cam, mquery_label = get_id(mquery_path)

    # Load Trained model
    print('\n\nLoading model ...')

    if opt.model_arch == 'densenet':
        model = ft_net_dense(opt.nclasses, stride=opt.stride, linear_num=opt.linear_num)
    elif opt.model_arch == 'nasnet':
        model = ft_net_NAS(opt.nclasses, linear_num=opt.linear_num)
    elif opt.model_arch == 'swin':
        model = ft_net_swin(opt.nclasses, stride=opt.stride, linear_num=opt.linear_num)
    elif opt.model_arch == 'swin_v2':
        model = ft_net_swinv2(opt.nclasses, (h, w), stride=opt.stride, linear_num=opt.linear_num)
    elif opt.model_arch == 'efficientnet':
        model = ft_net_efficient(opt.nclasses, linear_num=opt.linear_num)
    elif opt.model_arch == 'hrnet':
        model = ft_net_hr(opt.nclasses, linear_num=opt.linear_num)
    elif opt.model_arch == 'convnext':
        model = ft_net_convnext(opt.nclasses, linear_num=opt.linear_num)
    elif opt.model_arch == 'resnet':
        model = ft_net(opt.nclasses, stride=opt.stride, ibn=opt.ibn, linear_num=opt.linear_num)
    elif opt.model_arch == 'pcb':
        model = PCB(opt.nclasses)
    else:
        raise ValueError(f'model_arch = {opt.model_arch} is not supported!')

    # if opt.fp16:
    #     model = network_to_half(model)

    model = load_model(model, opt)
    # print(model)

    # Remove the final fc layer and classifier layer
    if opt.model_arch == 'pcb':
        model = PCB_test(model)
    else:
        model.classifier.classifier = nn.Sequential()

    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.cuda()

    # fuse conv and bn for faster inference, and it does not work for transformers.
    # Comment out this following line if you do not want to fuse conv & bn.
    # model = fuse_all_conv_bn(model)

    # We can optionally trace the forward method with PyTorch JIT so it runs faster.
    # To do so, call `.trace` on reparamtrized module with dummy inputs expected by module.
    # Comment out this following line if you do not want to trace.
    # dummy_forward_input = torch.rand(opt.batch_size, 3, h, w).cuda()
    # model = torch.jit.trace(model, dummy_forward_input)

    # convert to TensorRT feeding sample data as input
    # print('Please pip install nvidia-pyindex; pip install nvidia-tensorrt; '
    #     'git clone https://github.com/NVIDIA-AI-IOT/torch2trt; cd torch2trt; python setup.py install')
    # from torch2trt import torch2tr
    # model = torch2trt(model, [x], fp16_mode=True, max_batch_size=opt.batch_size)

    # Extract feature
    since = time.time()
    with torch.no_grad():
        query_feature = extract_feature(model, dataloaders, 'query', opt)
        gallery_feature = extract_feature(model, dataloaders, 'gallery', opt)
        if opt.multi:
            mquery_feature = extract_feature(model, dataloaders, 'multi-query', opt)
    time_elapsed = time.time() - since
    print('\n\nExtraction completes in {:.0f}m {:.2f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # Save to Matlab for check
    result_dir = f'{data_dir}/results_{opt.ckpt_name}'
    if os.path.isdir(result_dir) is False:
        os.makedirs(result_dir)

    result = {
        'gallery_feat': gallery_feature,
        'gallery_label': gallery_label,
        'gallery_cam': gallery_cam,
        'query_feat': query_feature,
        'query_label': query_label,
        'query_cam': query_cam,
    }
    scipy.io.savemat(f'{result_dir}/result.mat', result)

    # result_path = f'{result_dir}/result.txt'
    # os.system('python evaluate_gpu.py | tee -a %s' % result_path)

    if opt.multi:
        result = {
            'mquery_feat': mquery_feature,
            'mquery_label': mquery_label,
            'mquery_cam': mquery_cam,
        }
        scipy.io.savemat(f'{result_dir}/multi_query.mat', result)


if __name__ == "__main__":

    # Unit-test
    data_dir = "F:/__Datasets__/DukeMTMC/preprocessed"
    data_name = "duke"
    ckpt_dir = "C:/Users/Mr. RIAH/Documents/Projects/chikamera/checkpoints/PersonReID"
    ckpt_name = "ResNet50"
    model_arch = "resnet"

    # Options
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0 / 0,1,2 / 0,2')
    parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3 or last')
    parser.add_argument('--test_dir', default=data_dir, type=str, help='./test_data')
    parser.add_argument('--ckpt_dir', default=ckpt_dir, type=str, help='./checkpoints')
    parser.add_argument('--ckpt_name', default=ckpt_name, type=str, help='checkpoint sub-directory')
    parser.add_argument('--data_name', default=data_name, type=str, choices=['market','duke','mtmc'], 
                                                                    help='dataset name')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')

    # backbone
    parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
    parser.add_argument('--stride', default=2, type=int, help='stride')
    parser.add_argument('--model_arch', default=model_arch, type=str, 
                        choices=['densenet','nasnet','swin','swin_v2','efficientnet',
                                'hrnet','convnext','resnet','pcb'], help='model architecture')
    parser.add_argument('--ibn', action='store_true', help='use resnet+ibn')

    # optimization
    parser.add_argument('--multi', action='store_true', help='use multiple query' )
    parser.add_argument('--fp16', action='store_true', help='use fp16.' )
    parser.add_argument('--ms', default='1', type=str, help='multiple_scale: e.g. 1 / 1,1.1 / 1,1.1,1.2')

    opt = parser.parse_args()

    # Configurations
    # load the training config
    config_path = os.path.join(opt.ckpt_dir, opt.ckpt_name, 'opts.yaml')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    opt.fp16 = config.get('fp16', opt.fp16)
    opt.ibn = config.get('ibn', opt.ibn)
    opt.stride = config.get('stride', opt.stride)
    opt.nclasses = config.get('nclasses', 751)
    opt.linear_num = config.get('linear_num', opt.linear_num)

    run(opt)

