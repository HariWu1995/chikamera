# -*- coding: utf-8 -*-

from __future__ import print_function, division

import os
import sys
import yaml
import time
import argparse
from tqdm import tqdm
from shutil import copyfile
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets, transforms
from pytorch_metric_learning import losses, miners # pip install pytorch-metric-learning

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# from PIL import Image

import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ODFA import ODFA
from models import *
from losses import *
from dgfolder import DGFolder
from annotations import RandomErasing


version =  torch.__version__

# try:
#     # will be 3.x series
#     from apex import amp
#     from apex.fp16_utils import *
#     from apex.optimizers import FusedSGD

# except ImportError:
#     print('If you want to use low precision, i.e., fp16, '
#           'please install the apex with CUDA support (https://github.com/NVIDIA/apex) '
#           'and update pytorch to 1.0')


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long().cuda()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def train_model(opt, dataloaders, dataset_sizes, 
                model, criterion, optimizer, scheduler):

    since = time.time()
    use_gpu = torch.cuda.is_available()
    ckpt_dir = opt.ckpt_dir

    # best_model_wts = model.state_dict()
    best_val_acc = 0.0

    # loss history
    y_loss = {}
    y_loss['train'] = []
    y_loss['val'] = []

    y_err = {}
    y_err['train'] = []
    y_err['val'] = []

    # visual logging
    x_epoch = []
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="top1-err")

    def draw_curve(current_epoch):
        x_epoch.append(current_epoch)
        ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
        ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
        ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
        ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
        if current_epoch == 0:
            ax0.legend()
            ax1.legend()
        ax0.set_yscale('log')
        ax1.set_yscale('log')
        fig.savefig(os.path.join(ckpt_dir, 'train.jpg'))

    def save_model(model, suffix):
        if len(opt.gpu_ids) > 1:
            save_network(model.module, ckpt_dir, suffix)
        else:
            save_network(model, ckpt_dir, suffix)

    warm_up = 0.1
    warm_iteration = round(dataset_sizes['train'] / opt.batch_size) * opt.warm_epoch # first 5 epoch

    embedding_size = model.classifier.linear_num
    if opt.arcface:
        criterion_arcface = losses.ArcFaceLoss(num_classes=opt.nclasses, embedding_size=embedding_size)
    if opt.cosface: 
        criterion_cosface = losses.CosFaceLoss(num_classes=opt.nclasses, embedding_size=embedding_size)
    if opt.circle:
        criterion_circle = CircleLoss(m=0.25, gamma=32) # gamma = 64 may lead to a better result.
    if opt.triplet:
        miner = miners.MultiSimilarityMiner()
        criterion_triplet = losses.TripletMarginLoss(margin=0.3)
    if opt.lifted:
        criterion_lifted = losses.GeneralizedLiftedStructureLoss(neg_margin=1, pos_margin=0)
    if opt.contrast: 
        criterion_contrast = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    if opt.instance:
        criterion_instance = InstanceLoss(gamma=opt.ins_gamma)
    if opt.sphere:
        criterion_sphere = losses.SphereFaceLoss(num_classes=opt.nclasses, embedding_size=embedding_size, margin=4)
    
    for epoch in range(opt.total_epoch):
        print('Epoch {} / {}'.format(epoch, opt.total_epoch-1))
        # print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            # Phases 'train' and 'val' are visualized in two separate progress bars
            pbar = tqdm()
            pbar.reset(total=len(dataloaders[phase].dataset))
            phase_tracker = OrderedDict(phase="", Loss="", Acc="")

            running_loss = 0.
            running_corrects = 0.

            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs
                inputs, labels = data
                B, c, h, w = inputs.shape
                pbar.update(B)  # update the pbar even in the last batch
                if B < opt.batch_size: # skip the last batch
                    continue

                # wrap them in Variable
                if use_gpu:
                    inputs = inputs.cuda().detach()
                    labels = labels.cuda().detach()

                # if we use low precision, input also need to be fp16
                # if fp16:
                #     inputs = inputs.half()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                if opt.adv > 0 and i % opt.aiter == 0: 
                    inputs_adv = ODFA(model, inputs)
                    outputs_adv = model(inputs_adv)

                sm = nn.Softmax(dim=1)
                log_sm = nn.LogSoftmax(dim=1)

                if opt.return_feature: 
                    logits, ff = outputs
                    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                    ff = ff.div(fnorm.expand_as(ff))
                    loss = criterion(logits, labels) 
                    _, preds = torch.max(logits.data, 1)

                    if opt.adv > 0 and i % opt.aiter == 0:
                        logits_adv, _ = outputs_adv
                        loss += opt.adv * criterion(logits_adv, labels)
                    if opt.arcface:
                        loss += criterion_arcface(ff, labels) / B
                    if opt.cosface:
                        loss += criterion_cosface(ff, labels) / B
                    if opt.circle:
                        loss += criterion_circle(*convert_label_to_similarity(ff, labels)) / B
                    if opt.triplet:
                        hard_pairs = miner(ff, labels)
                        loss += criterion_triplet(ff, labels, hard_pairs) #/B
                    if opt.lifted:
                        loss += criterion_lifted(ff, labels) #/B
                    if opt.contrast:
                        loss += criterion_contrast(ff, labels) #/B
                    if opt.instance:
                        loss += criterion_instance(ff) / B
                    if opt.sphere:
                        loss += criterion_sphere(ff, labels) / B

                elif opt.model_arch == 'pcb':
                    part = {}
                    num_part = 6
                    for i in range(num_part):
                        part[i] = outputs[i]

                    score = sm(part[0]) + sm(part[1]) + sm(part[2]) + \
                            sm(part[3]) + sm(part[4]) + sm(part[5])
                    _, preds = torch.max(score.data, 1)

                    loss = criterion(part[0], labels)
                    for i in range(num_part-1):
                        loss += criterion(part[i+1], labels)

                else:  # norm
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    if opt.adv > 0 and i % opt.aiter == 0:
                        loss += opt.adv * criterion(outputs_adv, labels)

                del inputs

                # use extra DG Dataset (https://github.com/NVlabs/DG-Net#dg-market)
                if opt.enhance_DG and phase == 'train' and epoch > opt.total_epoch * 0.1:
                    # print("DG-Market is involved. It will double the training time.")
                    try:
                        _, batch = DGloader_iter.__next__()
                    except StopIteration: 
                        DGloader_iter = enumerate(dataloaders['DG'])
                        _, batch = DGloader_iter.__next__()
                    except UnboundLocalError: # first iteration
                        DGloader_iter = enumerate(dataloaders['DG'])
                        _, batch = DGloader_iter.__next__()
                        
                    inputs1, inputs2, _ = batch
                    inputs1 = inputs1.cuda().detach()
                    inputs2 = inputs2.cuda().detach()

                    # use memory in vivo loss (https://arxiv.org/abs/1912.11164)
                    outputs1 = model(inputs1)
                    if opt.return_feature:
                        outputs1, _ = outputs1
                    elif opt.model_arch == 'pcb':
                        for i in range(num_part):
                            part[i] = outputs1[i]
                        outputs1 = part[0] + part[1] + part[2] + part[3] + part[4] + part[5]

                    outputs2 = model(inputs2)
                    if opt.return_feature:
                        outputs2, _ = outputs2
                    elif opt.model_arch == 'pcb':
                        for i in range(num_part):
                            part[i] = outputs2[i]
                        outputs2 = part[0] + part[1] + part[2] + part[3] + part[4] + part[5]

                    mean_pred = sm(outputs1 + outputs2)
                    kl_loss = nn.KLDivLoss(reduction='batchmean')
                    reg = (kl_loss(log_sm(outputs2), mean_pred) + \
                           kl_loss(log_sm(outputs1), mean_pred)) / 2
                    loss += 0.01 * reg
                    del inputs1, inputs2

                # backward + optimize only if in training phase
                if phase == 'train' and epoch < opt.warm_epoch:
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss = loss * warm_up
                    # print(loss, warm_up)

                if phase == 'train':
                    # if fp16:
                    #     with amp.scale_loss(loss, optimizer) as scaled_loss:
                    #         scaled_loss.backward()
                    # else:
                    loss.backward()
                    optimizer.step()

                # statistics
                if int(version[0]) > 0 or int(version[2]) > 3: 
                    # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * B
                    phase_tracker["Loss"] = f"{loss.item():.4f}"
                else: 
                    # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * B
                    phase_tracker["Loss"] = f"{loss.data[0]:.4f}"
                
                del loss
                running_corrects += float(torch.sum(preds == labels.data))
                
                # Refresh the progress bar in every batch
                phase_tracker["phase"] = phase
                phase_tracker["Acc"] = f"{(float(torch.sum(preds == labels.data)) / B):.4f}"

                pbar.set_postfix(ordered_dict=phase_tracker)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val' \
            and best_val_acc < epoch_acc:
                best_val_acc = epoch_acc
                save_model(model, 'best')

            phase_tracker["phase"] = phase
            phase_tracker["Loss"] = f"{epoch_loss:.4f}"
            phase_tracker["Acc"] = f"{epoch_acc:.4f}"
            
            pbar.set_postfix(ordered_dict=phase_tracker)
            pbar.close()
            
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1-epoch_acc)

            # deep copy the model
            if phase == 'val' and (epoch+1) % opt.save_every == 0:
                last_model_wts = model.state_dict()
                print(f'Saving @ epoch {epoch+1} ...')
                save_model(model, str(epoch+1))

            if phase == 'val':
                draw_curve(epoch)
            elif phase == 'train':
                scheduler.step()

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc)

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_model(model, 'last')

    return model


def run(opt):
    print(opt)

    fp16 = opt.fp16
    data_dir = opt.data_dir

    # set gpu ids
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)
    opt.gpu_ids = gpu_ids

    if len(gpu_ids) > 0:
        # torch.cuda.set_device(gpu_ids[0])
        cudnn.enabled = True
        cudnn.benchmark = True

    # Load Data
    print('\n\nLoading data ...')
    if opt.model_arch == 'swin':
        h, w = 224, 224
    else:
        h, w = 256, 128

    transform_train_list = [
        # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((h, w), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((h, w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_val_list = [
        transforms.Resize(size=(h, w), interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    if opt.model_arch == 'pcb':
        transform_train_list = [
            transforms.Resize((384, 192), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        transform_val_list = [
            transforms.Resize(size=(384, 192), interpolation=3), #Image.BICUBIC
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

    if opt.erasing_p > 0:
        transform_train_list = transform_train_list + [RandomErasing(prob=opt.erasing_p, mean=[0., 0., 0.])]

    if opt.color_jitter:
        transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

    print(transform_train_list)
    data_transforms = {
        'train': transforms.Compose(transform_train_list),
          'val': transforms.Compose(transform_val_list),
    }

    train_dir = os.path.join(data_dir, 'train' + ('_all' if opt.train_all else ''))
    val_dir = os.path.join(data_dir, 'val')

    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(train_dir, data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(val_dir, data_transforms['val'])

    dataloaders = {
            x: DataLoader(image_datasets[x], batch_size=opt.batch_size, shuffle=True, 
                        num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)
        for x in ['train', 'val']
    }

    # Use extra DG-Market Dataset for training. 
    # Please download it from https://github.com/NVlabs/DG-Net#dg-market.
    if opt.enhance_DG and os.path.isdir(opt.DG_dir):
        # if os.path.isdir('../DG-Market') is False:
        #     os.system('gdown 126Gn90Tzpk3zWp2c7OBYPKc-ZjhptKDo')
        #     os.system('unzip DG-Market.zip -d ../')
        #     os.system('rm DG-Market.zip')
            
        image_datasets['DG'] = DGFolder(opt.DG_dir, data_transforms['train'])
        dataloaders['DG'] = DataLoader(image_datasets['DG'], batch_size=max(8, opt.batch_size // 2),
                                                shuffle=True, num_workers=2, pin_memory=True)
        DGloader_iter = enumerate(dataloaders['DG'])

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # inputs, classes = next(iter(dataloaders['train']))


    ######################################################################
    # Training the model
    # ------------------
    #
    # Now, let's write a general function to train a model. Here, we will illustrate:
    #
    # -  Scheduling the learning rate
    # -  Saving the best model
    #
    # In the following, parameter ``scheduler`` is an LR scheduler object from
    # ``torch.optim.lr_scheduler``.
    print('\n\nTraining ...')

    opt.return_feature = opt.arcface or opt.cosface or opt.circle or opt.triplet or \
                        opt.contrast or opt.instance or opt.lifted or opt.sphere

    # Load a pretrained model and reset final fully-connected layer.
    if opt.model_arch == 'densenet':
        model = ft_net_dense(len(class_names), opt.droprate, opt.stride, circle=opt.return_feature, linear_num=opt.linear_num)
    elif opt.model_arch == 'nasnet':
        model = ft_net_NAS(len(class_names), opt.droprate, linear_num=opt.linear_num)
    elif opt.model_arch == 'swin':
        model = ft_net_swin(len(class_names), opt.droprate, opt.stride, circle=opt.return_feature, linear_num=opt.linear_num)
    elif opt.model_arch == 'swin_v2':
        model = ft_net_swinv2(len(class_names), (h, w), opt.droprate, opt.stride, circle=opt.return_feature, linear_num=opt.linear_num)
    elif opt.model_arch == 'efficientnet':
        model = ft_net_efficient(len(class_names), opt.droprate, circle=opt.return_feature, linear_num=opt.linear_num)
    elif opt.model_arch == 'hrnet':
        model = ft_net_hr(len(class_names), opt.droprate, circle=opt.return_feature, linear_num=opt.linear_num)
    elif opt.model_arch == 'convnext':
        model = ft_net_convnext(len(class_names), opt.droprate, circle=opt.return_feature, linear_num=opt.linear_num)
    elif opt.model_arch == 'resnet':
        model = ft_net(len(class_names), opt.droprate, opt.stride, circle=opt.return_feature, ibn=opt.ibn, linear_num=opt.linear_num)
    elif opt.model_arch == 'pcb':
        model = PCB(len(class_names))
    else:
        raise ValueError(f'model_arch = {opt.model_arch} is not supported!')

    opt.nclasses = len(class_names)

    # model to gpu
    model = model.cuda()
    # print(model)

    optim_name = optim.SGD # apex.optimizers.FusedSGD
    # if opt.FSGD: # apex is needed
    #     optim_name = FusedSGD

    if sys.platform == 'linux' \
    and torch.cuda.get_device_capability()[0] > 6 \
    and len(opt.gpu_ids) == 1 \
    and int(version[0]) > 1:
        torch.set_float32_matmul_precision('high')
        print("Compiling model ... The 1st epoch may be slow, which is expected!")
        # https://huggingface.co/docs/diffusers/main/en/optimization/torch2.0
        model = torch.compile(model, mode="reduce-overhead", dynamic=True) # pytorch 2.0

    if len(opt.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
        if not opt.model_arch == 'pcb':
            ignored_params = list(map(id, model.module.classifier.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())
            classifier_params = model.module.classifier.parameters()
            optimizer_ft = optim_name([
                {'params': base_params, 'lr': 0.1*opt.lr},
                {'params': classifier_params, 'lr': opt.lr},
            ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)
        else:
            ignored_params  =  list(map(id, model.module.model.fc.parameters()))
            ignored_params += (list(map(id, model.module.classifier0.parameters()))
                            +  list(map(id, model.module.classifier1.parameters()))
                            +  list(map(id, model.module.classifier2.parameters()))
                            +  list(map(id, model.module.classifier3.parameters()))
                            +  list(map(id, model.module.classifier4.parameters()))
                            +  list(map(id, model.module.classifier5.parameters()))
                            #+ list(map(id, model.module.classifier6.parameters()))
                            #+ list(map(id, model.module.classifier7.parameters()))
                            )
            base_params       = filter(lambda p: id(p) not in ignored_params, model.module.parameters())
            classifier_params = filter(lambda p: id(p)     in ignored_params, model.module.parameters())
            optimizer_ft = optim_name([
                {'params': base_params, 'lr': 0.1*opt.lr},
                {'params': classifier_params, 'lr': opt.lr}
            ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)
    else:
        if not opt.model_arch == 'pcb':
            ignored_params = list(map(id, model.classifier.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
            classifier_params = model.classifier.parameters()
            optimizer_ft = optim_name([
                {'params': base_params, 'lr': 0.1*opt.lr},
                {'params': classifier_params, 'lr': opt.lr}
            ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)
        else:
            ignored_params  =  list(map(id, model.model.fc.parameters()))
            ignored_params += (list(map(id, model.classifier0.parameters())) 
                            +  list(map(id, model.classifier1.parameters()))
                            +  list(map(id, model.classifier2.parameters()))
                            +  list(map(id, model.classifier3.parameters()))
                            +  list(map(id, model.classifier4.parameters()))
                            +  list(map(id, model.classifier5.parameters()))
                            #+ list(map(id, model.classifier6.parameters()))
                            #+ list(map(id, model.classifier7.parameters()))
                        )
            base_params       = filter(lambda p: id(p) not in ignored_params, model.parameters())
            classifier_params = filter(lambda p: id(p)     in ignored_params, model.parameters())
            optimizer_ft = optim_name([
                {'params': base_params, 'lr': 0.1*opt.lr},
                {'params': classifier_params, 'lr': opt.lr}
            ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)

    # Decay LR by a factor of 0.1 every 40 epochs
    if opt.cosine:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, opt.total_epoch, eta_min=0.01*opt.lr)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=opt.total_epoch*2//3, gamma=0.1)

    dir_name = os.path.join(opt.ckpt_dir, opt.name)
    if os.path.isdir(dir_name) is False:
        os.makedirs(dir_name)
    opt.ckpt_dir = dir_name

    # record every run
    # copyfile('./train.py', dir_name + '/train.py')
    # copyfile('./model.py', dir_name + '/model.py')

    # save opts
    with open(f'{dir_name}/opts.yaml', 'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

    criterion = nn.CrossEntropyLoss()

    # if fp16:
    #     # model = network_to_half(model)
    #     # optimizer_ft = FP16_Optimizer(optimizer_ft, static_loss_scale=128.0)
    #     model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level="O1")

    model = train_model(opt, dataloaders, dataset_sizes, 
                        model, criterion, optimizer_ft, lr_scheduler)


if __name__ == "__main__":

    # Unit-test
    data_dir = "F:/__Datasets__/Market1501/preprocessed"
    extra_dir = "F:/__Datasets__/Market-DG"
    ckpt_dir = "C:/Users/Mr. RIAH/Documents/Projects/chikamera/checkpoints/PersonReID"
    ckpt_name = "Resnet50_ft"
    model_arch = "resnet"

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0 / 0,1,2 / 0,2')
    parser.add_argument('--name', default=ckpt_name, type=str, help='checkpoint name')

    # data
    parser.add_argument('--data_dir', default=data_dir, type=str, help='training dir path')
    parser.add_argument('--DG_dir', default=extra_dir, type=str, help='extra training dir path. ONLY use for Market-1501')
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--train_all', action='store_true', help='use all training data')
    parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
    parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
    parser.add_argument('--enhance_DG', action='store_false', help='use synthetic DG-Market Dataset for training. '
                                                                    'Please download it from https://github.com/NVlabs/DG-Net#dg-market.')
    # optimizer & trainer
    parser.add_argument('--lr', default=0.025, type=float, help='learning rate')
    parser.add_argument('--total_epoch', default=50, type=int, help='total training epoch')
    parser.add_argument('--warm_epoch', default=5, type=int, help='the first K epoch that needs warm up')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay. More Regularization Smaller Weight.')
    parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50%% memory')
    parser.add_argument('--FSGD', action='store_true', help='use fused sgd, which will speed up trainig slightly. apex is needed.')
    parser.add_argument('--cosine', action='store_false', help='use cosine lr')

    # backbone
    parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
    parser.add_argument('--stride', default=1, type=int, help='stride')
    parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
    parser.add_argument('--model_arch', default=model_arch, type=str, 
                                        choices=['densenet','nasnet','swin','swin_v2','efficientnet',
                                                 'hrnet','convnext','resnet','pcb'], help='model architecture')
    parser.add_argument('--ibn', action='store_true', help='use resnet+ibn')

    # loss
    parser.add_argument('--all_loss', action='store_false', help='use all losses')
    parser.add_argument('--arcface', action='store_true', help='use ArcFace loss')
    parser.add_argument('--circle', action='store_true', help='use Circle loss')
    parser.add_argument('--cosface', action='store_true', help='use CosFace loss')
    parser.add_argument('--contrast', action='store_true', help='use contrast loss')
    parser.add_argument('--instance', action='store_true', help='use instance loss')
    parser.add_argument('--triplet', action='store_true', help='use triplet loss')
    parser.add_argument('--lifted', action='store_true', help='use lifted loss')
    parser.add_argument('--sphere', action='store_true', help='use sphere loss')
    parser.add_argument('--adv', default=0.0, type=float, help='use adv loss as 1.0')
    parser.add_argument('--aiter', default=10, type=float, help='use adv loss per iteration')
    parser.add_argument('--ins_gamma', default=32, type=int, help='gamma for instance loss')

    # callbacks
    parser.add_argument('--ckpt_dir', default=ckpt_dir, type=str, help='./checkpoints')
    parser.add_argument('--save_every', default=10, type=int, help='number of every epochs to save weights')

    opt = parser.parse_args()

    if opt.all_loss:
        opt.arcface = True
        opt.circle = True
        opt.cosface = True
        opt.contrast = True
        opt.instance = True
        opt.triplet = True
        opt.lifted = True
        opt.sphere = True
        opt.adv = max(opt.adv, 0.1)

    run(opt)
