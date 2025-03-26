import os
import logging
import collections

import time
from datetime import timedelta

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda import amp
import torch.distributed as dist

from utils.meter import AverageMeter
from loss.supcontrast import SupConLoss


def do_train_stage1(
        cfg,
        model,
        train_loader,
        optimizer,
        scheduler,
        local_rank
    ):
    n_epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD 
    ckpt_period = cfg.SOLVER.STAGE1.ckpt_period

    logger = logging.getLogger("transreid.train")
    logger.info('start training')

    _LOCAL_PROCESS_GROUP = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda'):
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    
    # train
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))

    image_features = []
    labels = []
    with torch.no_grad():
        for i, (img, vid, target_cam, target_view) in enumerate(train_loader):
            img = img.to(device)
            target = vid.to(device)
            with amp.autocast(enabled=True):
                image_feature = model(img, target, get_image = True)
                for i, img_feat in zip(target, image_feature):
                    labels.append(i)
                    image_features.append(img_feat.cpu())
    
        labels_list = torch.stack(labels, dim=0).to(device=device)
        image_features_list = torch.stack(image_features, dim=0).to(device=device)

        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
        num_image = labels_list.shape[0]
        i_ter = num_image // batch

    del labels, image_features

    for epoch in range(1, n_epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        iter_list = torch.randperm(num_image).to(device)
        for i in range(i_ter+1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i*batch:(i+1)* batch]
            else:
                b_list = iter_list[i*batch:num_image]
            
            target = labels_list[b_list]
            image_features = image_features_list[b_list]
            with amp.autocast(enabled=True):
                text_features = model(label=target, get_text=True)

            loss_i2t = xent(image_features, text_features, target, target)
            loss_t2i = xent(text_features, image_features, target, target)
            loss = loss_i2t + loss_t2i

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                logger.info(
                    f"[Epoch {epoch}] [Iteration {i+1} / {len(train_loader)}] "
                    f"Loss: {loss_meter.avg:.3f}, Acc: {acc_meter.avg:.3f}, Lr: {scheduler.get_lr(epoch)[0]:.2e}"
                )

        if epoch % ckpt_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))
