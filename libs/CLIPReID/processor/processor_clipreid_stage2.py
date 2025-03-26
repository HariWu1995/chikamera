import os
import logging
from tqdm import tqdm

import time
from datetime import timedelta

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda import amp
import torch.distributed as dist

from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from loss.supcontrast import SupConLoss


def do_train_stage2(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_fn,
        num_query, 
        local_rank = -1,
    ):

    n_instances = cfg.DATALOADER.NUM_INSTANCE
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    ckpt_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    batch_size = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    n_epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')

    _LOCAL_PROCESS_GROUP = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda'):
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes
    device = torch.device(device)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    
    # train
    all_start_time = time.monotonic()

    i_ter = num_classes // batch_size
    batch_r = num_classes - batch_size * (num_classes // batch_size)
    if batch_r != 0 :
        i_ter += 1
    
    text_features = []
    with torch.no_grad():
        for i in range(i_ter):
            if i+1 != i_ter:
                l_list = torch.arange(i * batch_size, (i+1) * batch_size)
            else:
                l_list = torch.arange(i * batch_size, num_classes)
            with amp.autocast(enabled=True):
                text_feature = model(label=l_list, get_text=True)
            text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0).to(device=device)

    for epoch in range(1, n_epochs + 1):
        start_time = time.time()

        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step()

        model.train()
    
        for i, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device) if cfg.MODEL.SIE_CAMERA else None
            target_view = target_view.to(device) if cfg.MODEL.SIE_VIEW else None
            with amp.autocast(enabled=True):
                score, feat, image_features = model(x = img, 
                                                label = target, 
                                            cam_label = target_cam, 
                                           view_label = target_view)
                logits = image_features @ text_features.t()
                loss = loss_fn(score, feat, target, target_cam, logits)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (logits.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                logger.info(
                    f"[Epoch {epoch}] [Iteration {i+1} / {len(train_loader)}] "
                    f"Loss: {loss_meter.avg:.3f}, Acc: {acc_meter.avg:.3f}, Lr: {scheduler.get_lr()[0]:.2e}"
                )

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (i + 1)

        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info(
                "Epoch {} done. Time per batch: {:.3f} [s] Speed: {:.1f} [samples/s]"
                .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch)
            )

        if epoch % ckpt_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for i, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device) if cfg.MODEL.SIE_CAMERA else None
                            target_view = target_view.to(device) if cfg.MODEL.SIE_VIEW else None
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for i, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device) if cfg.MODEL.SIE_CAMERA else None
                        target_view = target_view.to(device) if cfg.MODEL.SIE_VIEW else None
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print("Please check details @", cfg.OUTPUT_DIR)


def do_inference(cfg, model, val_loader, num_query):

    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda'):
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)
    model.eval()

    for i, (img, pid, camid, camids,
            target_view, imgpath) in tqdm(enumerate(val_loader), total=len(val_loader)):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device) if cfg.MODEL.SIE_CAMERA else None
            target_view = target_view.to(device) if cfg.MODEL.SIE_VIEW else None
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))

    return cmc[0], cmc[4]

