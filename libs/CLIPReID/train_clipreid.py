import os
import argparse

import random as rd
import numpy as np
import torch

from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from utils.logger import setup_logger
from processor.processor_clipreid_stage1 import do_train_stage1
from processor.processor_clipreid_stage2 import do_train_stage2


def set_seed(seed):
    rd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--config_file", default="configs/person/vit_clipreid.yml", type=str, help="path to config file")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Modify config options using the command-line")
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    train_loader_stage2, train_loader_stage1, val_loader, \
            num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer_1stage = make_optimizer_1stage(cfg, model)
    scheduler_1stage = create_scheduler(optimizer = optimizer_1stage, 
                                        num_epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS, 
                                            lr_min = cfg.SOLVER.STAGE1.LR_MIN,
                                    warmup_lr_init = cfg.SOLVER.STAGE1.WARMUP_LR_INIT, 
                                          warmup_t = cfg.SOLVER.STAGE1.WARMUP_EPOCHS, 
                                        noise_range = None)
    do_train_stage1(
        cfg,
        model,
        train_loader_stage1,
        optimizer_1stage,
        scheduler_1stage,
        args.local_rank
    )

    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(cfg, model, center_criterion)
    scheduler_2stage = WarmupMultiStepLR(optimizer = optimizer_2stage, 
                                        milestones = cfg.SOLVER.STAGE2.STEPS, 
                                             gamma = cfg.SOLVER.STAGE2.GAMMA, 
                                     warmup_factor = cfg.SOLVER.STAGE2.WARMUP_FACTOR,
                                      warmup_iters = cfg.SOLVER.STAGE2.WARMUP_ITERS, 
                                     warmup_method = cfg.SOLVER.STAGE2.WARMUP_METHOD)
    do_train_stage2(
        cfg,
        model,
        center_criterion,
        train_loader_stage2,
        val_loader,
        optimizer_2stage,
        optimizer_center_2stage,
        scheduler_2stage,
        loss_func,
        num_query, args.local_rank
    )

