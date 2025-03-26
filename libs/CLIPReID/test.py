import os
import argparse

from config import cfg_base as cfg
from utils.logger import setup_logger
from model.make_model import make_model
from processor.processor import do_inference
from datasets.make_dataloader import make_dataloader


def run(args, num_trials: int = 10):

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, \
    num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)

    if cfg.DATASETS.NAMES == 'VehicleID':

        for trial in range(num_trials):
            train_loader, train_loader_normal, val_loader, \
            num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
            rank_1, rank5, mAP = do_inference(cfg, model, val_loader, num_query)
            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank5
                all_mAP = mAP
            else:
                all_rank_1 = all_rank_1 + rank_1
                all_rank_5 = all_rank_5 + rank5
                all_mAP = all_mAP + mAP
            logger.info("rank_1: {}, rank_5: {}, trial: {}".format(rank_1, rank5, mAP, trial))
            
        avg_rank_1 = all_rank_1.sum() / num_trials
        avg_rank_5 = all_rank_5.sum() / num_trials
        avg_mAP    =    all_mAP.sum() / num_trials
        logger.info(f"∑rank_1: {avg_rank_1:.1%}, ∑rank_5: {avg_rank_5:.1%}, ∑mAP {avg_mAP:.1%}")

    else:
       do_inference(cfg, model, val_loader, num_query)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ReID Testing")
    parser.add_argument("--config_file", default="configs/person/vit_base.yml", type=str, help="path to config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Modify config options using the command-line")

    args = parser.parse_args()
    run(args)

