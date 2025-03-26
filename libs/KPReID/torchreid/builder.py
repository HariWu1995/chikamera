import os
from pathlib import Path

import torch
import torch.nn as nn

from torchreid import models
from torchreid.data import ImageDataManager, VideoDataManager, compute_parts_num_and_names
from torchreid.optim import build_optimizer, build_lr_scheduler
from torchreid.engine import (ImageSoftmaxEngine, ImageTripletEngine, ImagePartBasedEngine, 
                              VideoSoftmaxEngine, VideoTripletEngine)
from torchreid.utils import (Logger, Writer, collect_env_info,
                            set_random_seed, check_isfile, 
                            load_checkpoint, load_pretrained_weights,
                     resume_from_checkpoint, compute_model_complexity,)
from torchreid.utils.engine_state import EngineState

from .default_config import (
    imagedata_kwargs,
    videodata_kwargs,
    optimizer_kwargs,
    get_default_config,
    lr_scheduler_kwargs,
    display_config_diff,
)


def build_datamanager(cfg):
    if cfg.data.type == "image":
        return ImageDataManager(**imagedata_kwargs(cfg))
    else:
        return VideoDataManager(**videodata_kwargs(cfg))


def build_engine(cfg, datamanager, model, optimizer, scheduler, writer, engine_state):
    if cfg.data.type == "image":
        if cfg.loss.name == "softmax":
            engine = ImageSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                save_model_flag=cfg.model.save_model_flag,
                writer=writer,
                engine_state=engine_state,
            )

        elif cfg.loss.name == "triplet":
            engine = ImageTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                save_model_flag=cfg.model.save_model_flag,
                writer=writer,
                engine_state=engine_state,
            )

        elif cfg.loss.name == "part_based":
            engine = ImagePartBasedEngine(
                datamanager,
                model,
                optimizer=optimizer,
                loss_name=cfg.loss.part_based.name,
                config=cfg,
                margin=cfg.loss.triplet.margin,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                save_model_flag=cfg.model.save_model_flag,
                writer=writer,
                engine_state=engine_state,
                dist_combine_strat=cfg.test.part_based.dist_combine_strat,
                batch_size_pairwise_dist_matrix=cfg.test.batch_size_pairwise_dist_matrix,
                mask_filtering_training=cfg.model.kpr.mask_filtering_training,
                mask_filtering_testing=cfg.model.kpr.mask_filtering_testing,
            )

    else:
        if cfg.loss.name == "softmax":
            engine = VideoSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                pooling_method=cfg.video.pooling_method,
                save_model_flag=cfg.model.save_model_flag,
                writer=writer,
                engine_state=engine_state,
            )

        else:
            engine = VideoTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                save_model_flag=cfg.model.save_model_flag,
                writer=writer,
                engine_state=engine_state,
            )

    return engine


def reset_config(cfg, args):
    if hasattr(args, "root"):
        if args.root:
            cfg.data.root = args.root
    if hasattr(args, "sources"):
        if args.sources:
            cfg.data.sources = args.sources
    if hasattr(args, "targets"):
        if args.targets:
            cfg.data.targets = args.targets
    if hasattr(args, "transforms"):
        if args.transforms:
            cfg.data.transforms = args.transforms
    if hasattr(args, "job_id"):
        if args.job_id:
            cfg.project.job_id = args.job_id
    if hasattr(args, "save_dir"):
        if args.save_dir:
            cfg.data.save_dir = args.save_dir
    if hasattr(args, "ckpt_file"):
        if args.ckpt_file:
            cfg.model.load_weights = args.ckpt_file
    if hasattr(args, "inference"):
        if args.inference:
            cfg.inference.enabled = args.inference


def build_config(args=None, config=None, config_path=None, display_diff=False):
    cfg = get_default_config()
    default_cfg_copy = cfg.clone()

    # use gpu if available
    cfg.use_gpu = torch.cuda.is_available()

    # overwrite default configs with provided args
    if config_path is not None:
        cfg.merge_from_file(config_path)
        cfg.project.config_file = os.path.basename(config_path)
    if config is not None:
        cfg.merge_from_other_cfg(config)
    if args is not None:
        reset_config(cfg, args)
        cfg.merge_from_list(args.opts)

    # set parts information (number of parts K and each part name),
    # depending on the original loaded masks size or the transformation applied:
    compute_parts_num_and_names(cfg)

    # load model from disk if provided
    if cfg.model.load_weights:
        assert check_isfile(cfg.model.load_weights), \
            f"Weight file `{cfg.model.load_weights}` doesn't exist."
    
    if cfg.model.load_weights \
    and cfg.model.load_config:
        checkpoint = load_checkpoint(cfg.model.load_weights)

        # if a config is saved with the model weights, 
        # overwrite the current model config with the saved config
        if "config" in checkpoint:
            print(f"Overwrite current config with {cfg.model.load_weights}")

            kpr_config = checkpoint["config"].model.kpr
            if checkpoint["config"].data.sources[0] != cfg.data.targets[0]:
                print("WARNING: the train dataset of the loaded model is different "
                            "from the target dataset in the current config.")

            kpr_config.pop("backbone_pretrained_path", None)
            kpr_config.masks.pop("dir", None)

            if "keypoints" in kpr_config:
                kpr_config.keypoints.pop("enabled", None)
                kpr_config.keypoints.pop("kp_dir", None)

            if cfg.model.discard_test_params:
                kpr_config.masks.pop("dir", None)
                kpr_config.masks.pop("preprocess", None)
                kpr_config.pop("mask_filtering_testing", None)
                kpr_config.pop("learnable_attention_enabled", None)
                kpr_config.pop("test_embeddings", None)
                kpr_config.pop("test_use_target_segmentation", None)
                kpr_config.pop("testing_binary_visibility_score", None)

            cfg.model.kpr.merge_from_other_cfg(kpr_config)

            if "vit" in checkpoint["config"].model:
                if cfg.model.discard_test_params:
                    checkpoint["config"].model.promptable_trans.pop("disable_inference_prompting", None)
                cfg.model.promptable_trans.merge_from_other_cfg(checkpoint["config"].model.promptable_trans)
            
            if "transreid" in checkpoint["config"].model:
                cfg.model.transreid.merge_from_other_cfg(checkpoint["config"].model.transreid)
        else:
            print(f"Could not load config from file {cfg.model.load_weights}")

    # display differences between default and final config
    if display_diff:
        display_config_diff(cfg, default_cfg_copy)

    # init save dir
    cfg.data.save_dir = os.path.join(cfg.data.save_dir, str(cfg.project.job_id))
    if os.path.isdir(cfg.data.save_dir) is False:
        os.makedirs(cfg.data.save_dir)
    print("Save dir created at {}".format(os.path.join(Path().resolve(), cfg.data.save_dir)))

    return cfg


def build_model(cfg, num_train_pids=1, cam_num=0, view=0):
    model = models.build_model(
         num_classes = num_train_pids,
                name = cfg.model.name,
                loss = cfg.loss.name,
          pretrained = cfg.model.pretrained,
             use_gpu = cfg.use_gpu,
             cam_num = cam_num,
                view = view,
              config = cfg,
    )
    cfg.model.kpr.spatial_feature_shape = model.spatial_feature_shape
    if cfg.model.compute_complexity:
        num_params, flops = compute_model_complexity(model, cfg)
        print("Model complexity: #params = {:,} #flops = {:,}".format(num_params, flops))
    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)
    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()
    return model


def build_model_engine(cfg):
    if cfg.project.debug_mode:
        torch.autograd.set_detect_anomaly(True)

    logger = Logger(cfg)
    writer = Writer(cfg)

    set_random_seed(cfg.train.seed)

    print('*'*19)
    print("Show configuration\n{}\n".format(cfg))
    print("Collecting env info ...")
    print('** System info **\n{}\n'.format(collect_env_info()))
    print('*'*19)

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    datamanager = build_datamanager(cfg)
    engine_state = EngineState(cfg.train.start_epoch, cfg.train.max_epoch)
    writer.init_engine_state(engine_state, cfg.model.kpr.masks.parts_num)

    print("\n\nBuilding model: {}".format(cfg.model.name))
    model = build_model(cfg, datamanager.num_train_pids, 
                             datamanager.train_loader.dataset.cam_num, 
                             datamanager.train_loader.dataset.view)
    logger.add_model(model)

    optimizer = build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = build_lr_scheduler(optimizer, **lr_scheduler_kwargs(cfg))

    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
        )

    print("\n\nBuilding {}-engine for {}-reid".format(cfg.loss.name, cfg.data.type))
    engine = build_engine(cfg, datamanager, model, optimizer, scheduler, writer, engine_state)

    return engine, model
