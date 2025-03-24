"""
Reproduce the below code in script:

python scripts/main.py \
    --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml \
    --root $PATH_TO_DATA \
    model.load_weights log/osnet_x1_0_market1501_softmax_cosinelr/model.pth.tar-250 \
    test.evaluate True

from https://kaiyangzhou.github.io/deep-person-reid/
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import inspect

current_dir = os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(current_dir)

# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

from .scripts.main import *


if __name__ == "__main__":

    # Test multi-domain
    # ckpt_name = "osnet_ain_x1_0_domain_multi.pth.tar"
    # config_name = "im_osnet_ain_x1_0_softmax_256x128_amsgrad_cosine"

    # Test Market-1501 / DukeMTMC
    # data_name = 'market1501'
    data_name = 'dukemtmcreid'
    # ckpt_name = "osnet_x1_0_market.pth"
    ckpt_name = "osnet_x1_0_duke.pth"
    config_name = "im_osnet_x1_0_softmax_256x128_amsgrad_cosine"
    
    default_data_root = "F:/__Datasets__"
    default_ckpt_path = f"./checkpoints/TorchReID/{ckpt_name}"
    default_config_file = os.path.join(current_dir, 'configs', f'{config_name}.yaml')

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', type=str, default=default_config_file, help='path to config file')
    parser.add_argument('-r','--root', type=str, default=default_data_root, help='path to data root')
    parser.add_argument('-s','--sources', type=str, default=[data_name], nargs='+', help='source datasets (delimited by space)')
    parser.add_argument('-t','--targets', type=str, default=[data_name], nargs='+', help='target datasets (delimited by space)')
    parser.add_argument('-tr','--transforms', type=str, nargs='+', help='data augmentation')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help='Modify config options using the command-line')

    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)
    check_cfg(cfg)

    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print('*'*19)
    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))
    print('*'*19)

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    datamanager = build_datamanager(cfg)

    print('\n\nBuilding model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(
                name = cfg.model.name,
         num_classes = datamanager.num_train_pids,
                loss = cfg.loss.name,
          pretrained = cfg.model.pretrained,
             use_gpu = cfg.use_gpu
    )
    n_params, flops = compute_model_complexity(model, (1, 3, cfg.data.height, cfg.data.width))
    print('\nModel complexity: #params = {:,} #flops = {:,}'.format(n_params, flops))

    # HARDCODE
    cfg.test.evaluate = True # test only
    cfg.model.load_weights = default_ckpt_path
    load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(cfg))

    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
        )

    print('\n\nBuilding {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type))
    engine = build_engine(cfg, datamanager, model, optimizer, scheduler)
    
    # Inference
    print('\n\nInferencing ...')
    engine.run(**engine_run_kwargs(cfg))

    # engine.extract_features(input)

