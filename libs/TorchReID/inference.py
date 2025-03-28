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
    parser.add_argument('-c','--config-file', type=str, default=default_config_file, help='path to config file')

    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    # Modeling
    print('\n\nBuilding model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(
         num_classes = 100, # NOTE: any number because we do NOT use classfier
                name = cfg.model.name,
                loss = cfg.loss.name,
          pretrained = cfg.model.pretrained,
             use_gpu = cfg.use_gpu,
    )
    n_params, flops = compute_model_complexity(model, (1, 3, cfg.data.height, cfg.data.width))
    print('\nModel complexity: #params = {:,} #flops = {:,}'.format(n_params, flops))
    
    load_pretrained_weights(model, default_ckpt_path)
    
    model.eval()
    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()
    
    # Inference
    print('\n\nInferencing ...')
    test_case = torch.rand(1, 3, cfg.data.height, cfg.data.width)
    if cfg.use_gpu:
        test_case = test_case.cuda()

    output = model(test_case)
    print(output.shape)


