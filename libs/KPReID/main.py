import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import inspect

current_dir = os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(current_dir)

from torchreid.builder import build_config, build_model_engine
from torchreid.default_config import engine_run_kwargs
from tools.extract_part_features import extract_reid_features


def run_process(
        config_file: str = '',
        ckpt_path: str = '',
        save_root: str = './temp/kpreid',
        data_root: str = './datasets',
        data_name: str = 'market1501',
        inference: bool = False,
        job_id = None,
    ):
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m','--ckpt-file', type=str, default=ckpt_path, help='path to model checkpoint file')
    parser.add_argument('-c','--config-file', type=str, default=config_file, help='path to config file')
    parser.add_argument('-s','--sources', type=str, nargs='+', default=[data_name], help='source datasets (delimited by space)')
    parser.add_argument('-t','--targets', type=str, nargs='+', default=[data_name], help='target datasets (delimited by space)')
    parser.add_argument('-tr','--transforms', type=str, nargs='+', help='data augmentation')
    parser.add_argument('-r','--root', type=str, default=data_root, help='path to data root')
    parser.add_argument('--save_dir', type=str, default=save_root, help='path to output dir')
    parser.add_argument('--inference', type=bool, default=inference, help='Inference mode')
    parser.add_argument('-j','--job-id', type=int, default=job_id, help='Slurm job id')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help='Modify config options using the command-line')
    args = parser.parse_args()

    cfg = build_config(args=args, config_path=args.config_file, display_diff=True)

    engine, model = build_model_engine(cfg)
    if not cfg.inference.enabled:
        print(f"\nExperiment {cfg.project.experiment_id} "
               f"with job id {cfg.project.job_id} @ {cfg.project.start_time}")
        engine.run(**engine_run_kwargs(cfg))

    else:
        print("Starting inference on external data")
        extract_reid_features(cfg, cfg.inference.input_folder, cfg.data.save_dir, model)


if __name__ == "__main__":

    # Test DukeMTMC-Occ
    # model = 'solider'
    # data_name = ['occluded_duke','DukeMTMC-Occluded']
    # ckpt_file = "kpr_occDuke_SOLIDER.pth.tar"
    # config_file = "kpr_occ_duke_test.yaml"
    # job_id = 19_05_1995

    # Test Market-1501
    model = 'imagenet'
    data_name = ['market1501','Market1501']
    ckpt_file = "kpr_market_ImageNet.pth.tar"
    config_file = "kpr_market_test.yaml"
    job_id = 27_11_1995
    
    default_data_root = "F:/__Datasets__"
    default_ckpt_path = f"./checkpoints/KPReID/{ckpt_file}"
    default_config_path = os.path.join(current_dir, 'configs', 'kpr', model, config_file)

    default_save_root = f"{default_data_root}/{data_name[1]}/results_kpreid"

    # Evaluation
    run_process(
        config_file = default_config_path,
          ckpt_path = default_ckpt_path,
          save_root = default_save_root,
          data_root = default_data_root,
          data_name = data_name[0],
             job_id = job_id,
    )

    # Inference
