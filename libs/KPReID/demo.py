"""
OVERVIEW

This script demonstrates how to use the KPR model to extract features from images and keypoints prompts.

It uses images and keypoints from the "assets/demo/soccer_players" folder.
It uses Matplotlib to display the images, prompts and model outputs.

The figures are plotted or saved in the "assets/demo/results" folder depending on the following config:

⚠️ NOTE: Re-identification is known to be challenging in a cross domain settings. 
Since cross-domain ReID was not part of our study, I cannot guarantee KPR will work on your data.
Finally, we also plan to release a KPR model that is trained on several datasets at the same time, 
    hoping to improve its generalization capabilities (stay tuned).

------------------------------------
0) Install instructions :
Follow the instructions in the README.md file to setup your python environment
Download the model weights from the following link: 
    https://drive.google.com/file/d/1Np5wu3nQa_Fl_z7Zw2kchJNC8JZVwsh5/view?usp=sharing
Put the downloaded file under 
    '/path_to_working_dir/pretrained_models/kpr_occ_pt_IN_82.34_92.33_42323828.pth.tar'

1) Load the configuration
go inside 'configs/kpr/imagenet/kpr_occ_posetrack_test.yaml' and change the path in the 'load_weights' config if you saved the downloaded
weights in a different location
-> have a look at '/torchreid/scripts/default_config.py' for detailed comments about all available options

2) Initialize the feature extractor, which is a convenient wrapper around the KPR model 
that handles preprocessing the input images and prompts, and postprocessing the outputs.

3) Load our demo samples from the "assets/demo/soccer_players" folder. 
This folder contains 2 subfolders, containing images and keypoints for a group of soccer players. 
The keypoints are stored in JSON files with the same name as the corresponding image files.

4) Display all the samples in a grid. 
Keypoints prompts are displayed in the image as dots, with a color per body part, 
while pure red dots indicates negative keypoints.

5) Extract features for both groups of samples. 
KPRFeatureExtractor returns the updated samples list with 3 new values: 
    'embeddings', 'visibility_scores', and 'parts_masks' as numpy arrays. 
It also returns the raw batched torch tensors with embeddings, visibility scores, and parts masks, 
    for further processing if needed.

    keypoints are optional: 
        "keypoints_xyc" and "negative_kps" keys can be omitted from the samples if not available.
        "negative_kps" should contain an empty array if no negative keypoints are available.

    samples_grp_i is a list of dictionaries, each dictionary containing 3 field: 
    - "image",
    - "keypoints_xyc" (positive keypoints)
    - "negative_kps" (negative keypoints). 

    Both "keypoints_xyc" and "negative_kps" are optional and can be omitted if not available. 
    In this case, KPR will perform re-id based on the image only, without using keypoints prompts.

6) Call again the display function, this time with the updated samples, 
    to visualize the part attention maps output by the model.

7) Compute the distance matrix in appearances between the 1st and 2nd group of samples. 
    A distance close to 0 indicate a strong similarity (samples have likely the same identity) 
    A distance close to 1 indicate a strong difference (samples have likely different identities).

8) Display the resulting distance matrix. 
    ⛔ Visually inspecting the part attention masks (and the computed distances)
        does not always reflect the true overall performance of the model. 
    ✅ Refer to the evaluation code using standard metrics such as mAP and Rank-1 
        for a more accurate evaluation.
"""
import os
from pathlib import Path

import json
import cv2
import numpy as np
import torch

import sys
import inspect

current_dir = os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(current_dir)

from tools.feature_extractor import KPRFeatureExtractor
from torchreid.builder import build_config
from torchreid.metrics.distance import compute_distance_matrix_using_bp_features
from torchreid.utils.visualization.display_kpr_samples import display_kpr_reid_samples_grid, display_distance_matrix


DISPLAY_MODE = 'save'   # 'plot' or 'save'
IGNORE_KEYPOINTS = True

base_folder = Path('./libs/KPReID/samples/soccer_players')
result_path = base_folder / 'results_no_kpt'
if os.path.isdir(result_path) is False:
    os.makedirs(result_path)


def load_kpr_samples(images_folder, keypoints_folder):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]

    # Initialize an empty list to store the samples
    samples = []

    # Iterate over the image files and construct each sample dynamically
    for img_name in image_files:
        # Construct full paths
        img_path = os.path.join(images_folder, img_name)
        json_path = os.path.join(keypoints_folder, img_name.replace('.jpg', '.json'))

        # Load the image
        img = cv2.imread(img_path)

        # Load the keypoints from the JSON file
        with open(json_path, 'r') as json_file:
            keypoints_data = json.load(json_file)

        # Initialize lists to hold keypoints
        keypoints_xyc = []
        negative_kps = []

        # Process the keypoints data
        for entry in keypoints_data:
            if entry["is_target"]:
                keypoints_xyc.append(entry["keypoints"])
            else:
                negative_kps.append(entry["keypoints"])

        assert len(keypoints_xyc) == 1, "Only 1 target keypoint set is supported for now."

        # Convert lists to numpy arrays
        keypoints_xyc = np.array(keypoints_xyc[0])
        negative_kps = np.array(negative_kps)

        # Create the sample dictionary
        sample = {
                    "image": img,
            "keypoints_xyc": keypoints_xyc, # positive prompts indicating re-id target
             "negative_kps": negative_kps,  # negative keypoints indicating other pedestrians
        }

        if IGNORE_KEYPOINTS:
            del sample['keypoints_xyc']
            del sample['negative_kps']

        # Append the sample to the list
        samples.append(sample)
    return samples


def run_demo(args):

    # Step 1
    kpr_cfg = build_config(args=args, config_path=config_path)
    
    # Disable prompting during inference (keypoints prompts are ignored by KPR)
    if IGNORE_KEYPOINTS:
        kpr_cfg.model.promptable_trans.disable_inference_prompting = True

    # Step 2
    extractor = KPRFeatureExtractor(kpr_cfg)

    # Step 3
    group1_folder = base_folder / 'group1'
    group2_folder = base_folder / 'group2'

    samples_grp_1 = load_kpr_samples(group1_folder/'images', group1_folder/'keypoints')
    samples_grp_2 = load_kpr_samples(group2_folder/'images', group2_folder/'keypoints')

    # Step 4
    display_kpr_reid_samples_grid(samples_grp_1 + samples_grp_2, display_mode=DISPLAY_MODE)

    # Step 5
    samples_grp_1, embeddings_grp_1, visibility_scores_grp_1, parts_masks_grp_1 = extractor(samples_grp_1)
    samples_grp_2, embeddings_grp_2, visibility_scores_grp_2, parts_masks_grp_2 = extractor(samples_grp_2)

    # Step 6
    display_kpr_reid_samples_grid(
        samples_grp_1 + samples_grp_2, 
        display_mode = DISPLAY_MODE, 
        save_path = result_path / 'samples_grid.png',
    )

    # Step 7
    distance_matrix, \
    body_parts_distmat = compute_distance_matrix_using_bp_features(
                            embeddings_grp_1,
                            embeddings_grp_2,
                            visibility_scores_grp_1,
                            visibility_scores_grp_2,
                            use_gpu=False,
                            use_logger=False
    )
    # The above function returns distances within the [0, 2] range
    distances = distance_matrix.cpu().detach().numpy() / 2

    # Step 8
    display_distance_matrix(
        distances, 
        samples_grp_1, 
        samples_grp_2,
        display_mode = DISPLAY_MODE, 
        save_path = result_path / 'distance_matrix.png',
    )


if __name__ == "__main__":

    # Unit-test
    model = "solider"
    ckpt_file = "kpr_occPoseTrack_SOLIDER.pth.tar"
    config_file = "kpr_occ_posetrack_test.yaml"

    config_path = os.path.join(current_dir, 'configs', 'kpr', model, config_file)
    ckpt_path = f"./checkpoints/KPReID/{ckpt_file}"

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m','--ckpt-file', type=str, default=ckpt_path, help='path to model checkpoint file')
    parser.add_argument('-c','--config-file', type=str, default=config_path, help='path to config file')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help='Modify config options using the command-line')
    args = parser.parse_args()

    run_demo(args)
