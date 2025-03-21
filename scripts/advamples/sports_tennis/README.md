
# Tennis Analysis

## Introduction
This project analyzes Tennis players in a video to measure:
- player speed, 
- ball shot speed, and
- number of shots. 

This project will detect players and the tennis ball using YOLO and also utilizes CNNs to extract court keypoints. This hands-on project is perfect for polishing your machine learning, and computer vision skills. 

## Output
Here is a screenshot from output:

![Screenshot](./assets/output.jpeg)

## Models
* (Pretrained) `YOLO-v8` for player detection
* (Fine-tuned) `YOLO-v5` for tennis ball detection | [Checkpoint](https://drive.google.com/file/d/1UZwiG1jkWgce9lNhxJ2L0NVjX1vGM05U/view?usp=sharing) | [Notebook](./notebooks/tennis_ball_detector_training.ipynb)
* (Fine-tuned) `ResNet-50` for Court Keypoint detection | [Checkpoint](https://drive.google.com/file/d/1QrTOF1ToQ4plsSZbkBs3zOLkVt3MBlta/view?usp=sharing) | [Notebook](./notebooks/tennis_court_keypoints_training.ipynb)

## Requirements
* pytorch
* ultralytics
* numpy 
* opencv
* pandas
