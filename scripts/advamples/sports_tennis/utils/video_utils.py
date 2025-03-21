import os
import cv2


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def save_video(video_frames, video_path):
    frame_shape = (video_frames[0].shape[1], 
                   video_frames[0].shape[0])
    fourcc = get_video_fourcc(video_path)
    fourcc = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(video_path, fourcc, 24, frame_shape)
    for frame in video_frames:
        writer.write(frame)
    writer.release()


def get_video_fourcc(video_path):
    video_ext = os.path.splitext(video_path)[1].lower()
    if video_ext == '.mp4':
        return 'mp4v'
    elif video_ext == '.avi':
        return 'XVID'
    else:
        raise TypeError(f'{video_ext} is not supported!')
