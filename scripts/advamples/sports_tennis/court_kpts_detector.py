import cv2
import numpy as np

import torch
from torchvision import models
from torchvision import transforms


FONT = cv2.FONT_HERSHEY_SIMPLEX


class CourtKptDetector:

    def __init__(self, model_path: str):
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) 
        self.model.load_state_dict(
                        torch.load(model_path, map_location='cpu'))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        original_h, original_w = image.shape[:2]

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        keypoints = outputs.squeeze().cpu().numpy()
        keypoints[0::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0
        return keypoints

    def draw_keypoints_single(self, image, keypoints):
        # Plot keypoints on the image
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), FONT, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image
    
    def draw_keypoints(self, video_frames, keypoints):
        output_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints_single(frame, keypoints)
            output_frames.append(frame)
        return output_frames


if __name__ == "__main__":

    default_ckpt_court_path = "./checkpoints/resnet50_court_keypoints.pth"
    
    model = CourtKptDetector(default_ckpt_court_path)

