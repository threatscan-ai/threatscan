import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from threat_scanner.tools.frame_preprocess import preprocess_frames

class ThreatDataset(Dataset):
    def __init__(self, directories, labels, processor, max_frames=16):
        self.directories = directories
        self.labels = labels
        self.processor = processor
        self.max_frames = max_frames
        self.data = []
        self._load_data()

    def _load_data(self):
        for directory, label in zip(self.directories, self.labels):
          for filename in os.listdir(directory):
            if filename.endswith(".mov") or filename.endswith(".avi") or filename.endswith(".mp4"):
                video_path = os.path.join(directory, filename)
                self.data.append((video_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) > self.max_frames:
            start_index = np.random.randint(0, len(frames) - self.max_frames)
            frames = frames[start_index:start_index + self.max_frames]

        frames = preprocess_frames(frames)
        inputs = self.processor(list(frames), return_tensors="pt")
        return inputs, torch.tensor(label)
