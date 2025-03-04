import os
import torch as th
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


class PreprocessedFrameExtractor():
    def __init__(self, size=224):
        self.size = size
        self.transform = self._transform(self.size)

    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])   
    
    def get_frames_data(self, frames_path):
        frame_files = sorted([os.path.join(frames_path, fn) for fn in os.listdir(frames_path) if fn.endswith('.jpg')])

        frames_data = []
        for frame_file in frame_files:
            frame = Image.open(frame_file).convert("RGB")  # convert to RGB
            frame = self.transform(frame)
            frames_data.append(frame)

        if len(frames_data) > 0:
            frames_data = th.tensor(np.stack(frames_data))  # use np.stack and torch.tensor
        else:
            print("frame error")
            frames_data = th.zeros(1)
        return {'frames': frames_data}
    
    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor
    
RawFrameExtractor = PreprocessedFrameExtractor



