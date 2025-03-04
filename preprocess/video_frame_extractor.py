"""
https://github.com/ArrowLuo/CLIP4Clip/blob/master/dataloaders/rawvideo_util.py
"""

import torch as th
import os
import cv2
import glob
import numpy as np
import json
import traceback
import argparse
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

class RawVideoExtractorCV2():
    def __init__(self, size=224, framerate=-1):
        self.framerate = framerate
        self.size = size
        self.transform = self._transform(self.size)

    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
        ])

    def video_to_tensor(self, video_file, preprocess, sample_fp=0, start_time=None, end_time=None):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                and start_time > -1 and end_time > start_time
        assert sample_fp > -1

        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration

        if start_time is not None:
            start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

        interval = 1
        if sample_fp > 0:
            interval = fps // sample_fp
        else:
            sample_fp = fps
        if interval == 0: interval = 1

        inds = [ind for ind in np.arange(0, fps, interval)]
        assert len(inds) >= sample_fp
        inds = inds[:sample_fp]

        ret = True
        images = []

        for sec in np.arange(start_sec, end_sec + 1):
            if not ret: break
            sec_base = int(sec * fps)
            for ind in inds:
                cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base + ind)
                ret, frame = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(preprocess(Image.fromarray(frame_rgb)))

        cap.release()

        if len(images) > 0:
            video_data = np.stack(images)
        else:
            video_data = np.zeros(1)
        return {'video': video_data}

    def get_video_data(self, video_path, start_time=None, end_time=None):
        image_input = self.video_to_tensor(video_path, self.transform, sample_fp=self.framerate, start_time=start_time, end_time=end_time)
        return image_input

    def process_frame_order(self, raw_video_data, frame_order=0):
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.shape[0] - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.shape[0])
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data

class RawVideoFrameSaverCV2(RawVideoExtractorCV2):
    def save_frames_as_jpg(self, video_file, output_dir, sample_fp=1, start_time=None, end_time=None, max_frames=32, slice_framepos=2):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        video_data = self.video_to_tensor(video_file, self.transform, sample_fp=self.framerate, start_time=start_time, end_time=end_time)
        video_data = self.process_frame_order(video_data['video'], frame_order=0)

        frame_indices = list(range(video_data.shape[0]))
        if max_frames < video_data.shape[0]:
            if slice_framepos == 0:
                video_data = video_data[:max_frames]
                frame_indices = frame_indices[:max_frames]
            elif slice_framepos == 1:
                video_data = video_data[-max_frames:]
                frame_indices = frame_indices[-max_frames:]
            else:
                sample_indx = np.linspace(0, video_data.shape[0] - 1, num=max_frames, dtype=int)
                video_data = video_data[sample_indx]
                frame_indices = [frame_indices[i] for i in sample_indx]

        for i, frame in enumerate(video_data):
            cv2.imwrite(os.path.join(output_dir, f'frame_{i:03d}.jpg'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        return frame_indices

def main():
    parser = argparse.ArgumentParser(description='Video Frame Extractor')
    parser.add_argument('--raw_video_path', type=str, required=True, help='Path to the raw video files (e.g., *.mp4)')
    parser.add_argument('--extracted_frame_path', type=str, required=True, help='Directory to save extracted frames')

    args = parser.parse_args()

    video_saver = RawVideoFrameSaverCV2(framerate=1)

    video_files = glob.glob(args.raw_video_path, "*")
    base_dir = args.extract_frame_path

    for i, video_file in enumerate(video_files):
        print(f'{i}: {video_file}')
        video_name = os.path.basename(video_file).split('.')[0]
        output_dir = os.path.join(base_dir, video_name)
        try:
            video_saver.save_frames_as_jpg(video_file, output_dir, max_frames=32)
        except Exception as e:
            print(f"Error processing file {video_file}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
