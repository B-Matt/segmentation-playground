"""
    Based on YOLOv5 dataloaders.py script.
    URL: https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py
"""

import os
import cv2
import glob
import pathlib

import numpy as np

from utils.dataset import Dataset

# Logging
from utils.logging import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Constants
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'

# Image Loader
class LoadImages:
    def __init__(self, paths, img_size=256, transforms=None):
        if isinstance(paths, str) and pathlib.Path(paths).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            paths = pathlib.Path(paths).read_text().rsplit()

        files = []
        for path in sorted(paths) if isinstance(paths, (list, tuple)) else [paths]:
            path = pathlib.Path(path).resolve()
            path = fr'{str(path)}'

            if '*' in path:
                files.extend(sorted(glob.glob(path, recursive=True)))                                   # glob
            elif os.path.isdir(path):
                files.extend(sorted(glob.glob(os.path.join(path, '*.*'))))                              # Dir
            elif os.path.isfile(path):
                files.append(path)                                                                      # Files
            else:
                raise FileNotFoundError(f'{path} does not exist')

        images = [i for i in files if i.split('.')[-1].lower() in IMG_FORMATS]
        videos = [v for v in files if v.split('.')[-1].lower() in VID_FORMATS]
        num_images, num_videos = len(images), len(videos)
        
        self.img_size = img_size
        self.all_files = images + videos
        self.num_files = num_images + num_videos
        self.video_flag = [False] * num_images + [True] * num_videos
        self.transforms = transforms

        if any(videos):
            self._new_video(videos[0])
        else:
            self.cap = None

        if self.num_files == 0:
            log.error(f'No images nor videos found in {paths}')

    def __iter__(self):
        self.count = 0
        return self
    
    def skip_file(self):
        if self.count == self.num_files:
            raise StopIteration

        self.count += 1

    def __next__(self):
        if self.count == self.num_files:
            raise StopIteration

        path = self.all_files[self.count]
        string = ''

        if self.video_flag[self.count]:                                                                     # Read video
            self.cap.grab()
            ret_val, img_0 = self.cap.retrieve()

            while not ret_val:
                self.count += 1
                self.cap.release()

                if self.count == self.num_files:
                    raise StopIteration
                
                path = self.all_files[self.count]
                self._new_video(path)
                ret_value, img_0 = self.cap.read()

            img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
            img_0 = Dataset._resize_and_pad(img_0, (self.img_size, self.img_size), (0, 0, 0))

            self.frame += 1
            string = f'video {self.count + 1}/{self.num_files} ({self.frame}/{self.frames}) {path}: '
        else:                                                                                               # Read image
            self.count += 1
            img_0 = cv2.imread(path)
            img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
            img_0 = Dataset._resize_and_pad(img_0, (self.img_size, self.img_size), (0, 0, 0))
            string = f'image {self.count}/{self.num_files} {path}: '

        img = np.ascontiguousarray(img_0)
        if self.transforms:
            img = self.transforms(img)

        return path, img, img_0, self.cap, string

    def __len__(self):
        return self.num_files

    def _new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))