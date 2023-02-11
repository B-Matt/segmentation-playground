import os
import torch
import string
import pathlib
import numpy as np

from PIL import Image
from typing import List
from pathlib import Path
from torch.utils.data import Dataset

from utils.general import data_info_tuple
from utils.dataset.diskcache import init_cache
from utils.dataset.dataset import Dataset, DatasetType, DatasetCacheType

mem = init_cache()

class BinaryDataset(Dataset):
    def __init__(self,
        data_dir: string,
        img_dir: string,
        images: List = None,
        cache_type: DatasetCacheType = DatasetCacheType.NONE,
        type: DatasetType = DatasetType.TRAIN,
        is_combined_data: bool = True,
        patch_size: int = 128,
        transform = None
    ) -> None:
        self.all_imgs = images
        self.is_searching_dirs = images == None and img_dir != None
        self.is_combined_data = is_combined_data
        self.patch_size = patch_size
        self.transform = transform
        self.cache_type = cache_type
        self.images_data = []
        
        self.img_tupels = []
        if self.is_searching_dirs:
            self.img_tupels = self.preload_image_data_dir(data_dir, img_dir, type)
        else:
            self.img_tupels = self.preload_image_data(data_dir)

        # if self.cache_type == DatasetCacheType.DISK:
        #     mem = init_cache()
        #     self.load_sample = mem.cache(self.load_sample)
    
    def preload_image_data_dir(self, data_dir: string, img_dir: string, type: DatasetType):
        dataset_files: List = []
        with open(pathlib.Path(data_dir, f'{type.value}.txt'), mode='r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                path = pathlib.Path(data_dir, img_dir, line.strip())
                data_info = data_info_tuple(
                    line.strip(),
                    pathlib.Path(path, 'Image'),
                    pathlib.Path(path, 'Mask')
                )
                dataset_files.append(data_info)
        return dataset_files

    @mem.memoize(typed=True)
    def load_sample(self, index):
        info = self.img_tupels[index]        
        input_image = np.array(Image.open(str(Path(info.image, os.listdir(info.image)[0]))).convert("RGB"))
        input_image = Dataset._resize_and_pad(input_image, (self.patch_size, self.patch_size), (0, 0, 0))

        input_mask = np.array(Image.open(str(Path(info.mask, '0.png'))).convert("L"))
        input_mask = Dataset._resize_and_pad(input_mask, (self.patch_size, self.patch_size), (0, 0, 0))

        # visualize(
        #     save_path=None,
        #     prefix=None,
        #     image=input_image,
        #     predicted_mask=input_mask,
        # )
        return input_image, input_mask

    def __len__(self):
        return len(self.img_tupels)
    
    def __getitem__(self, index: int):
        img, mask = self.load_sample(index)

        if self.transform is not None:
            augmentation = self.transform(image=img, mask=mask)
            temp_img = augmentation['image']
            temp_mask = augmentation['mask']

        # visualize(
        #     save_path=None,
        #     prefix=None,
        #     image=temp_img.permute(1, 2, 0),
        #     predicted_mask=temp_mask,
        # )

        temp_mask = temp_mask / 255.0
        temp_mask = temp_mask.unsqueeze(0)

        return {
            'image': torch.as_tensor(temp_img).float(),
            'mask': temp_mask
        }