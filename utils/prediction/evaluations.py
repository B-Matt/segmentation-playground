import cv2
import string
import pathlib

from typing import List
from matplotlib import pyplot as plt

from utils.dataset import Dataset
from utils.rgb import rgb2mask

def visualize(save_path: pathlib.Path, prefix, **images):
    """
        Plot images in one row.
    """
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)

    if prefix is not None and save_path is not None:
        plt.savefig(f'{str(save_path.resolve())}/visualisation_{prefix}.png')
    else:
        plt.show()

def preload_image_data(data_dir: string, img_dir: string, is_mask: bool = False, patch_size: int = 256, dataset_list: string = 'test_dataset.txt'):
    """
        Loads all images from data_dir.
    """
    dataset_files: List = []
    dataset_file_names: List = []
    with open(pathlib.Path(data_dir, dataset_list), mode='r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            path = pathlib.Path(data_dir, img_dir, line.strip(), f'Image/{line.strip()}.png' if is_mask == False else f'Mask/0.png')

            # Load image
            img = cv2.imread(str(path))
            img = Dataset._resize_and_pad(img, (patch_size, patch_size), (0, 0, 0))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if is_mask:
                img = rgb2mask(img)

            dataset_files.append(img)
            dataset_file_names.append(line.strip())
    return dataset_files, dataset_file_names
