import collections
import pathlib
import cv2

import pandas as pd

def count_resolutions_export(dataset_name, data_dir, img_dir):
    dataset_files = {}
    with open(pathlib.Path(data_dir, f'{dataset_name}.txt'), mode='r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            path = pathlib.Path(data_dir, img_dir, line.strip(), f'Image/{line.strip()}.png')
            img = cv2.imread(str(path))

            if not img.shape[0] in dataset_files:
                dataset_files.update({img.shape[0] : 0})
                
            dataset_files[img.shape[0]] += 1

    df = pd.DataFrame(data=collections.OrderedDict(sorted(dataset_files.items())), index=[1])
    df.to_excel(f'{dataset_name}_resolutions.xlsx', index=False)

count_resolutions_export('training_dataset', r'data', r'imgs')
count_resolutions_export('validation_dataset', r'data', r'imgs')
count_resolutions_export('test_dataset', r'data', r'imgs')