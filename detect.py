import os
import sys
import torch
import argparse

from pathlib import Path

from utils.prediction.dataloaders import *

# Current Paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Relative Path

# Functions
def run():
    pass

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=ROOT / 'data/test', help='file/dir')
    parser.add_argument('--view-img', action='store_true', help='Show results')
    parser.add_argument('--visualize', action='store_true', help='Visualize inference')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))