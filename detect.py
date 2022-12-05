import os
import sys
import torch
import argparse

from pathlib import Path

from utils.general import *
from utils.prediction.dataloaders import *

# Current Paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Relative Path

# Functions
def run(weights, inf_size, conf_thres, source, view_img, visualize):

    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')

    if is_url and is_file:
        source = check_file(source)
    
    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'checkpoint.pth.tar', help='model path or triton URL')
    parser.add_argument('--inf-size', nargs='+', type=int, default=[640, 640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--source', default=ROOT / 'data/test', help='file/dir')
    parser.add_argument('--view_img', action='store_true', help='Show results')
    parser.add_argument('--visualize', action='store_true', help='Visualize inference')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))