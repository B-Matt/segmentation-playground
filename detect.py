import os
import sys
import torch
import argparse

from pathlib import Path

from utils.general import *
from utils.rgb import mask2rgb
from utils.prediction.dataloaders import *
from utils.prediction.predict import Prediction
from utils.prediction.evaluations import visualize

# Current Paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Relative Path

# Functions
def run(model, patch_size, conf_thres, source, view_img, show_inf):

    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    params = {
        'model_name': model[0],
        'patch_width': patch_size[0],
        'patch_height': patch_size[0],
        'n_channels': 3,
        'n_classes': 3
    }
    # webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    # screenshot = source.lower().startswith('screen')

    if is_url or is_file:
        source = check_file(source)

    dataset = LoadImages(source, img_size=patch_size[0])
    predict = Prediction(params)
    predict.initialize()

    for path, img, img0, vid_cap, s in dataset:
        # Do the prediction
        predicted = predict.predict_image(img)

        visualize(
            save_path=None,
            prefix=None,
            image=img,
            predicted_mask=mask2rgb(predicted),
        )
    
    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', type=str, default=ROOT / 'checkpoint.pth.tar', help='model path or triton URL')
    parser.add_argument('--patch-size', nargs='+', type=int, default=640, help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--source', default=ROOT / 'data/test', help='file/dir')
    parser.add_argument('--view_img', action='store_true', help='Show results')
    parser.add_argument('--show-inf', action='store_true', help='Visualize inference')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))