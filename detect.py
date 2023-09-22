import os
import re
import cv2
import sys
import time
import ffmpeg
import argparse
import subprocess

import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path


from utils.general import *
from utils.rgb import mask2rgb, mask2bw, colorize_mask
from utils.prediction.dataloaders import *
from utils.prediction.predict import Prediction
from utils.prediction.area import calc_mean_area
# from utils.prediction.tensorrt import TensorRTEngineInference

# Current Paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Add ROOT to PATHli-ion
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Relative Path

# USAGE: python detect.py --model "checkpoints/spring-deluge-294/best-checkpoint.pth.tar" --patch-size 800 --conf-thres 0.5 --encoder "resnext50_32x4d" --source "1512411018435.jpg" --view-img
# USAGE: python detect.py --model "checkpoints/spring-deluge-294/best-checkpoint.pth.tar" --patch-size 800 --conf-thres 0.5 --encoder "resnext50_32x4d" --source "Fire in warehouse [10BabBYvjL8].mp4" --view-plots --max-frames 500
# USAGE: python detect.py --model "checkpoints/matej-data-sc/best-checkpoint.pth.tar" --patch-size 640 --conf-thres 0.5 --encoder "mit_b4" --source "playground/examples/carton-boxes.mp4" "playground/examples/christmas-tree.mp4" "playground/examples/li-ion.mp4" "playground/examples/christmas-tree.mp4" "playground/examples/wood.mp4" "playground/examples/paper-standard.mp4" --view-plots --max-frames 300
# python detect.py --model "checkpoints/matej-data-sc/best-checkpoint.pth.tar" --patch-size 640 --conf-thres 0.5 --encoder "mit_b4" --source 0 --view-img
# python detect.py --model "checkpoints/matej-data-sc/best-checkpoint.pth.tar" --patch-size 640 --conf-thres 0.5 --encoder "mit_b4" --source rtsp://192.168.0.4/defaultPrimary1?streamType=u --view-img

# python detect.py --model "checkpoints/spring-deluge-294/best-checkpoint.pth.tar" --patch-size 800 --conf-thres 0.5 --encoder "resnext50_32x4d" --source "playground/examples/wood.mp4"
# python detect.py --model "checkpoints/spring-deluge-294/best-checkpoint.pth.tar" --patch-size 800 --conf-thres 0.5 --encoder "resnext50_32x4d" --source "playground/examples/paper-open-array.mp4" --max-frames 3000

# Functions
def prepare_mask_data(img: np.array, pred: np.array, classes: int = 1):
    if classes > 1:
        mask = mask2rgb(pred)
        bw_mask = mask2bw(pred)
        mask_fire = cv2.inRange(pred, 1, 1)

        pil_img = Image.fromarray(img)
        pil_mask = Image.fromarray(mask)
        alpha_mask = Image.fromarray(bw_mask).convert('L')
        pil_img.paste(pil_mask, (0, 0), alpha_mask)
        num_areas, mean_area = calc_mean_area(mask_fire, 10.0)
    else:
        mask = pred * 255.0
        mask_fire = cv2.inRange(mask, 254, 255)

        mask = cv2.cvtColor(mask.astype(np.float32), cv2.COLOR_GRAY2RGBA)
        mask = colorize_mask(mask, (0, 0, 255, 255)).astype(np.uint8)

        pil_img = Image.fromarray(img)
        pil_mask = Image.fromarray(mask)
        alpha_mask = Image.fromarray(mask).convert('L')
        alpha_mask = alpha_mask.point(lambda p: 255 if p > 128 else 0)
        alpha_mask = alpha_mask.convert('1')
    
        pil_img.paste(pil_mask, (0, 0), alpha_mask)                
        num_areas, mean_area = calc_mean_area(mask_fire, 10.0)
    return pil_img, mean_area, num_areas

def plot_title(path):
    title_str = os.path.basename(path)
    title_str = ' '.join(title_str.split('-'))
    title_str = title_str.replace('.mp4', '')
    title_str = title_str.title()
    return title_str

def snake_case(s):
  return '_'.join(
    re.sub('([A-Z][a-z]+)', r' \1',
    re.sub('([A-Z]+)', r' \1',
    s.replace('-', ' '))).split()).lower()

def run(model: str = "", patch_size: int = 640, classes: int = 1, conf_thres: float = 0.5, source: str = "", encoder: str = None, max_frames: int = None, view_img: bool = True, save_video: bool = False):
    if not isinstance(source, (list, tuple)) or (isinstance(source, list) and len(source) == 1):
        if isinstance(source, list) and len(source) == 1:
            source = str(source[0])
        else:
            source = str(source)

        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

        if is_url or is_file:
            source = check_file(source)

    ffmpeg_process = None
    is_stream = not isinstance(source, list) and (source.isnumeric() or source.endswith('.streams') or (is_url and not is_file))

    # Predictions and loaders
    if is_stream:
        dataset = LoadStreams(source, img_size=patch_size[0])
    else:
        dataset = LoadImages(source, img_size=patch_size[0])

    # Get prediction engine & warmup
    predict = Prediction(model, encoder, (patch_size[0], patch_size[0]), 3, classes)
    predict.warmup(10)       

    # Frame data statistics
    frame_count = 0
    plot_areas = [0]
    plot_frames = [0]
    max_frame_areas = 0

    try:
        for path, img, img0, vid_cap, current_frame, total_frames in dataset:
            # Do the prediction
            predicted, inference_time = predict.predict_image(img, conf_thres, False)

            if view_img or save_video:
                pil_img, mean_area, frame_areas = prepare_mask_data(img0, predicted, classes)

                if frame_areas > max_frame_areas:
                    max_frame_areas = frame_areas

                image = np.asarray(pil_img)
                image = cv2.putText(image, f'Inference Time: {(inference_time * 1000):.2f}ms', (10, patch_size[0] - 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
                image = cv2.putText(image, f'Mean Area: {mean_area:.1f}px', (10, patch_size[0] - 15), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA)

                if save_video and ffmpeg_process is None:
                    ffmpeg_args = (
                        ffmpeg
                        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(patch_size[0],  patch_size[0]))
                        .filter('fps', fps=25, round='up')
                        .output(f'{path}_inf.mp4', pix_fmt='yuv420p')
                        .overwrite_output()
                        .compile()
                    )
                    ffmpeg_process = subprocess.Popen(ffmpeg_args, shell=True, stdin=subprocess.PIPE)

                if view_img:
                    if is_stream:
                        cv2.imshow(path, image[:,:,::-1])
                    else:
                        cv2.imshow(path, image[:,:,::-1])
                        cv2.waitKey(0)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                if save_video and (type(dataset) == LoadStreams or dataset.video_flag[0] is True):
                    ffmpeg_process.stdin.write(
                        image
                        .astype(np.uint8)
                        .tobytes()
                    )

            frame_count += 1
            plot_areas.append(mean_area)
            plot_frames.append(frame_count)

            if max_frames is not None and (frame_count >= max_frames or current_frame >= total_frames):
                max_frame_areas = 0
                dataset.skip_file()
                frame_count = 0

    except KeyboardInterrupt:
        cv2.destroyAllWindows()

    if save_video:
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        ffmpeg_process = None

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=ROOT / 'checkpoint.pth.tar', help='Model path or triton URL')
    parser.add_argument('--patch-size', nargs='+', type=int, default=640, help='Inference size h,w')
    parser.add_argument('--classes', type=int, default=1, help='Classes number')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--source', nargs='+', default=ROOT / 'data/test', help='File/dir')
    parser.add_argument('--encoder', default="", help='Backbone encoder')
    parser.add_argument('--max-frames', type=int, default=None, help='Max. number of processed frames')
    parser.add_argument('--view-img', action='store_true', help='Show results')
    parser.add_argument('--save-video', action='store_true', help='Save video results')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))