import os
import sys
import time
import argparse

import ffmpeg
import subprocess

from pathlib import Path
from PIL import Image

from utils.general import *
from utils.rgb import mask2rgb, mask2bw
from utils.prediction.dataloaders import *
from utils.prediction.predict import Prediction
from utils.prediction.area import calc_area, calc_mean_area

# Current Paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Relative Path

# USAGE: python detect.py --model "checkpoints/giddy-leaf-375/checkpoint.pth.tar" --patch-size 864 --source "1512411018435.jpg" --view-img
# USAGE: python detect.py --model "checkpoints/giddy-leaf-375/best-checkpoint.pth.tar" --patch-size 864 --source "Fire in warehouse [10BabBYvjL8].mp4" --view-img
# USAGE: python detect.py --model "checkpoints/sleek-microwave-405/best-checkpoint.pth.tar" --patch-size 864 --source "davor2.mp4" --encoder resnext50_32x4d --view-img

# Functions
def run(model = "", patch_size = 640, conf_thres = 0.5, source = "", encoder = None, view_img = True, save_video = False):
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
    predict.initialize(encoder)
    ffmpeg_process = None

    try:
        for path, img, img0, vid_cap, s in dataset:
            # Do the prediction
            start_time = time.time()
            img = cv2.flip(img, 0) # TODO: Detekcija kada treba flipat (https://github.com/ultralytics/yolov5/blob/1ea901bd5257e8688a122a27afcb21d74b7c5fbc/utils/dataloaders.py#L40)
            predicted = predict.predict_image(img, min_proba=conf_thres)
            end_time = time.time() - start_time
        
            if view_img or save_video:
                mask = mask2rgb(predicted)
                bw_mask = mask2bw(predicted)
                mask_fire = cv2.inRange(predicted, 1, 1)

                pil_img = Image.fromarray(img)
                pil_mask = Image.fromarray(mask)
                alpha_mask = Image.fromarray(bw_mask).convert('L')
                pil_img.paste(pil_mask, (0, 0), alpha_mask)
                mean_area = calc_mean_area(mask_fire, 10.0)

                image = np.asarray(pil_img)
                image = cv2.putText(image, f'Inference Time: {(end_time * 1000):.2f}ms', (10, patch_size[0] - 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
                image = cv2.putText(image, f'Mean Area: {mean_area:.1f}px', (10, patch_size[0] - 15), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA)

                if save_video and ffmpeg_process is None:
                    print(path)
                    ffmpeg_args = (
                        ffmpeg
                        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(patch_size[0],  patch_size[0]))
                        .output(f'{path}_inf.mp4', pix_fmt='yuv420p')
                        .overwrite_output()
                        .compile()
                    )
                    ffmpeg_process = subprocess.Popen(ffmpeg_args, shell=True, stdin=subprocess.PIPE)
                
                if view_img:
                    cv2.imshow(path, image[:,:,::-1])
                    cv2.waitKey(0)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if dataset.video_flag[0] is True and save_video:
                    ffmpeg_process.stdin.write(
                        image
                        .astype(np.uint8)
                        .tobytes()
                    )

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
    
    if save_video:
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        ffmpeg_process = None

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', type=str, default=ROOT / 'checkpoint.pth.tar', help='model path or triton URL')
    parser.add_argument('--patch-size', nargs='+', type=int, default=640, help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--source', default=ROOT / 'data/test', help='file/dir')
    parser.add_argument('--encoder', default="", help='Backbone encoder')
    parser.add_argument('--view-img', action='store_true', help='Show results')
    parser.add_argument('--save-video', action='store_true', help='Save video results')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))