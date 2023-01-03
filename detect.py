import os
import sys
import argparse

from pathlib import Path
from PIL import Image

from utils.general import *
from utils.rgb import mask2rgb, mask2bw
from utils.prediction.dataloaders import *
from utils.prediction.predict import Prediction
from utils.ui.gui import VideoWindow, ImageWindow

# Current Paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Relative Path

# USAGE: python detect.py --model "checkpoints/giddy-leaf-375/checkpoint.pth.tar" --patch-size 960 --source "1512411018435.jpg" --view-img
# USAGE: python detect.py --model "checkpoints/giddy-leaf-375/best-checkpoint.pth.tar" --patch-size 960 --source "fbmsugz5y17a1.webp" --view-img
# USAGE: python detect.py --model "checkpoints/giddy-leaf-375/best-checkpoint.pth.tar" --patch-size 960 --source "Fire in warehouse [10BabBYvjL8].mp4" --view-img

# Functions
def run(model = "", patch_size = 640, conf_thres = 0.5, source = "", view_img = True):
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
    detect_window = None
    is_video = False

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (patch_size[0],  patch_size[0]))

    for path, img, img0, vid_cap, s in dataset:
        # Do the prediction
        predicted = predict.predict_image(img, min_proba=None)
    
        if view_img:            
            mask = mask2rgb(predicted)
            pil_img = Image.fromarray(img)
            pil_mask = Image.fromarray(mask)
            alpha_mask = Image.fromarray(mask2bw(predicted)).convert('L')
            pil_img.paste(pil_mask, (0, 0), alpha_mask)

            if dataset.video_flag[0] is False:
                detect_window = ImageWindow(size=(patch_size[0],  patch_size[0]))
                detect_window.title(path)
                # img_tk = img_tk._PhotoImage__photo.zoom(2) # Zoomiranje slike
                detect_window.setup_image(pil_img)
            else:
                is_video = True
                out.write(np.asarray(pil_img)[:,:,::-1])

    
    if view_img:
        if is_video == True:
            detect_window = VideoWindow(size=(patch_size[0],  patch_size[0]))
            out.release()
        detect_window.mainloop()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', type=str, default=ROOT / 'checkpoint.pth.tar', help='model path or triton URL')
    parser.add_argument('--patch-size', nargs='+', type=int, default=640, help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--source', default=ROOT / 'data/test', help='file/dir')
    parser.add_argument('--view-img', action='store_true', help='Show results')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))