import time
import torch
import pathlib

import numpy as np
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp

from utils.dataset import Dataset
from models.mcdcnn import FFMMNet1

# Logging
from utils.logging import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Functions
class Prediction:
    def __init__(self, params) -> None:
        self.model_name = params['model_name']
        self.patch_w = params['patch_width']
        self.patch_h = params['patch_height']
        self.n_channels = params['n_channels']
        self.n_classes = params['n_classes']

    def initialize(self, encoder=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        log.info(f'[PREDICTION]: Loading model {self.model_name} ({encoder})')
        torch.cuda.empty_cache()
        model_path = pathlib.Path(self.model_name).resolve()
        model_path = fr'{str(model_path)}'

        state_dict = torch.load(model_path)
        if state_dict['model_name'] == 'UnetPlusPlus':
            self.net = smp.UnetPlusPlus(encoder_name=(encoder if encoder else "resnet34"), encoder_weights="imagenet", decoder_use_batchnorm=True, in_channels=self.n_channels, classes=self.n_classes)
        elif state_dict['model_name'] == 'MAnet':
            self.net = smp.MAnet(encoder_name=(encoder if encoder else "resnet34"), encoder_depth=5, encoder_weights='imagenet', decoder_use_batchnorm=True, in_channels=self.n_channels, classes=self.n_classes)
        elif state_dict['model_name'] == 'Linknet':
            self.net = smp.Linknet(encoder_name=(encoder if encoder else "resnet34"), encoder_depth=5, encoder_weights="imagenet", decoder_use_batchnorm=True, in_channels=self.n_channels, classes=self.n_classes)
        elif state_dict['model_name'] == 'FPN':
            self.net = smp.FPN(encoder_name=(encoder if encoder else "resnet34"), encoder_weights="imagenet", in_channels=self.n_channels, classes=self.n_classes)
        elif state_dict['model_name'] == 'PSPNet':
            self.net = smp.PSPNet(encoder_name=(encoder if encoder else "resnet34"), encoder_weights="imagenet", in_channels=self.n_channels, classes=self.n_classes)
        elif state_dict['model_name'] == 'PAN':
            self.net = smp.PAN(encoder_name=(encoder if encoder else "resnet34"), encoder_weights="imagenet", in_channels=self.n_channels, classes=self.n_classes)
        elif state_dict['model_name'] == 'DeepLabV3':
            self.net = smp.DeepLabV3(encoder_name=(encoder if encoder else "resnet34"), encoder_weights="imagenet", in_channels=self.n_channels, classes=self.n_classes)
        elif state_dict['model_name'] == 'DeepLabV3Plus':
            self.net = smp.DeepLabV3Plus(encoder_name=(encoder if encoder else "resnet34"), encoder_weights="imagenet", in_channels=self.n_channels, classes=self.n_classes)
        else:
            self.net = smp.Unet(encoder_name=(encoder if encoder else "resnet34"), decoder_use_batchnorm=True, in_channels=3, classes=self.n_classes)

        self.net.load_state_dict(state_dict['model_state'])
        self.net.to(self.device)
        self.net.eval()

    def warmup(self, warmup_iters: int) -> None:
        log.info(f'[PREDICTION]: Model warm up for {warmup_iters} iteration/s.')
        dummy_input = torch.rand((1, 3, self.patch_w, self.patch_h))

        for i in range(1, warmup_iters):
            self.predict_image(dummy_input, None, False)


    def predict_image(self, image: np.array, threshold: float = 0.5, resize: bool = False) -> np.array:
        # Resize image to preserve CUDA memory
        if resize:
            image = Dataset._resize_and_pad(image, (self.patch_w, self.patch_h), (0, 0, 0))

        # Convert numpy to torch tensor
        if type(image) is np.ndarray:
            start_time = time.time()
            patch_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            end_time = time.time() - start_time
            log.info(f'[PREDICTION]: Inference patch preparation took {(end_time * 1000):.2f} ms.')

        # Do not convert if is warmup tensor
        if type(image) is torch.Tensor:
            patch_tensor = image

        patch_tensor = patch_tensor.to(self.device)

        # Do prediction
        start_time = time.time()
        with torch.autocast(device_type='cuda'):
            with torch.no_grad():
                model_logits = self.net(patch_tensor)
                mask = torch.sigmoid(model_logits) if threshold is None else torch.sigmoid(model_logits) > threshold
                mask = mask.squeeze(0).detach().cpu().numpy()

        end_time = time.time() - start_time
        log.info(f'[PREDICTION]: Inference prediction took {(end_time * 1000):.2f} ms.')
        return mask[0]