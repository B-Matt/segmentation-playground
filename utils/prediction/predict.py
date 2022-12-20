import torch
import numpy as np

import segmentation_models_pytorch as smp

from utils.dataset import Dataset
from unet.model import UNet

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


    def initialize(self):
        log.info(f'[PREDICTION]: Loading model {self.model_name}')
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")

        state_dict = torch.load(self.model_name)
        if state_dict['model_name'] == 'UnetPlusPlus':
            self.net = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights="imagenet", decoder_use_batchnorm=True, in_channels=self.n_channels, classes=self.n_classes)
        else:
            self.net = UNet(n_channels=self.n_channels, n_classes=self.n_classes)

        self.net.load_state_dict(state_dict['model_state'])
        self.net.to(self.device)
        self.net.eval()


    def predict_proba(self, image: np.array, resize: bool = False, min_proba=None):
        if resize:
            # Resize image to preserve CUDA memory
            image = Dataset._resize_and_pad(image, (self.patch_w, self.patch_h), (0, 0, 0))

        # Convert numpy to torch tensor
        if len(image.shape) == 2:
            patch_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        if len(image.shape) == 3:
            patch_tensor = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float()

        patch_tensor = patch_tensor.to(device=self.device, dtype=torch.float32)

        # Do prediction
        with torch.no_grad():
            pred = self.net(patch_tensor)
            prob_predict = torch.sigmoid(pred).squeeze().detach().cpu().numpy()

            if min_proba is not None:
                prob_predict = prob_predict >= min_proba
            prob_predict = np.transpose(prob_predict, (1, 2, 0))
        return prob_predict

    def predict_image(self, image, resize=False, min_proba=None):        
        prob_predict = self.predict_proba(image, resize, min_proba)
        mask = np.argmax(prob_predict, axis=2)
        return mask