import torch
import numpy as np

import torchvision.transforms as transforms
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
        self.transform = transforms.ToTensor()

    def initialize(self, encoder=None):
        # log.info(f'[PREDICTION]: Loading model {self.model_name}')
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")

        state_dict = torch.load(self.model_name)
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
            self.net = smp.Unet(encoder_name=(encoder if encoder else "resnet34"), encoder_weights='imagenet', decoder_use_batchnorm=True, in_channels=3, classes=self.n_classes)

        self.net.load_state_dict(state_dict['model_state'])
        self.net.to(self.device)
        self.net.eval()

    def predict_image(self, image, resize=False):
        # Resize image to preserve CUDA memory
        if resize:
            image = Dataset._resize_and_pad(image, (self.patch_w, self.patch_h), (0, 0, 0))

        # Convert numpy to torch tensor
        patch_tensor = self.transform(image).unsqueeze(0)
        patch_tensor = patch_tensor.to(device=self.device, dtype=torch.float32)

        # Do prediction
        with torch.no_grad():
            pred = self.net(patch_tensor)

        return pred.cpu().numpy()