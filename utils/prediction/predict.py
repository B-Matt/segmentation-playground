import torch
import pathlib

import torchvision.transforms as transforms
import segmentation_models_pytorch as smp

from utils.dataset import Dataset
from models.mcdcnn import FFMMNet

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
        log.info(f'[PREDICTION]: Loading model {self.model_name} ({encoder})')
        model_path = pathlib.Path(self.model_name).resolve()
        model_path = fr'{str(model_path)}'
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")

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
        elif state_dict['model_name'] == 'FFMMNet':
            self.net = FFMMNet()
        else:
            self.net = smp.Unet(encoder_name=(encoder if encoder else "resnet34"), decoder_use_batchnorm=True, in_channels=3, classes=self.n_classes)

        self.net.load_state_dict(state_dict['model_state'])
        self.net.to(self.device)
        self.net.eval()

    def predict_image(self, image, threshold=0.5, resize=False):
        # Resize image to preserve CUDA memory
        if resize:
            image = Dataset._resize_and_pad(image, (self.patch_w, self.patch_h), (0, 0, 0))

        # Convert numpy to torch tensor
        patch_tensor = torch.cat(image, dim=0)
        print(patch_tensor.shape)
        # patch_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        # patch_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        patch_tensor = patch_tensor.to(self.device)

        # Do prediction
        with torch.no_grad():
            mask = torch.sigmoid(self.net(patch_tensor)) if threshold is None else torch.sigmoid(self.net(patch_tensor)) > threshold
            mask = mask.detach().cpu().numpy()
            # mask = mask.squeeze(0).detach().cpu().numpy()

        return mask[0]