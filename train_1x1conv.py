import os
import cv2
import sys
import tqdm
import torch
import wandb
import pathlib
import datetime
import argparse

import numpy as np

from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch.utils.meter as meter

from models.mcdcnn import MCDCNN
from utils.early_stopping import YOLOEarlyStopping

# Logging
from utils.logging import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Best Models
models_data = [
    { 
        'resolution': 256,
        'batch_size': 8,
        'models': [
            'Unet 256x256-ResNxt50',
            'Unet++ 256x256-EffB7',
            'MAnet 256x256-EffB7',
            'FPN 256x256-ResNxt50',
            # 'DeepLabV3+ 256x256-EffB7',
        ]
    },
    {
        'resolution': 640,
        'batch_size': 2,
        'models': [
            'Unet 640x640-ResNxt50',
            'Unet++ 640x640-ResNxt50',
            'MAnet 640x640-EffB7',
            'FPN 640x640-ResNxt50',
            # 'DeepLabV3+ 640x640-EffB7',
        ]
    },
    {
        'resolution': 800,
        'batch_size': 2,
        'models': [
            'Unet 800x800-ResNxt50',
            'Unet++ 800x800-EffB7',
            'MAnet 800x800-EffB7',
            'FPN 800x800-ResNxt50',
            # 'DeepLabV3+ 800x800-ResNxt50',
        ]
    },
]

# Dataset
class BinaryImageDataset(Dataset):
    def __init__(self, model_paths, resolution, dataset_type, transform=None):
        self.model_paths = model_paths
        self.resolution = resolution
        self.dataset_type = dataset_type
        self.transform = transform

        imgs_path = pathlib.Path('playground', 'preped_data', model_paths[0], self.dataset_type)
        self.samples = os.listdir(imgs_path)
        self.samples = sorted(self.samples, key=lambda f: int(os.path.splitext(f)[0]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        images = []

        for model in self.model_paths:
            img_path = pathlib.Path('playground', 'preped_data', model, self.dataset_type, f'{idx}.png')
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if self.transform:
                img = self.transform(img)

            img.unsqueeze(0)
            images.append(img.float())

        mask = self.load_gt_mask(idx)
        if self.transform:
            mask = self.transform(mask)

        mask.unsqueeze(0)
        return images, torch.as_tensor(np.concatenate(images, axis=0)), torch.as_tensor(mask, dtype=torch.float32)
    
    def load_gt_mask(self, idx):
        path = pathlib.Path('playground', 'ground_truth_masks', str(self.resolution), self.dataset_type, f'{idx}.png')
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.inRange(img, (139, 189, 7), (139, 189, 7))
        return img
    
# Define the transformations to be applied to each image
img_transforms = transforms.Compose([
    transforms.ToTensor()
])

# Functions 
def save_checkpoint(model, optimizer, epoch: int, run_name, is_best: bool = False):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    state = {
        'time': str(datetime.datetime.now()),
        'model_state': model.state_dict(),
        'model_name': type(model).__name__,
        'optimizer_state': optimizer.state_dict(),
        'optimizer_name': type(optimizer).__name__,
        'epoch': epoch
    }

    if is_best is False:
        # log.info('[SAVING MODEL]: Model checkpoint saved!')
        torch.save(state, pathlib.Path('checkpoints', '1x1_conv', run_name, 'checkpoint.pth.tar'))

    if is_best:
        log.info('[SAVING MODEL]: Saving checkpoint of best model!')
        torch.save(state, pathlib.Path('checkpoints', '1x1_conv', run_name, 'best-checkpoint.pth.tar'))

def validate(net, dataloader, device, epoch, wandb_log):
    net.eval()
    criterion = torch.nn.MSELoss()
    criterion = criterion.to(device=device)
    loss_meter = meter.AverageValueMeter()

    for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc='Validation', position=1, unit='batch', leave=False):
        _, batch_imgs, batch_mask = batch
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        batch_mask = batch_mask.to(device, non_blocking=True)

        with torch.no_grad():
            mask_pred = net(batch_imgs)
            loss = criterion(mask_pred, batch_mask)
            loss_meter.add(loss.item())

    net.train()
    return loss_meter.mean

def train(model_idx, epochs, cool_down_epochs, learning_rate, weight_decay, adam_eps, dropout):
    # Training vars
    batch_size = models_data[model_idx]['batch_size'] #4
    patch_size = models_data[model_idx]['resolution']
    is_saving_checkpoints = True

    # Model and device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MCDCNN(dropout, input_channels=4).to(device)

    # Dataloaders
    train_dataset = BinaryImageDataset(models_data[model_idx]['models'], patch_size, 'training', img_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    val_dataset = BinaryImageDataset(models_data[model_idx]['models'], patch_size, 'validation', img_transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    # Optimizers and Schedulers
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay, eps=adam_eps, lr=learning_rate)
    early_stopping = YOLOEarlyStopping(patience=20)
    loss_meter = meter.AverageValueMeter()

    log.info(f'''[TRAINING]:
        Model:           {model.__class__.__name__}
        Resolution:      {patch_size}x{patch_size}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {int(len(train_dataset))}
        Validation size: {int(len(val_dataset))}
        Device:          {device.type}
    ''')

    wandb_log = wandb.init(project='semantic-article', entity='firebot031')
    wandb_log.config.update(
        dict(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patch_size=models_data[model_idx]['resolution'],
            model=model.__class__.__name__,
        )
    )
    
    run_name = wandb.run.name if wandb.run.name is not None else f'{model.__class__.__name__}-{batch_size}-{patch_size}'
    save_path = pathlib.Path('checkpoints', '1x1_conv', run_name)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    criterion = torch.nn.MSELoss()
    last_best_score = float('inf')

    torch.cuda.empty_cache()
    for epoch in range(epochs):
        val_loss = 0.0
        # progress_bar = tqdm.tqdm(total=int(len(train_dataset)), desc=f'Epoch {epoch + 1}/{epochs}', unit='img', position=0)
        # inputMasks = None

        for batch in tqdm.tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', position=1, unit='img', leave=True):
            optimizer.zero_grad(set_to_none=True)

            # Get Batch Of Images
            inputMasks, batch_imgs, batch_mask = batch
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)

            # Prediction
            masks_pred = model(batch_imgs)
            loss = criterion(masks_pred, batch_mask)

            # Statistics
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())

        # Evaluation of training
        val_loss = validate(model, val_loader, device, epoch, wandb_log)

        if epoch >= cool_down_epochs:
            early_stopping(epoch, val_loss)

        if val_loss < last_best_score and is_saving_checkpoints and epoch >= cool_down_epochs:
            save_checkpoint(model, optimizer, epoch, run_name, True)
            last_best_score = val_loss

        # Update WANDB with Images
        try:
            # if epoch >= 1:
            #     # test = torch.round(test)
            #     pred_img = masks_pred.squeeze(0).permute(1, 2, 0).detach().cpu().float().numpy() * 255.0 #torch.sigmoid(masks_pred.squeeze(0).permute(1, 2, 0).detach().cpu().float()).numpy() * 255.0
            #     gt_img = batch_mask.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0

            #     visualize(
            #         save_path=None,
            #         prefix=None,
            #         # input_mask1=inputMasks[1].squeeze(0).permute(1, 2, 0).detach().cpu().float().numpy() * 255.0,
            #         # input_mask2=inputMasks[2].squeeze(0).permute(1, 2, 0).detach().cpu().float().numpy() * 255.0,
            #         # input_mask3=inputMasks[3].squeeze(0).permute(1, 2, 0).detach().cpu().float().numpy() * 255.0,
            #         # input_mask4=inputMasks[4].squeeze(0).permute(1, 2, 0).detach().cpu().float().numpy() * 255.0,
            #         gt_img=gt_img,
            #         pred_img=pred_img,
            #     )

            wandb_log.log({
                'Images [training]': {                    
                    'Input Mask 1': wandb.Image(inputMasks[0].squeeze(0).permute(1, 2, 0).detach().cpu().float().numpy() * 255.0),
                    'Input Mask 2': wandb.Image(inputMasks[1].squeeze(0).permute(1, 2, 0).detach().cpu().float().numpy() * 255.0),
                    'Input Mask 3': wandb.Image(inputMasks[2].squeeze(0).permute(1, 2, 0).detach().cpu().float().numpy() * 255.0),
                    'Input Mask 4': wandb.Image(inputMasks[3].squeeze(0).permute(1, 2, 0).detach().cpu().float().numpy() * 255.0),
                    'Ground Truth': wandb.Image(batch_mask.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0),
                    'Prediction': wandb.Image(masks_pred.squeeze(0).permute(1, 2, 0).detach().cpu().float().numpy() * 255.0),
                },
            }, step=epoch)
        except Exception as e:
            print('Exception', e)

        # Update WANDB
        wandb_log.log({
            'Learning Rate': optimizer.param_groups[0]['lr'],
            'Epoch': epoch,
            'Loss [training]': loss_meter.mean,
            'Loss [validation]': val_loss,
        }, step=epoch)

        # Saving last model
        if is_saving_checkpoints:
            save_checkpoint(model, optimizer, epoch, run_name, False)

        # Early Stopping
        if early_stopping.early_stop:
            save_checkpoint(model, optimizer, epoch, run_name, False)
            log.info(
                f'[TRAINING]: Early stopping training at epoch {epoch}!')
            break

    # Push average training metrics
    wandb_log.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-idx', type=int, default=0, help='Which model you want to train?')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--adam-eps', nargs='+', type=float, default=1e-3, help='Adam epsilon')
    parser.add_argument('--weight-decay', type=float, default=1e-7, help='Weight decay that is used for AdamW')
    parser.add_argument('--cool-down-epochs', type=int, default=50, help='Cool down epochs')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs')
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()

    # Start Training
    try:
        # for i in range(len(models_data)):
        train(args.model_idx, args.epochs, args.cool_down_epochs, args.learning_rate, args.weight_decay, args.adam_eps, args.dropout)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    # End Training
    logging.info('[TRAINING]: Training finished!')
    torch.cuda.empty_cache()

    net = None
    training = None

    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
