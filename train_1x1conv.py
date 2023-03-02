import os
import cv2
import sys
import tqdm
import torch
import wandb
import pathlib
import datetime

import numpy as np
import torch.nn.functional as TF
import torchmetrics.functional as F

from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch.utils.meter as meter

from models.mcdcnn import MCDCNN
from utils.early_stopping import YOLOEarlyStopping
from utils.prediction.evaluations import visualize

# Logging
from utils.logging import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Best Models
models_data = [
    { 
        'resolution': 256,
        'models': [
            'Unet 256x256-ResNxt50',
            'Unet++ 256x256-EffB7',
            'MAnet 256x256-EffB7',
            'FPN 256x256-ResNxt50',
            'DeepLabV3+ 256x256-EffB7',
        ]
    },
    # {
    #     'resolution': 640,
    #     'models': [
    #         'Unet 640x640-ResNxt50',
    #         'Unet++ 640x640-ResNxt50',
    #         'MAnet 640x640-EffB7',
    #         'FPN 640x640-ResNxt50',
    #         'DeepLabV3+ 640x640-EffB7',
    #     ]
    # },
    # {
    #     'resolution': 800,
    #     'models': [
    #         'Unet 800x800-ResNxt50',
    #         'Unet++ 800x800-EffB7',
    #         'MAnet 800x800-EffB7',
    #         'FPN 800x800-ResNxt50',
    #         'DeepLabV3+ 800x800-ResNxt50',
    #     ]
    # },
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
            # img = Image.open(img_path).convert('1')
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = img / 255.0
            img = img.astype(np.float32)

            if self.transform:
                img = self.transform(img)
            images.append(img)

        mask = self.load_gt_mask(idx)
        if self.transform:
            mask = self.transform(mask)

        # mask = mask / 255.0
        mask.unsqueeze(0)
        return torch.cat(images, dim=0), torch.as_tensor(mask, dtype=torch.float32)
    
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
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion = criterion.to(device=device)

    metrics = [
        F.dice,
        F.classification.binary_jaccard_index,
    ]
    loss_meter = meter.AverageValueMeter()
    metrics_meters = { metric.__name__: meter.AverageValueMeter() for metric in metrics }

    for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc='Validation', position=1, unit='batch', leave=False):
        image = batch[0].to(device=device, non_blocking=True)
        mask_true = batch[1].to(device=device, non_blocking=True)

        with torch.no_grad():
            mask_pred = net(image)
            loss = criterion(mask_pred, mask_true)
            loss_meter.add(loss.item())

            for metric_fn in metrics:
                metric_value = metric_fn(mask_pred, mask_true.long(), ignore_index=0).cpu().detach().numpy()
                metrics_meters[metric_fn.__name__].add(metric_value)
            metrics_logs = {k: v.mean for k, v in metrics_meters.items()}

    # Update WANDB
    wandb_log.log({
        'Dice Score [validation]': metrics_logs['dice'],
        'IoU Score [validation]': metrics_logs['binary_jaccard_index'],
    }, step=epoch)

    net.train()
    return loss_meter.mean

def train():
    # Training vars
    epochs = 20
    batch_size = 4
    learning_rate = 1e-3
    patch_size = models_data[0]['resolution']
    is_saving_checkpoints = True

    # Model and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MCDCNN().to(device)

    # Dataloaders
    train_dataset = BinaryImageDataset(models_data[0]['models'], patch_size, 'training', img_transforms)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1, persistent_workers=True)

    val_dataset = BinaryImageDataset(models_data[0]['models'], patch_size, 'validation', img_transforms)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=1, persistent_workers=True)

    # Optimizers and Schedulers
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader), epochs=epochs)
    early_stopping = YOLOEarlyStopping(patience=10)

    # Metrics
    metrics = [
        F.dice,
        F.classification.binary_jaccard_index,
    ]
    loss_meter = meter.AverageValueMeter()
    metrics_meters = { metric.__name__: meter.AverageValueMeter() for metric in metrics }

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
            patch_size=models_data[0]['resolution'],
            model=model,
        )
    )
    
    run_name = wandb.run.name if wandb.run.name is not None else f'{model.__class__.__name__}-{batch_size}-{patch_size}'
    save_path = pathlib.Path('checkpoints', '1x1_conv', run_name)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    criterion = torch.nn.BCEWithLogitsLoss()
    criterion = criterion.to(device=device)

    global_step = 0
    last_best_score = float('inf')

    torch.cuda.empty_cache()
    for epoch in range(epochs):
        val_loss = 0.0
        progress_bar = tqdm.tqdm(total=int(len(train_dataset)), desc=f'Epoch {epoch + 1}/{epochs}', unit='img', position=0)

        for i, batch in enumerate(train_loader):
            # Get Batch Of Images
            batch_image = batch[0].to(device, non_blocking=True)
            batch_mask = batch[1].to(device, non_blocking=True)

            masks_pred = model(batch_image)
            loss = criterion(masks_pred, batch_mask)
            # masks_pred = torch.sigmoid(masks_pred)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # Show batch progress to terminal
            progress_bar.update(batch_image.shape[0])
            global_step += 1

            # Statistics
            loss_meter.add(loss.item())

            for metric_fn in metrics:
                metric_value = metric_fn(masks_pred, batch_mask.long(), ignore_index=0).cpu().detach().numpy()
                metrics_meters[metric_fn.__name__].add(metric_value)
            metrics_logs = {k: v.mean for k, v in metrics_meters.items()}

        # Evaluation of training
        val_loss = validate(model, val_loader, device, epoch, wandb_log)
        early_stopping(epoch, val_loss)

        if val_loss < last_best_score and is_saving_checkpoints:
            save_checkpoint(model, optimizer, epoch, run_name, True)
            last_best_score = val_loss

        # Update WANDB with Images
        try:
            pred_img = masks_pred.squeeze(0).detach().cpu().permute(1, 2 ,0).numpy()
            gt_img = batch_mask.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()

            gt_img *= 255.0
            pred_img *= 255.0

            # print(batch_image.shape, batch_mask.shape, masks_pred.shape)
            # print(pred_img.shape, gt_img.shape)
            # print(np.unique(pred_img), np.unique(gt_img))

            # visualize(
            #     save_path=None,
            #     prefix=None,
            #     pred_img=pred_img,
            #     gt_img=gt_img,
            # )

            wandb_log.log({
                'Images [training]': {
                    'Prediction': wandb.Image(pred_img),
                    'Ground Truth': wandb.Image(gt_img),
                },
            }, step=epoch)
        except Exception as e:
            print('Exception', e)

        # Update Progress Bar
        progress_bar.set_postfix(**{'Loss': loss_meter.mean})
        progress_bar.close()
       
        # Update WANDB
        wandb_log.log({
            'Learning Rate': optimizer.param_groups[0]['lr'],
            'Epoch': epoch,
            'Loss [training]': loss_meter.mean,
            'Loss [validation]': val_loss,
            'IoU Score [training]': metrics_logs['binary_jaccard_index'],
            'Dice Score [training]': metrics_logs['dice'],
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
    # Start Training
    try:
        train()
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
