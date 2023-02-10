import os
import sys
import torch
import wandb
import pathlib
import datetime
import argparse

import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp

from tqdm import tqdm
from pathlib import Path
from evaluate import evaluate
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from utils.dataset import Dataset, DatasetCacheType, DatasetType, BinaryDataset
from utils.early_stopping import YOLOEarlyStopping

# Logging
from utils.logging import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class UnetTraining:
    def __init__(self, args, net):
        assert net is not None

        self.args = args
        self.start_epoch = 0
        self.check_best_cooldown = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.class_weights = torch.tensor([ 27745 / 23889 ], dtype=torch.float).to(self.device) #torch.tensor([ 1.0, 27745 / 23889, 27745 / 3502 ], dtype=torch.float).to(self.device)
        self.model = net.to(self.device)

        self.get_augmentations()
        self.get_loaders()        

        self.optimizer = torch.optim.AdamW(self.model.parameters(), weight_decay=self.args.weight_decay, eps=self.args.adam_eps, lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.lr, total_steps=(self.args.epochs * self.args.batch_size))
        self.early_stopping = YOLOEarlyStopping(patience=30)
        self.class_labels = { 0: 'background', 1: 'fire' }

        if self.args.load_model:
            self.load_checkpoint(Path('checkpoints'))
            self.model.to(self.device)

    def get_augmentations(self):
        self.train_transforms = A.Compose(
            [
                # A.LongestMaxSize(max_size=800, interpolation=1),
                # A.PadIfNeeded(min_height=800, min_width=800, border_mode=0, value=(0, 0, 0), p=1.0),

                # Geometric transforms
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=5, p=0.5),
                A.CoarseDropout(
                    max_holes=6, max_height=12, max_width=12, min_holes=1, p=0.5
                ),
                A.ShiftScaleRotate(shift_limit=0.09, rotate_limit=0, p=0.2),
                A.OneOf(
                    [
                        A.GridDistortion(distort_limit=0.1, p=0.5),
                        A.OpticalDistortion(distort_limit=0.08, shift_limit=0.4, p=0.5),
                    ],
                    p=0.6
                ),
                A.Perspective(scale=(0.02, 0.07), p=0.5),

                # Color transforms
                A.ColorJitter(
                    brightness=0, contrast=0, saturation=0.12, hue=0.01, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.05, 0.20), contrast_limit=(-0.05, 0.20), p=0.6
                ),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10.0, 20.0), p=0.5),
                        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.02, 0.09), p=0.5),
                    ],
                    p=0.5
                ),
                A.GaussianBlur(blur_limit=(5, 7), p=0.39),
                ToTensorV2()
            ]
        )

        self.val_transforms = A.Compose(
            [
                # A.LongestMaxSize(max_size=self.args.patch_size, interpolation=1),
                # A.PadIfNeeded(min_height=self.args.patch_size, min_width=self.args.patch_size, border_mode=0, value=(0,0,0), p=1.0),
                ToTensorV2(),
            ],
        )

    def get_loaders(self):
        if self.args.search_files:
            # Full Dataset
            all_imgs = [file for file in os.listdir(pathlib.Path(
                'data', 'imgs')) if not file.startswith('.')]

            # Split Dataset
            val_percent = 0.6
            n_dataset = int(round(val_percent * len(all_imgs)))

            # Load train & validation datasets
            self.train_dataset = Dataset(data_dir='data', images=all_imgs[:n_dataset], type=DatasetType.TRAIN, is_combined_data=True, patch_size=self.args.patch_size, transform=self.train_transforms)
            self.val_dataset = Dataset(data_dir='data', images=all_imgs[n_dataset:], type=DatasetType.VALIDATION, is_combined_data=True, patch_size=self.args.patch_size, transform=self.val_transforms)

            # Get Loaders
            self.train_loader = DataLoader(self.train_dataset, num_workers=self.args.workers, batch_size=self.args.batch_size, pin_memory=self.args.pin_memory, shuffle=True, drop_last=True, persistent_workers=True, worker_init_fn=worker_init)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, num_workers=self.args.workers, pin_memory=self.args.pin_memory, shuffle=False, drop_last=False, persistent_workers=True, worker_init_fn=worker_init)
            return

        self.train_dataset = BinaryDataset(data_dir=r'data', img_dir=r'imgs', cache_type=DatasetCacheType.NONE, type=DatasetType.TRAIN, is_combined_data=True, patch_size=self.args.patch_size, transform=self.train_transforms)
        self.val_dataset = BinaryDataset(data_dir=r'data', img_dir=r'imgs', cache_type=DatasetCacheType.NONE, type=DatasetType.VALIDATION, is_combined_data=True, patch_size=self.args.patch_size, transform=self.val_transforms)

        # Get Loaders    
        self.train_loader = DataLoader(self.train_dataset, num_workers=self.args.workers, batch_size=self.args.batch_size, pin_memory=self.args.pin_memory, shuffle=True, drop_last=True, persistent_workers=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, num_workers=self.args.workers, pin_memory=self.args.pin_memory, shuffle=False, drop_last=False, persistent_workers=True)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        if not self.args.save_checkpoints:
            return

        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module

        state = {
            'time': str(datetime.datetime.now()),
            'model_state': self.model.state_dict(),
            'model_name': type(self.model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch
        }

        if is_best is False:
            # log.info('[SAVING MODEL]: Model checkpoint saved!')
            torch.save(state, Path('checkpoints', self.run_name, 'checkpoint.pth.tar'))

        if is_best:
            log.info('[SAVING MODEL]: Saving checkpoint of best model!')
            torch.save(state, Path('checkpoints', self.run_name, 'best-checkpoint.pth.tar'))

    def load_checkpoint(self, path: Path):
        log.info('[LOADING MODEL]: Started loading model checkpoint!')
        best_path = Path(path, 'best-checkpoint.pth.tar')

        if best_path.is_file():
            path = best_path
        else:
            path = Path(path, 'checkpoint.pth.tar')

        if not path.is_file():
            return

        state_dict = torch.load(path)
        self.start_epoch = state_dict['epoch']
        self.model.load_state_dict(state_dict['model_state'])
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        self.optimizer.name = state_dict['optimizer_name']
        log.info(
            f"[LOADING MODEL]: Loaded model with stats: epoch ({state_dict['epoch']}), time ({state_dict['time']})")

    def train(self):
        log.info(f'''[TRAINING]:
            Model:           {self.args.model}
            Encoder:         {self.args.encoder}
            Resolution:      {self.args.patch_size}x{self.args.patch_size}
            Epochs:          {self.args.epochs}
            Batch size:      {self.args.batch_size}
            Patch size:      {self.args.patch_size}
            Learning rate:   {self.args.lr}
            Training size:   {int(len(self.train_dataset))}
            Validation size: {int(len(self.val_dataset))}
            Checkpoints:     {self.args.save_checkpoints}
            Device:          {self.device.type}
            Mixed Precision: {self.args.use_amp}
        ''')

        wandb_log = wandb.init(project='semantic-article', entity='firebot031')
        wandb_log.config.update(
            dict(
                epochs=self.args.epochs,
                batch_size=self.args.batch_size,
                learning_rate=self.args.lr,
                save_checkpoint=self.args.save_checkpoints,
                patch_size=self.args.patch_size,
                amp=self.args.use_amp,
                weight_decay=self.args.weight_decay,
                adam_epsilon=self.args.adam_eps,
                encoder=self.args.encoder,
                model=self.args.model,
            )
        )

        self.run_name = wandb.run.name if wandb.run.name is not None else f'{self.args.model}-{self.args.encoder}-{self.args.batch_size}-{self.args.patch_size}'
        save_path = Path('checkpoints', self.run_name)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights, reduction='mean') if self.args.classes > 1 else torch.nn.BCEWithLogitsLoss()
        criterion = criterion.to(device=self.device)

        global_step = 0
        last_best_score = float('inf')
        masks_pred = []

        torch.cuda.empty_cache()
        for epoch in range(self.start_epoch, self.args.epochs):
            epoch_loss = []
            val_loss = 0.0
            progress_bar = tqdm(total=int(len(self.train_dataset)), desc=f'Epoch {epoch + 1}/{self.args.epochs}', unit='img', position=0)

            for i, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad(set_to_none=True)

                # Get Batch Of Images
                batch_image = batch['image'].to(self.device, non_blocking=True)
                batch_mask = batch['mask'].to(self.device, non_blocking=True)

                # Predict
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    masks_pred = self.model(batch_image)
                    loss = criterion(masks_pred, batch_mask)

                # Scale Gradients                
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 255.0)

                grad_scaler.step(self.optimizer)
                grad_scaler.update()

                # Show batch progress to terminal
                progress_bar.update(batch_image.shape[0])
                global_step += 1
                epoch_loss.append(loss.item())

                # Evaluation of training
                eval_step = (int(len(self.train_dataset)) // (self.args.eval_step * self.args.batch_size))
                if eval_step > 0 and global_step % eval_step == 0:
                    val_loss = evaluate(self.model, self.val_loader, self.device, self.args.classes)
                    self.scheduler.step()
                    self.early_stopping(epoch, val_loss)

                    if epoch >= self.check_best_cooldown and val_loss < last_best_score:
                        self.save_checkpoint(epoch, True)
                        last_best_score = val_loss

                    # Update WANDB with Images
                    try:
                        wandb_log.log({
                            'Images [training]': {
                                'Image': wandb.Image(batch_image[0].cpu()),
                                'Ground Truth': wandb.Image(batch_mask[0].squeeze(0).detach().cpu().numpy()),
                                'Prediction': wandb.Image(torch.sigmoid(masks_pred[0].squeeze(0).detach().cpu().float()).numpy()),
                            },
                        }, step=epoch)
                    except Exception as e:
                        print(e)

            # Update Progress Bar
            mean_loss = np.mean(epoch_loss)
            progress_bar.set_postfix(**{'Loss': mean_loss})
            progress_bar.close()

            # Update WANDB
            wandb_log.log({
                'Learning Rate': self.optimizer.param_groups[0]['lr'],
                'Epoch': epoch,
                'Loss [training]': mean_loss,
                'Loss [validation]': val_loss,
            }, step=epoch)

            # Saving last model
            if self.args.save_checkpoints:
                self.save_checkpoint(epoch, False)

            # Early Stopping
            if self.early_stopping.early_stop:
                self.save_checkpoint(epoch, False)
                log.info(
                    f'[TRAINING]: Early stopping training at epoch {epoch}!')
                break

        # Push average training metrics
        wandb_log.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Unet', help='Which model you want to train?')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--adam-eps', nargs='+', type=float, default=1e-3, help='Adam epsilon')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay that is used for AdamW')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--encoder', default="", help='Backbone encoder')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--workers', type=int, default=8, help='Number of DataLoader workers')
    parser.add_argument('--classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--patch-size', type=int, default=800, help='Patch size')
    parser.add_argument('--pin-memory', type=bool, default=True, help='Use pin memory for DataLoader?')
    parser.add_argument('--eval-step', type=int, default=1, help='Run evaluation every # step')
    parser.add_argument('--load-model', action='store_true', help='Load model from directories?')
    parser.add_argument('--save-checkpoints', action='store_true', help='Save checkpoints after every epoch?')
    parser.add_argument('--use-amp', action='store_true', help='Use Pytorch Automatic Mixed Precision?')
    parser.add_argument('--search-files', type=bool, default=False, help='Should DataLoader search your files for images?')
    args = parser.parse_args()

    args.encoder = 'resnet34' if args.encoder == '' else args.encoder

    if args.model == 'UnetPlusPlus':
        net = smp.UnetPlusPlus(encoder_name=args.encoder, encoder_weights='imagenet', decoder_use_batchnorm=True, in_channels=3, classes=args.classes)
    elif args.model == 'MAnet':
        net = smp.MAnet(encoder_name=args.encoder, encoder_depth=5, encoder_weights='imagenet', decoder_use_batchnorm=True, in_channels=3, classes=args.classes)
    elif args.model == 'Linknet':
        net = smp.Linknet(encoder_name=args.encoder, encoder_depth=5, encoder_weights='imagenet', decoder_use_batchnorm=True, in_channels=3, classes=args.classes)
    elif args.model == 'FPN':
        net = smp.FPN(encoder_name=args.encoder, encoder_weights='imagenet', in_channels=3, classes=args.classes)
    elif args.model == 'PSPNet':
        net = smp.PSPNet(encoder_name=args.encoder, encoder_weights='imagenet', in_channels=3, classes=args.classes)
    elif args.model == 'PAN':
        net = smp.PAN(encoder_name=args.encoder, encoder_weights='imagenet', in_channels=3, classes=args.classes)
    elif args.model == 'DeepLabV3':
        net = smp.DeepLabV3(encoder_name=args.encoder, encoder_weights='imagenet', in_channels=3, classes=args.classes)
    elif args.model == 'DeepLabV3Plus':
        net = smp.DeepLabV3Plus(encoder_name=args.encoder, encoder_weights='imagenet', in_channels=3, classes=args.classes)
    else:
        net = smp.Unet(encoder_name=args.encoder, encoder_weights='imagenet', decoder_use_batchnorm=True, in_channels=3, classes=args.classes)

    training = UnetTraining(args, net)
    try:
        training.train()
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
