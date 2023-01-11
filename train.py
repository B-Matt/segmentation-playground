import os
import sys
import torch
import wandb
import pathlib
import datetime
import argparse

import albumentations as A
import segmentation_models_pytorch as smp

from tqdm import tqdm
from pathlib import Path
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from evaluate import evaluate

from utils.dataset import Dataset, DatasetCacheType, DatasetType
from utils.early_stopping import EarlyStopping
from utils.metrics import SegmentationMetrics

# Logging
from utils.logging import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

torch.backends.cudnn.benchmark = True

class UnetTraining:
    def __init__(self, args, net):
        assert net is not None

        self.args = args
        self.start_epoch = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.class_weights = torch.tensor([ 1.0, 27745 / 23889, 27745 / 3502 ], dtype=torch.float).to(self.device)
        self.model = net.to(self.device)

        self.get_augmentations()
        self.get_loaders()        

        self.optimizer = torch.optim.AdamW(self.model.parameters(), weight_decay=self.args.weight_decay, eps=self.args.adam_eps)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=2e-3, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        self.early_stopping = EarlyStopping(patience=30, verbose=True)
        self.class_labels = { 0: 'background', 1: 'fire', 2: 'smoke' }

        if self.args.load_model:
            self.load_checkpoint(Path('checkpoints'))
            self.model.to(self.device)

    def get_augmentations(self):
        self.train_transforms = A.Compose(
            [
                A.LongestMaxSize(max_size=self.args.patch_size, interpolation=1),
                A.PadIfNeeded(min_height=self.args.patch_size, min_width=self.args.patch_size, border_mode=0, value=(0,0,0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=0.8),

                A.Rotate(limit=(0, 10), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.ElasticTransform(p=0.3),
                    A.GridDistortion(p=0.4),
                ], p=0.8),
                A.OneOf([
                    A.Blur(p=0.3),
                    A.MotionBlur(p=0.5),
                    A.Sharpen(p=0.2),
                ], p=0.85),
                
                ToTensorV2(),
            ]
        )

        self.val_transforms = A.Compose(
            [
                A.LongestMaxSize(max_size=self.args.patch_size, interpolation=1),
                A.PadIfNeeded(min_height=self.args.patch_size, min_width=self.args.patch_size, border_mode=0, value=(0,0,0), p=1.0),

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
            train_sampler = RandomSampler(self.train_dataset)
            val_sampler = SequentialSampler(self.val_dataset)

            self.train_loader = DataLoader(self.train_dataset, sampler=train_sampler, num_workers=self.args.workers, batch_size=self.args.batch_size, pin_memory=self.args.pin_memory, shuffle=False)
            self.val_loader = DataLoader(self.val_dataset, sampler=val_sampler, batch_size=self.args.batch_size, num_workers=self.args.workers, pin_memory=self.args.pin_memory, shuffle=False)
            return

        self.train_dataset = Dataset(data_dir=r'data', img_dir=r'imgs', cache_type=DatasetCacheType.NONE, type=DatasetType.TRAIN, is_combined_data=True, patch_size=self.args.patch_size, transform=self.train_transforms)
        self.val_dataset = Dataset(data_dir=r'data', img_dir=r'imgs', cache_type=DatasetCacheType.NONE, type=DatasetType.VALIDATION, is_combined_data=True, patch_size=self.args.patch_size, transform=self.val_transforms)

        # Get Loaders
        train_sampler = RandomSampler(self.train_dataset)
        val_sampler = SequentialSampler(self.val_dataset)
    
        self.train_loader = DataLoader(self.train_dataset, sampler=train_sampler, num_workers=self.args.workers, batch_size=self.args.batch_size, pin_memory=self.args.pin_memory, shuffle=False)
        self.val_loader = DataLoader(self.val_dataset, sampler=val_sampler, batch_size=self.args.batch_size, num_workers=self.args.workers, pin_memory=self.args.pin_memory, shuffle=False)

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

        log.info('[SAVING MODEL]: Model checkpoint saved!')
        torch.save(state, Path('checkpoints', 'checkpoint.pth.tar'))

        if is_best:
            log.info('[SAVING MODEL]: Saving checkpoint of best model!')
            torch.save(state, Path('checkpoints', 'best-checkpoint.pth.tar'))

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

        wandb_log = wandb.init(project='firebot-unet', resume='allow', entity='firebot031')
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
            )
        )

        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights, reduction='mean').to(device=self.device)
        metric_calculator = SegmentationMetrics(activation='softmax')

        global_step = 0
        last_best_score = float('inf')

        pixel_acc = 0.0
        dice_score = 0.0
        jaccard_index = 0.0

        torch.cuda.empty_cache()
        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.start_epoch, self.args.epochs):
            self.model.train()

            epoch_loss = []
            progress_bar = tqdm(total=int(len(self.train_dataset)), desc=f'Epoch {epoch + 1}/{self.args.epochs}', unit='img', position=0)

            for i, batch in enumerate(self.train_loader):
                # Zero Grad
                self.optimizer.zero_grad(set_to_none=True)

                # Get Batch Of Images
                batch_image = batch['image'].to(self.device, non_blocking=True)
                batch_mask = batch['mask'].to(self.device, non_blocking=True)

                # Predict
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    masks_pred = self.model(batch_image)
                    metrics = metric_calculator(batch_mask, masks_pred)
                    loss = criterion(masks_pred, batch_mask)

                # Scale Gradients
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 15)

                grad_scaler.step(self.optimizer)
                grad_scaler.update()
                self.scheduler.step()                

                # Show batch progress to terminal
                progress_bar.update(batch_image.shape[0])
                global_step += 1

                # Calculate training metrics
                pixel_acc += metrics['pixel_acc']
                dice_score += metrics['dice_score']
                jaccard_index += metrics['jaccard_index']
                epoch_loss.append(loss)

                # Evaluation of training
                eval_step = (int(len(self.train_dataset)) // (self.args.eval_step * self.args.batch_size))
                if eval_step > 0 and global_step % eval_step == 0:
                    val_loss = evaluate(self.model, self.val_loader, self.device, wandb_log)
                    progress_bar.set_postfix(**{'Loss': torch.mean(torch.tensor(epoch_loss)).item()})

                    try:
                        wandb_log.log({
                            'Learning Rate': self.optimizer.param_groups[0]['lr'],
                            'Images [training]': wandb.Image(batch_image[0].cpu(), masks={
                                'ground_truth': {
                                    'mask_data': batch_mask[0].cpu().numpy(),
                                    'class_labels': self.class_labels
                                },
                                'prediction': {
                                    'mask_data': masks_pred.argmax(dim=1)[0].cpu().numpy(),
                                    'class_labels': self.class_labels
                                }
                            }
                            ),
                            'Epoch': epoch,
                            'Pixel Accuracy [training]': metrics['pixel_acc'].item(),
                            'IoU Score [training]': metrics['jaccard_index'].item(),
                            'Dice Score [training]': metrics['dice_score'].item(),
                        })
                    except:
                        wandb_log.log({
                            'Learning Rate': self.optimizer.param_groups[0]['lr'],
                            'Epoch': epoch,
                            'Pixel Accuracy [training]': metrics['pixel_acc'].item(),
                            'IoU Score [training]': metrics['jaccard_index'].item(),
                            'Dice Score [training]': metrics['dice_score'].item(),
                        })

                    if val_loss < last_best_score:
                        self.save_checkpoint(epoch, True)
                        last_best_score = val_loss

            # Update Progress Bar
            mean_loss = torch.mean(torch.tensor(epoch_loss)).item()
            progress_bar.set_postfix(**{'Loss': mean_loss})
            progress_bar.close()

            wandb_log.log({
                'Loss [training]': mean_loss,
                'Epoch': epoch,
            })

            # Saving last modelself.val_dataset.type
            if self.save_checkpoint:
                self.save_checkpoint(epoch, False)

            # Early Stopping
            if self.early_stopping.early_stop:
                self.save_checkpoint(epoch, True)
                log.info(
                    f'[TRAINING]: Early stopping training at epoch {epoch}!')
                break

        # Push average training metrics
        wandb_log.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Unet', help='Which model you want to train?')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--adam-eps', nargs='+', type=float, default=1e-2, help='Adam epsilon')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay that is used for AdamW')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--encoder', default="", help='Backbone encoder')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--workers', type=int, default=6, help='Number of DataLoader workers')
    parser.add_argument('--classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--patch-size', type=int, default=800, help='Patch size')
    parser.add_argument('--pin-memory', type=bool, default=True, help='Use pin memory for DataLoader?')
    parser.add_argument('--eval-step', type=int, default=2, help='Run evaluation every # step')
    parser.add_argument('--load-model', action='store_true', help='Load model from directories?')
    parser.add_argument('--save-checkpoints', action='store_true', help='Save checkpoints after every epoch?')
    parser.add_argument('--use-amp', type=bool, default=True, help='Use Pytorch Automatic Mixed Precision?')
    parser.add_argument('--search-files', type=bool, default=False, help='Should DataLoader search your files for images?')
    args = parser.parse_args()

    if args.model == 'UnetPlusPlus':
        net = smp.UnetPlusPlus(encoder_name=('resnet34' if args.encoder == '' else args.encoder), encoder_weights='imagenet', decoder_use_batchnorm=True, in_channels=3, classes=args.classes)
    elif args.model == 'MAnet':
        net = smp.MAnet(encoder_name=('resnet34' if args.encoder == '' else args.encoder), encoder_depth=5, encoder_weights='imagenet', decoder_use_batchnorm=True, in_channels=3, classes=args.classes)
    elif args.model == 'Linknet':
        net = smp.Linknet(encoder_name=('resnet34' if args.encoder == '' else args.encoder), encoder_depth=5, encoder_weights='imagenet', decoder_use_batchnorm=True, in_channels=3, classes=args.classes)
    elif args.model == 'FPN':
        net = smp.FPN(encoder_name=('resnet34' if args.encoder == '' else args.encoder), encoder_weights='imagenet', in_channels=3, classes=args.classes)
    elif args.model == 'PSPNet':
        net = smp.PSPNet(encoder_name=('resnet34' if args.encoder == '' else args.encoder), encoder_weights='imagenet', in_channels=3, classes=args.classes)
    elif args.model == 'PAN':
        net = smp.PAN(encoder_name=('resnet34' if args.encoder == '' else args.encoder), encoder_weights='imagenet', in_channels=3, classes=args.classes)
    elif args.model == 'DeepLabV3':
        net = smp.DeepLabV3(encoder_name=('resnet34' if args.encoder == '' else args.encoder), encoder_weights='imagenet', in_channels=3, classes=args.classes)
    elif args.model == 'DeepLabV3Plus':
        net = smp.DeepLabV3Plus(encoder_name=('resnet34' if args.encoder == '' else args.encoder), encoder_weights='imagenet', in_channels=3, classes=args.classes)
    else:
        net = smp.Unet(encoder_name=('resnet34' if args.encoder == '' else args.encoder), encoder_weights='imagenet', decoder_use_batchnorm=True, in_channels=3, classes=args.classes)

    training = UnetTraining(args, net)

    try:
        training.train()
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    
    logging.info('[TRAINING]: Training finished!')
