import os
import sys
import wandb
import torch
import pathlib
import datetime
import argparse

import torch.nn as nn
import albumentations as A
import torch.distributed as dist
import torch.multiprocessing as mp
import segmentation_models_pytorch as smp

from tqdm import tqdm
from pathlib import Path
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, DistributedSampler

from settings import *
from evaluate import evaluate

from utils.dataset import Dataset, DatasetCacheType, DatasetType
from utils.early_stopping import EarlyStopping
from utils.metrics import SegmentationMetrics

# Logging
from utils.logging import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

torch.backends.cudnn.benchmark = True

def get_augmentations():
    train_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=args.patch_size, interpolation=1),
            A.PadIfNeeded(min_height=args.patch_size, min_width=args.patch_size, border_mode=0, value=(0,0,0), p=1.0),
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

    val_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=args.patch_size, interpolation=1),
            A.PadIfNeeded(min_height=args.patch_size, min_width=args.patch_size, border_mode=0, value=(0,0,0), p=1.0),

            ToTensorV2(),
        ],
    )
    return train_transforms, val_transforms

def get_loaders(rank, args):
    train_transforms, val_transforms = get_augmentations()

    if args.search_files:
        # Full Dataset
        all_imgs = [file for file in os.listdir(pathlib.Path(
            'data', 'imgs')) if not file.startswith('.')]

        # Split Dataset
        val_percent = 0.6
        n_dataset = int(round(val_percent * len(all_imgs)))

        # Load train & validation datasets
        train_dataset = Dataset(data_dir='data', images=all_imgs[:n_dataset], type=DatasetType.TRAIN, is_combined_data=True, patch_size=args.patch_size, transform=train_transforms)
        val_dataset = Dataset(data_dir='data', images=all_imgs[n_dataset:], type=DatasetType.VALIDATION, is_combined_data=True, patch_size=args.patch_size, transform=val_transforms)

        # Get Loaders
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=rank)

        train_loader = DataLoader(train_dataset, sampler=train_sampler, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=args.pin_memory, shuffle=False)
        val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory, shuffle=False)
        return

    train_dataset = Dataset(data_dir=r'data', img_dir=r'imgs', cache_type=DatasetCacheType.NONE, type=DatasetType.TRAIN, is_combined_data=True, patch_size=args.patch_size, transform=train_transforms)
    val_dataset = Dataset(data_dir=r'data', img_dir=r'imgs', cache_type=DatasetCacheType.NONE, type=DatasetType.VALIDATION, is_combined_data=True, patch_size=args.patch_size, transform=val_transforms)

    # Get Loaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=rank)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=args.pin_memory, shuffle=False)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory, shuffle=False)
    return train_dataset, val_dataset, train_loader, val_loader

def save_checkpoint(args: str, epoch: int, is_best: bool = False, optimizer: torch.optim.Optimizer = None):
    if not args.saving_checkpoints or not optimizer:
        return

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

    log.info('[SAVING MODEL]: Model checkpoint saved!')
    torch.save(state, Path('checkpoints', 'checkpoint.pth.tar'))

    if is_best:
        log.info('[SAVING MODEL]: Saving checkpoint of best model!')
        torch.save(state, Path('checkpoints', 'best-checkpoint.pth.tar'))

def load_checkpoint(model, optimizer: torch.optim.Optimizer, path: Path):
    log.info('[LOADING MODEL]: Started loading model checkpoint!')
    best_path = Path(path, 'best-checkpoint.pth.tar')

    if best_path.is_file():
        path = best_path
    else:
        path = Path(path, 'checkpoint.pth.tar')

    if not path.is_file():
        return

    state_dict = torch.load(path)
    start_epoch = state_dict['epoch']
    model.load_state_dict(state_dict['model_state'])
    optimizer.load_state_dict(state_dict['optimizer_state'])
    optimizer.name = state_dict['optimizer_name']

    log.info(f"[LOADING MODEL]: Loaded model with stats: epoch ({state_dict['epoch']}), time ({state_dict['time']})")
    return start_epoch

def train(gpu, args):
    rank = args.node_ranking * args.gpu_num + gpu
    dist.init_process_group(backend='gloo', init_method='env://', rank=rank, world_size=args.world_size)

    # net = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", decoder_channels=[1024, 512, 256, 128, 64], decoder_use_batchnorm=True, in_channels=3, classes=args.num_classes)
    net = smp.UnetPlusPlus(encoder_name="efficientnet-b7", encoder_weights="imagenet", decoder_use_batchnorm=True, in_channels=3, classes=args.num_classes)
    train_dataset, val_dataset, train_loader, val_loader = get_loaders(rank, args)

    model = nn.parallel.DistributedDataParallel(net, device_ids=[gpu])
    optimizer = torch.optim.AdamW(net.parameters(), weight_decay=args.weight_decay, eps=args.adam_eps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, steps_per_epoch=len(train_loader), epochs=args.epochs)
    early_stopping = EarlyStopping(patience=30, verbose=True)
    class_labels = { 0: 'background', 1: 'fire', 2: 'smoke' }
    start_epoch = 0

    if LOAD_MODEL:
        start_epoch = load_checkpoint(model, optimizer, Path('checkpoints'))
        model.cuda(non_blocking=True)

    log.info(f'''[TRAINING]:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Patch size:      {args.patch_size}
        Learning rate:   {args.lr}
        Training size:   {int(len(train_dataset))}
        Validation size: {int(len(val_dataset))}
        Checkpoints:     {args.saving_checkpoints}
        Mixed Precision: {args.using_amp}
    ''')

    wandb_log = wandb.init(project='firebot-unet', resume='allow', entity='firebot031')
    wandb_log.config.update(
        dict(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            save_checkpoint=args.saving_checkpoints,
            patch_size=args.patch_size,
            amp=args.using_amp,
            weight_decay=args.weight_decay,
            adam_epsilon=ADAM_EPSILON,
        )
    )

    class_weights = torch.tensor([ 1.0, 27745 / 23889, 27745 / 3502 ], dtype=torch.float).cuda()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.using_amp)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean').cuda()
    metric_calculator = SegmentationMetrics(activation='softmax')

    global_step = 0
    last_best_score = float('inf')

    pixel_acc = 0.0
    dice_score = 0.0
    jaccard_index = 0.0

    torch.cuda.empty_cache()
    optimizer.zero_grad(set_to_none=True)
    start = datetime.now()

    for epoch in range(start_epoch, args.epochs):
        model.train()

        epoch_loss = []
        progress_bar = tqdm(total=int(len(train_dataset)), desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img', position=0)

        for i, batch in enumerate(train_loader):
            # Zero Grad
            optimizer.zero_grad(set_to_none=True)

            # Get Batch Of Images
            batch_image = batch['image'].cuda(non_blocking=True)
            batch_mask = batch['mask'].cuda(non_blocking=True)

            # Predict
            with torch.cuda.amp.autocast(enabled=args.using_amp):
                masks_pred = model(batch_image)
                metrics = metric_calculator(batch_mask, masks_pred)
                loss = criterion(masks_pred, batch_mask)

            # Scale Gradients
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 15)

            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()

            # Show batch progress to terminal
            progress_bar.update(batch_image.shape[0])
            global_step += 1

            # Calculate training metrics
            pixel_acc += metrics['pixel_acc']
            dice_score += metrics['dice_score']
            jaccard_index += metrics['jaccard_index']
            epoch_loss.append(loss)

            # Evaluation of training
            eval_step = (int(len(train_dataset)) // (args.valid_eval_step * args.batch_size))
            if eval_step > 0 and global_step % eval_step == 0:
                val_loss = evaluate(model, val_loader, gpu, wandb_log)
                progress_bar.set_postfix(**{'Loss': torch.mean(torch.tensor(epoch_loss)).item()})

                if dist.get_rank() == 0:
                    try:
                        wandb_log.log({
                            'Learning Rate': optimizer.param_groups[0]['lr'],
                            'Images [training]': wandb.Image(batch_image[0].cpu(), masks={
                                'ground_truth': {
                                    'mask_data': batch_mask[0].cpu().numpy(),
                                    'class_labels': class_labels
                                },
                                'prediction': {
                                    'mask_data': masks_pred.argmax(dim=1)[0].cpu().numpy(),
                                    'class_labels': class_labels
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
                            'Learning Rate': optimizer.param_groups[0]['lr'],
                            'Epoch': epoch,
                            'Pixel Accuracy [training]': metrics['pixel_acc'].item(),
                            'IoU Score [training]': metrics['jaccard_index'].item(),
                            'Dice Score [training]': metrics['dice_score'].item(),
                        })

                if val_loss < last_best_score and dist.get_rank() == 0:
                    args.saving_checkpoint(epoch, True)
                    last_best_score = val_loss

        # Update Progress Bar
        if dist.get_rank() == 0:
            mean_loss = torch.mean(torch.tensor(epoch_loss)).item()
            progress_bar.set_postfix(**{'Loss': mean_loss})
            progress_bar.close()

            wandb_log.log({
                'Loss [training]': mean_loss,
                'Epoch': epoch,
            })

            # Saving last model
            if args.saving_checkpoint:
                save_checkpoint(epoch, False)

            # Early Stopping
            if early_stopping.early_stop:
                save_checkpoint(epoch, True)
                log.info(
                    f'[TRAINING]: Early stopping training at epoch {epoch}!')
                break

    # Push average training metrics
    dist.destroy_process_group()
    wandb_log.finish()

    if dist.get_rank() == 0:
        print("Training complete in: " + str(datetime.now() - start))


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'

    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', default=1, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--gpus', default=1, type=int, help='Number of gpus per node')
    parser.add_argument('-node-ranking', '--nr', default=0, type=int, help='Ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N', help='Number of total epochs to run')
    parser.add_argument('--batch-size', default=8, type=int, help='Batch size in training')
    parser.add_argument('--weight-decay', default=1e-3, type=float, help='Weight decay')
    parser.add_argument('--lr', default=1e-2, type=float, help='Learning rate')
    parser.add_argument('--patch-size', default=864, type=int, help='Patch size')
    parser.add_argument('--num-classes', default=3, type=int, help='Number of classes in the prediction')
    parser.add_argument('--adam-eps', default=1e-2, type=float, help='Adam optimizer epsilon')
    parser.add_argument('--valid-eval-step', default=2, type=int, help='Number of epochs before validation')
    parser.add_argument('--search-files', default=False, type=bool, help='Is dataloader searching files for the images?')
    args = parser.parse_args()

    args.num_workers = 0 if args.gpus > 0 else args.num_workers
    args.world_size = args.gpus * args.nodes

    # try:
    #     mp.spawn(train(), nprocs=args.gpus, args=(args,), join=True)
    # except KeyboardInterrupt:
    #     try:
    #         sys.exit(0)
    #     except SystemExit:
    #         os._exit(0)
    
    # logging.info('[TRAINING]: Training finished!')
