import torch
import numpy as np

from tqdm import tqdm
from utils.metrics import SegmentationMetrics

@torch.inference_mode()
def evaluate(net, dataloader, device, training):
    net.eval()
    num_val_batches = len(dataloader)
    # class_weights = torch.tensor([ 1.0, 3583 / 3100, 3583 / 439 ], dtype=torch.float).to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device=device)
    metric_calculator = SegmentationMetrics(activation='none')
    
    pixel_accuracy = []
    dice_score = []
    iou_score = []
    global_loss = []

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation', position=1, unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']

        image = image.to(device=device, non_blocking=True)
        mask_true = mask_true.to(device=device, non_blocking=True)

        with torch.no_grad():
            mask_pred = net(image)
            loss = criterion(mask_pred, mask_true)
            metrics = metric_calculator(mask_true, mask_pred)

            pixel_accuracy.append(metrics['pixel_acc'])
            iou_score.append(metrics['jaccard_index'])
            dice_score.append(metrics['dice_score'])
    
            global_loss.append(loss.item())

    net.train()
    return np.mean(global_loss), np.mean(pixel_accuracy), np.mean(iou_score), np.mean(dice_score)