import torch
import math

import numpy as np
import torchmetrics.functional as F
import segmentation_models_pytorch.utils.meter as meter

from tqdm import tqdm

@torch.inference_mode()
def evaluate(net, dataloader, device, classes, epoch, wandb_log):
    net.eval()
    num_val_batches = len(dataloader)

    criterion = torch.nn.CrossEntropyLoss() if classes > 1 else torch.nn.BCEWithLogitsLoss()
    criterion = criterion.to(device=device)

    loss_meter = meter.AverageValueMeter()
    reports_data = {
        'Dice Score': [],
        'IoU Score': [],
    }

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation', position=1, unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']

        image = image.to(device=device, non_blocking=True)
        mask_true = mask_true.to(device=device, non_blocking=True)

        with torch.no_grad():
            mask_pred = net(image)
            mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()

            loss = criterion(mask_pred, mask_true)
            loss_meter.add(loss.item())

            threshold = 0.5
            dice_score = F.dice(mask_pred, mask_true.long(), threshold=threshold, ignore_index=0).item()
            jaccard_index = F.classification.binary_jaccard_index(mask_pred, mask_true.long(), threshold=threshold, ignore_index=0).item()

            if math.isnan (dice_score):
                dice_score = 0.0

            if math.isnan (jaccard_index):
                jaccard_index = 0.0

            reports_data['Dice Score'].append(dice_score)
            reports_data['IoU Score'].append(jaccard_index)

    # Update WANDB
    wandb_log.log({
        'Loss [validation]': loss_meter.mean,
        'IoU Score [validation]': np.mean(reports_data['IoU Score']),
        'Dice Score [validation]': np.mean(reports_data['Dice Score']),
    }, step=epoch)

    net.train()
    return loss_meter.mean
