import torch
import numpy as np

import segmentation_models_pytorch as smp

from tqdm import tqdm

@torch.inference_mode()
def evaluate(net, dataloader, device, classes):
    net.eval()
    num_val_batches = len(dataloader)

    criterion = torch.nn.CrossEntropyLoss() if classes > 1 else torch.nn.BCEWithLogitsLoss()
    criterion = criterion.to(device=device)
    global_loss = []

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation', position=1, unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']

        image = image.to(device=device, non_blocking=True)
        mask_true = mask_true.to(device=device, non_blocking=True)

        with torch.no_grad():
            mask_pred = net(image)
            mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
            loss = criterion(mask_pred, mask_true)
            global_loss.append(loss.item())

    net.train()
    return np.mean(global_loss)