import torch
import numpy as np

from tqdm import tqdm

@torch.inference_mode()
def evaluate(net, dataloader, device, training):
    num_val_batches = len(dataloader)
    class_weights = torch.tensor([ 1.0, 3583 / 3100, 3583 / 439 ], dtype=torch.float).to(device)

    net.eval()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean').to(device=device)
    global_loss = []

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation', position=1, unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']

        image = image.to(device=device, non_blocking=True)
        mask_true = mask_true.to(device=device, non_blocking=True)

        with torch.no_grad():
            mask_pred = net(image)
            loss = criterion(mask_pred, mask_true)
            global_loss.append(loss.cpu())

    net.train()
    return np.mean(global_loss)