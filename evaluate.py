import torch
import numpy as np
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils.meter as meter

from tqdm import tqdm

@torch.inference_mode()
def evaluate(net, dataloader, device, classes, epoch, wandb_log):
    net.eval()
    num_val_batches = len(dataloader)

    criterion = torch.nn.CrossEntropyLoss() if classes > 1 else torch.nn.BCEWithLogitsLoss()
    criterion = criterion.to(device=device)

    metrics = [
        smp.metrics.iou_score,
        smp.metrics.f1_score,
        smp.metrics.accuracy,
        smp.metrics.recall,
    ]
    loss_meter = meter.AverageValueMeter()
    metrics_meters = { metric.__name__: meter.AverageValueMeter() for metric in metrics }

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation', position=1, unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']

        image = image.to(device=device, non_blocking=True)
        mask_true = mask_true.to(device=device, non_blocking=True)

        with torch.no_grad():
            mask_pred = net(image)
            mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()

            loss = criterion(mask_pred, mask_true)
            loss_meter.add(loss.item())

            tp, fp, fn, tn = smp.metrics.get_stats(mask_pred, mask_true.round().long(), mode='binary', threshold=0.5)
            for metric_fn in metrics:
                metric_value = metric_fn(tp, fp, fn, tn, reduction="micro").cpu().detach().numpy()
                metrics_meters[metric_fn.__name__].add(metric_value)
            metrics_logs = {k: v.mean for k, v in metrics_meters.items()}

    # Update WANDB
    wandb_log.log({
        'Loss [validation]': loss_meter.mean,
        'IoU [validation]': metrics_logs['iou_score'],
        'F1 Score [validation]': metrics_logs['f1_score'],
        'Recall [validation]': metrics_logs['sensitivity'],
        'Accuracy [validation]': metrics_logs['accuracy'],
    }, step=epoch)

    net.train()
    return loss_meter.mean