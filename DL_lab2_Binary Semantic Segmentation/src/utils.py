import torch

def dice_score(pred_mask, gt_mask, epsilon=1e-6):
    """
    計算 Dice Score
    pred_mask 和 gt_mask 都應該是 shape = (B, 1, H, W) 的 tensor
    預設 pred_mask 是已經經過 sigmoid 並 threshold 的 binary mask
    """
    pred_mask = pred_mask.float()
    gt_mask = gt_mask.float()

    intersection = (pred_mask * gt_mask).sum(dim=(1, 2, 3))
    union = pred_mask.sum(dim=(1, 2, 3)) + gt_mask.sum(dim=(1, 2, 3))

    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()

def center_crop_tensor(tensor, target_height, target_width):
    _, _, h, w = tensor.size()
    top = (h - target_height) // 2
    left = (w - target_width) // 2
    return tensor[:, :, top:top+target_height, left:left+target_width]
