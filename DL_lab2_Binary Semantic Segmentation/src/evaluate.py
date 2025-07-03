# src/evaluate.py

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import dice_score
from tqdm import tqdm

def load_model(model_name, weight_path, device):
    if model_name == 'unet':
        model = UNet(in_channels=3, out_channels=1)
    else:
        model = ResNet34_UNet(n_classes=1)

    ckpt = torch.load(weight_path, map_location=device)
    # 支援完整 checkpoint 或纯 state_dict
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    return model.to(device)

def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    total_dice = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            imgs  = batch['image'].float().to(device)  # (B,3,H,W)
            masks = batch['mask'].float().to(device)   # (B,1,H,W)
            
            outputs = model(imgs)  # (B,1,h_out,w_out)
            # 上采样回原始尺寸
            outputs_up = F.interpolate(
                outputs,
                size=(imgs.shape[2], imgs.shape[3]),
                mode='bilinear',
                align_corners=False
            )
            preds = (torch.sigmoid(outputs_up) > threshold).float()

            # 累加每张图的 Dice
            batch_size = preds.size(0)
            total_dice += dice_score(preds.cpu(), masks.cpu()) * batch_size
            total_samples += batch_size

    return total_dice / total_samples if total_samples > 0 else 0.0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate on test split")
    parser.add_argument('--data_path',  type=str, required=True,
                        help='数据集根目录')
    parser.add_argument('--model',      type=str, required=True,
                        choices=['unet','resnet'], help='模型架构')
    parser.add_argument('--weights',    type=str, required=True,
                        help='checkpoint 路径 (.pth)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批量大小')
    parser.add_argument('--threshold',  type=float, default=0.5,
                        help='二值化阈值')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    test_ds = load_dataset(args.data_path, mode='test')
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = load_model(args.model, args.weights, device)
    dice_score_avg = evaluate_model(model, test_loader, device, threshold=args.threshold)
    print(f"Average Dice Score on test set: {dice_score_avg:.4f}")
