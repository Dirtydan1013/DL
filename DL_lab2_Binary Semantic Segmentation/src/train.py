# src/train.py

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from oxford_pet import load_dataset
from utils import dice_score

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='数据集根目录')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--model', type=str, choices=['unet', 'resnet'], required=True,
                        help='模型架构：unet 或 resnet')
    return parser.parse_args()

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_ds = load_dataset(args.data_path, mode='train')
    val_ds   = load_dataset(args.data_path, mode='valid')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    if args.model == 'unet':
        from models.unet import UNet as Model
    else:
        from models.resnet34_unet import ResNet34_UNet as Model

    model     = Model().to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = BCEWithLogitsLoss()

    best_val_loss = float('inf')
    ckpt_dir = os.path.join('saved_models', args.model+"happy1")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # --- 训练 ---
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            imgs  = batch['image'].float().to(device)  # (B,3,H,W)
            masks = batch['mask'].float().to(device)   # (B,1,H,W)

            outputs = model(imgs)  # (B,1,h_out,w_out)
           
        
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * imgs.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # --- 验证 ---
        model.eval()
        total_val_loss = 0.0
        total_val_dice = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs  = batch['image'].float().to(device)
                masks = batch['mask'].float().to(device)

                outputs = model(imgs)
              
                loss = criterion(outputs, masks)
                total_val_loss += loss.item() * imgs.size(0)

                preds = (torch.sigmoid(outputs) > 0.5).float()
                total_val_dice += dice_score(preds.cpu(), masks.cpu()) * imgs.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        avg_val_dice = total_val_dice / len(val_loader.dataset)

        # --- Checkpoint ---
        ckpt = {
            'epoch':     epoch,
            'model':     model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        epoch_path = os.path.join(ckpt_dir, f"{args.model}_epoch{epoch}.pth")
        torch.save(ckpt, epoch_path)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(ckpt_dir, f"{args.model}_best.pth")
            torch.save(ckpt, best_path)

        print(f"Epoch {epoch}/{args.epochs} "
              f"| Train Loss: {avg_train_loss:.4f} "
              f"| Val Loss: {avg_val_loss:.4f} "
              f"| Val Dice: {avg_val_dice:.4f}")

if __name__ == '__main__':
    args = get_args()
    train(args)
