import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet

def get_args():
    parser = argparse.ArgumentParser(description="Single-image Inference for Lab2 Models")
    parser.add_argument('--image_path',  type=str, required=True,
                        help='待推论的圖像檔案路徑')
    parser.add_argument('--model',       type=str, required=True, choices=['unet', 'resnet'],
                        help='模型架構，選擇 unet 或 resnet')
    parser.add_argument('--weights',     type=str, required=True,
                        help='訓練好的 .pth 權重檔路徑')
    parser.add_argument('--threshold',   type=float, default=0.5,
                        help='Sigmoid 二值化閾值 (預設 0.5)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='結果 mask 存檔路徑（預設：<image_basename>_mask.png）')
    return parser.parse_args()

def load_model(weight_path, model_name, device):
    """根據名稱載入對應架構並讀取權重"""
    if model_name == 'resnet':
        model = ResNet34_UNet(n_classes=1)
    else:
        model = UNet(in_channels=3, out_channels=1)

     # 2. 载入 checkpoint
    ckpt = torch.load(weight_path, map_location=device)
    # 如果是完整 checkpoint，就取出 'model' 部分；否则就直接当做 state_dict
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

def preprocess_image(image_path, size=(256,256)):
    """
    讀圖 → resize → 轉 numpy float32 (0–255) → CHW → Tensor
    與 SimpleOxfordPetDataset.__getitem__ 中的 resize 完全對齊 :contentReference[oaicite:0]{index=0}
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)      # no /255.0
    chw = np.transpose(arr, (2,0,1))           # HWC → CHW
    tensor = torch.from_numpy(chw).unsqueeze(0)  # (1,3,H,W)
    return tensor

def postprocess_and_save(mask, save_path):
    """
    mask: Tensor (1,1,H,W)，值為 0/1
    → 轉 uint8 灰度圖並存檔
    """
    arr = (mask[0,0].cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr, mode='L').save(save_path)
    print(f"Saved mask → {save_path}")

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 載入模型
    model = load_model(args.weights, args.model, device)

    # 2. 前處理
    img_tensor = preprocess_image(args.image_path).to(device)

    # 3. 推論
    with torch.no_grad():
        logits = model(img_tensor)               # raw output
        probs  = torch.sigmoid(logits)
        binary = (probs > args.threshold).float()  # 與 evaluate.py 中 threshold 一致 :contentReference[oaicite:1]{index=1}

    # 4. 若需要，可把 binary upsample 回原尺寸；此处输入即为模型训练用的尺寸(256×256)
    #    如要上采样回原图大小，可解注释以下： 
    # orig = Image.open(args.image_path)
    # W, H = orig.size
    # binary = F.interpolate(binary, size=(H, W), mode='nearest')

    # 5. 存檔
    base = os.path.splitext(os.path.basename(args.image_path))[0]
    out_path = args.output_path or f"{base}_mask.png"
    postprocess_and_save(binary, out_path)

if __name__ == "__main__":
    main()
