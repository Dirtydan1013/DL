import torch
from torch.utils.data import DataLoader
from oxford_pet import load_dataset
from train import train, get_args
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet

if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入資料
    train_dataset = load_dataset(args.data_path, mode="train")
    val_dataset = load_dataset(args.data_path, mode="valid")
    test_dataset = load_dataset(args.data_path, mode="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 選擇模型架構
    if args.model == 'unet':
        model = UNet(in_channels=3, out_channels=1)
    elif args.model == 'resnet':
        model = ResNet34_UNet(n_classes=1)
    else:
        raise ValueError("Unsupported model architecture")

    model = model.to(device)

    # 執行訓練
    train(args)
