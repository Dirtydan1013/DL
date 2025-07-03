import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ICLEVRDataset(Dataset):
    def __init__(self, image_dir, json_path, object_json_path, image_size=64, device="cpu"):
        super().__init__()
        self.image_dir = image_dir
        self.device = device

        # 讀取 objects.json
        with open(object_json_path, 'r') as f:
            self.label2idx = json.load(f)
        self.idx2label = {v: k for k, v in self.label2idx.items()}
        self.num_classes = len(self.label2idx)

        # 讀取 train.json
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.filenames = list(self.data.keys())

        # Transform: 轉為 tensor 並正規化到 [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # (0, 1)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # → (-1, 1)
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        label_list = self.data[filename]

        # 圖片路徑
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image).to(self.device)

        # 多標籤 one-hot 向量
        label_tensor = torch.zeros(self.num_classes, device=self.device)
        for label in label_list:
            label_tensor[self.label2idx[label]] = 1.0

        return image, label_tensor
