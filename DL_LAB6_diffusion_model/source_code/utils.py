import torch
from torchvision.utils import save_image, make_grid
import os

def labels_to_onehot(labels, label2idx, num_classes=24):
    vec = torch.zeros(num_classes)
    for label in labels:
        if label in label2idx:
            vec[label2idx[label]] = 1.0
    return vec

def save_image_grid(images, filepath, nrow=8):
    # images: (B, 3, H, W), assumed to be in [-1, 1]
    images = (images.clamp(-1, 1) + 1) / 2  # convert to [0, 1]
    grid = make_grid(images, nrow=nrow)
    save_image(grid, filepath)

def save_denoising_sequence(sequence, output_path):
    """
    Save a sequence of images showing the denoising process.
    sequence: list of tensors [(1, 3, H, W), ...] in [-1, 1] domain
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sequence = torch.cat([(img.clamp(-1, 1) + 1) / 2 for img in sequence], dim=0)
    grid = make_grid(sequence, nrow=len(sequence))
    save_image(grid, output_path)
