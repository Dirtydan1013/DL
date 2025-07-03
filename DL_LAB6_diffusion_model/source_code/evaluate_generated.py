## using the following command to get accuracy
## python evaluate_generated.py --json new_test.json --image-dir C:path\images\new_test
##python evaluate_generated.py --json test.json --image-dir C:path\images\test  




import os
import argparse
import json
import torch
from torchvision import transforms
from PIL import Image
from evaluator import evaluation_model
from utils import labels_to_onehot
from tqdm import tqdm

# === Args ===
parser = argparse.ArgumentParser()
parser.add_argument("--json", type=str, required=True, help="Path to test.json or new_test.json")
parser.add_argument("--image-dir", type=str, required=True, help="Directory containing generated output images")
parser.add_argument("--objects", type=str, default="objects.json", help="Path to objects.json")
args = parser.parse_args()

# === Load label map ===
with open(args.objects, 'r') as f:
    label2idx = json.load(f)

# === Load label lists ===
with open(args.json, 'r') as f:
    label_lists = json.load(f)

# === Transform for images ===
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# === Load images and build batches ===
images = []
labels = []
print(f"Loading images from {args.image_dir}...")
for i, label in enumerate(tqdm(label_lists), 1):
    img_path = os.path.join(args.image_dir, f"output{i}.png")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = Image.open(img_path).convert("RGB")
    tensor_img = transform(img)
    images.append(tensor_img)

    onehot = labels_to_onehot(label, label2idx)
    labels.append(onehot)

images = torch.stack(images)
labels = torch.stack(labels)

# === Evaluate with pretrained classifier ===
evaluator = evaluation_model()
accuracy = evaluator.eval(images.cuda(), labels.cuda())
print(f"Accuracy: {accuracy:.4f}")

