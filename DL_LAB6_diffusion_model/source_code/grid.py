#### Using the following command to generate grid

## python grid.py --json new_test.json --image-dir output/new_test_model_050 --output output/grid
## python grid.py --json test.json --image-dir output/test_model_050 --output output/grid        



import os
import json
import torch
import argparse
from torchvision.utils import make_grid, save_image
from torchvision.io    import read_image
from tqdm              import tqdm

parser = argparse.ArgumentParser(description="Generate image grid from specified images and JSON labels")
parser.add_argument("--json",       type=str, required=True, help="Path to JSON file (test.json or new_test.json)")
parser.add_argument("--image-dir",  type=str, required=True, help="Directory containing generated output images")
parser.add_argument("--output",     type=str, default="output", help="Directory to save grid image")
args = parser.parse_args()

# Load conditions
cond_list = json.load(open(args.json, 'r'))
# Determine split name from JSON filename
split_name = os.path.splitext(os.path.basename(args.json))[0]

# Verify image directory
img_dir = args.image_dir
assert os.path.isdir(img_dir), f"Directory not found: {img_dir}. Run generation first."

# Read all images
imgs = []
for i in range(1, len(cond_list) + 1):
    path = os.path.join(img_dir, f"output{i}.png")
    assert os.path.exists(path), f"Image not found: {path}"
    img = read_image(path).float() / 255.0  # [3,H,W] -> [0,1]
    imgs.append(img)

# Stack and make grid
batch = torch.stack(imgs, dim=0)           # (N,3,H,W)
grid  = make_grid(batch, nrow=8, padding=2)

# Save
os.makedirs(args.output, exist_ok=True)
out_path = os.path.join(args.output, f"grid_{split_name}.png")
save_image(grid, out_path)
print(f"Saved grid image: {out_path}")

