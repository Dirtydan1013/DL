
## Using the following command to generate picture
# python generate.py  


import os
import json
import torch
from torchvision.utils import save_image
from diffusers import UNet2DConditionModel, DDPMScheduler
from utils import labels_to_onehot
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "output"
MODEL_PATH = "checkpoints/model_epoch_050.pt"
NUM_CLASSES = 24
IMAGE_SIZE = 64

# Load conditional UNet
model = UNet2DConditionModel(
    sample_size=IMAGE_SIZE,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 256),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    cross_attention_dim=NUM_CLASSES
).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

scheduler = DDPMScheduler()
num_timesteps = scheduler.config.num_train_timesteps

# Load label mappings and test conditions
with open("objects.json", 'r') as f:
    label2idx = json.load(f)
with open("new_test.json", 'r') as f:
    condition_list = json.load(f)

# Create output dir
os.makedirs(SAVE_DIR, exist_ok=True)
target_dir = os.path.join(SAVE_DIR, "new_test_model_050")
os.makedirs(target_dir, exist_ok=True)

print(f"Generating {len(condition_list)} images...")
for i, labels in enumerate(tqdm(condition_list), 1):
    # Prepare condition for cross-attention
    cond = labels_to_onehot(labels, label2idx, num_classes=NUM_CLASSES).to(DEVICE)        
    cond_seq = cond.unsqueeze(0).unsqueeze(1)  # (1,1,24)

    # Initial random noise image
    image = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE)

    # Reverse diffusion
    for t in reversed(range(num_timesteps)):
        t_tensor = torch.tensor([t], device=DEVICE)
        # Predict noise
        with torch.no_grad():
            noise_pred = model(image, timestep=t_tensor, encoder_hidden_states=cond_seq).sample
        # Scheduler step accepts int timestep
        image = scheduler.step(noise_pred, int(t), image).prev_sample

    # Denormalize and save RGB channels
    out = (image.clamp(-1, 1) + 1) / 2
    save_image(out[..., 0:3, :, :], os.path.join(target_dir, f"output{i}.png"))

print(f"Saved images to {target_dir}")
