
## perform python denoise.py to get denoising process


import os
import json
import torch
from torchvision.utils import make_grid, save_image
from diffusers   import UNet2DConditionModel, DDPMScheduler
from utils       import labels_to_onehot

# 配置
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT  = "checkpoints/model_epoch_050.pt"  # 或指定 epoch_ckpt
NUM_CLASSES = 24
IMAGE_SIZE  = 64
K           = 8

# 載入模型
model = UNet2DConditionModel(
    sample_size=IMAGE_SIZE,
    in_channels=3,
    out_channels=3,
    layers_per_block=(2, 2, 2),
    block_out_channels=(64, 128, 256),
    down_block_types=("DownBlock2D","DownBlock2D","AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D","UpBlock2D","UpBlock2D"),
    cross_attention_dim=NUM_CLASSES
).to(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

scheduler     = DDPMScheduler()
num_timesteps = scheduler.config.num_train_timesteps

with open("objects.json") as f:
    label2idx = json.load(f)

#固定條件
denoise_labels = ["red sphere","cyan cylinder","cyan cube"]
cond = labels_to_onehot(denoise_labels, label2idx, num_classes=NUM_CLASSES).to(DEVICE)
cond_seq = cond.unsqueeze(0).unsqueeze(1)  # (1,1,24)

# 初始噪聲
img = torch.randn((1,3,IMAGE_SIZE,IMAGE_SIZE), device=DEVICE)
frames = []
interval = num_timesteps // K

# 反向擴散
for t in reversed(range(num_timesteps)):
    t_int    = int(t)
    t_tensor = torch.tensor([t_int], device=DEVICE)
    with torch.no_grad():
        noise_pred = model(img, timestep=t_tensor, encoder_hidden_states=cond_seq).sample
    img = scheduler.step(noise_pred, t_int, img).prev_sample
    if t_int % interval == 0:
        out = (img.clamp(-1,1) + 1) / 2
        frames.append(out.squeeze(0))  # (3,64,64)

# save
grid = make_grid(torch.stack(frames, dim=0), nrow=K, padding=2)
save_image(grid, os.path.join("output", "denoise_process.png"))
print("Saved output/denoise_process.png")
