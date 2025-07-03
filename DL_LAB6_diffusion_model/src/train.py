
## Only need to perform python train.py 
## The hyper parameters is for continued training on checkpoint. 


# train.py
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel, DDPMScheduler
from dataset import ICLEVRDataset
from tqdm import tqdm
import argparse

# === Argument parser ===
parser = argparse.ArgumentParser(description="Train Conditional DDPM with cross-attention conditioning and resume capability")
parser.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")
parser.add_argument("--epochs", type=int, default=200,
                    help="Total number of epochs to train")
parser.add_argument("--batch-size", type=int, default=64,
                    help="Batch size for training")
args = parser.parse_args()

# === Config ===
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
SAVE_DIR = "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 24
IMAGE_SIZE = 64

# === Prepare dataset ===
dataset = ICLEVRDataset(
    image_dir="iclevr",
    json_path="train.json",
    object_json_path="objects.json",
    image_size=IMAGE_SIZE,
    device=DEVICE
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
steps_per_epoch = len(dataloader)

# === Prepare model and scheduler ===
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

scheduler = DDPMScheduler()
num_timesteps = scheduler.config.num_train_timesteps
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# === Resume from checkpoint ===
start_epoch = 1
if args.resume:
    if os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Start Training on {DEVICE}, from epoch {start_epoch}/{EPOCHS}")

for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    pbar = tqdm(dataloader, total=steps_per_epoch, desc=f"Epoch {epoch}/{EPOCHS}")
    for step, (x_0, cond) in enumerate(pbar, 1):
        B = x_0.size(0)
        t = torch.randint(0, num_timesteps, (B,), device=DEVICE).long()
        noise = torch.randn_like(x_0)
        x_t = scheduler.add_noise(x_0, noise, t)

        # cross-attention conditioning
        cond_seq = cond.unsqueeze(1)  # (B, 1, 24)
        pred_noise = model(x_t, timestep=t, encoder_hidden_states=cond_seq).sample

        loss = F.mse_loss(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        avg_loss = total_loss / (step * B)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    epoch_loss = total_loss / len(dataset)
    elapsed = time.time() - start_time
    print(f"Epoch {epoch}/{EPOCHS} - Loss: {epoch_loss:.4f} - Time: {elapsed:.1f}s")

    ckpt_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch:03}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, ckpt_path)

print("Training completed.")

