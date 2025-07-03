
##  use this command to watch the demo 
# python LAB5_109208063_陳詠鑫_Code/test_task1_model.py --model-path "LAB5_109208063_task1_cartpole.pt" --output-dir ./videos --episodes 3




import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import imageio
import argparse
import os

# --- DQN Model from task1.py ---
class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.network(x)

# --- Evaluation Function ---
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    num_actions = env.action_space.n

    model = DQN(num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    all_frames = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0

        while not done:
            frame = env.render()
            all_frames.append(frame)

            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            obs = next_obs
            total_reward += reward

        print(f"[Episode {ep}] Total reward: {total_reward}")

  
    out_path = os.path.join(args.output_dir, "task1.mp4")
    with imageio.get_writer(out_path, fps=30) as video:
        for f in all_frames:
            video.append_data(f)

    print(f"🎬 Saved all episodes to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained CartPole model (.pt)")
    parser.add_argument("--output-dir", type=str, default="./cartpole_eval_video")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate(args)
