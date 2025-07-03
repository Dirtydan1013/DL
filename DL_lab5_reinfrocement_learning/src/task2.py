

##  use this command to run this code
# python task2.py --wandb-run-name "pong-train-from-scratch" --num-episodes 5000 --batch-size 64 --memory-size 100000 --lr 0.0001 --discount-factor 0.99 --epsilon-start 1.0 --epsilon-min 0.05 
# --target-update-frequency 10000 --replay-start-size 50000 --max-episode-steps 10000 --train-per-step 1 --seed 42 --epsilon-decay-steps 1000000 --train-per-step 4



import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
import wandb
import ale_py
from collections import deque
import cv2

# --- Utility: set random seed ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- CNN DQN for Pong ---
class CNN_DQN(nn.Module):
    def __init__(self, num_actions):
        super(CNN_DQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.fc(x)
        return x

# --- Atari Preprocessor ---
class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

# --- Prioritized Replay Buffer ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_init = beta  # initial beta for annealing
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def anneal_beta(self, current_step, max_steps):
        self.beta = min(1.0, self.beta_init + (1.0 - self.beta_init) * current_step / max_steps)

    def add(self, state, action, reward, next_state, done):
        reward = np.clip(reward, -1.0, 1.0)
        max_prio = self.priorities.max() if self.buffer else 1.0
        data = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            weights,
            indices
        )

    def update_priorities(self, indices, errors):
        errors = np.abs(errors) + 1e-6
        self.priorities[indices] = errors

# --- DQNAgent for Pong ---
class DQNAgent:
    def __init__(self, args):
        gym.register_envs(ale_py)
        self.seed = args.seed
        self.env = gym.make("ALE/Pong-v5", render_mode=None)
        self.test_env = gym.make("ALE/Pong-v5", render_mode=None)
        self.env.reset(seed=self.seed)
        self.test_env.reset(seed=self.seed)

        self.num_actions = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.q_net = CNN_DQN(self.num_actions).to(self.device)
        self.target_net = CNN_DQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.memory = PrioritizedReplayBuffer(args.memory_size)
        self.preprocessor = AtariPreprocessor(frame_stack=4)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay_steps = args.epsilon_decay_steps
        self.epsilon_min = args.epsilon_min
        self.target_update_freq = args.target_update_frequency
        self.train_start = args.replay_start_size
        self.max_episode_steps = args.max_episode_steps
        self.train_per_step = args.train_per_step

        self.total_steps = 0
        self.best_eval_reward = -float('inf')
        os.makedirs(args.save_dir, exist_ok=True)
        self.save_dir = args.save_dir

        wandb.init(project="DLP-Lab5-Pong", name=args.wandb_run_name, config=vars(args))

        if args.load_model_path:
            self.q_net.load_state_dict(torch.load(args.load_model_path))
            self.target_net.load_state_dict(self.q_net.state_dict())
            print(f"Loaded model from {args.load_model_path}")

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state)
        return q_values.argmax().item()

    def train(self):
        if len(self.memory.buffer) < self.train_start:
            return

        # Anneal beta
        self.memory.anneal_beta(self.total_steps, self.epsilon_decay_steps)

        states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q_values

        # 使用 Huber Loss
        loss = nn.functional.smooth_l1_loss(q_values, target_q, reduction='none')
        loss = (loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)

        self.optimizer.step()

        # Logging
        wandb.log({
            "loss": loss.item(),
            "Q_max": q_values.max().item(),
            "Q_min": q_values.min().item(),
            "Q_mean": q_values.mean().item(),
            "beta": self.memory.beta,  # 可視化 beta 變化
        }, step=self.total_steps)

        self.memory.update_priorities(indices, (q_values - target_q).abs().detach().cpu().numpy())

        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def run(self, num_episodes=10000):
        for episode in range(num_episodes):
            obs, _ = self.env.reset(seed=self.seed)
            state = self.preprocessor.reset(obs)
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.preprocessor.step(next_obs)
                done = terminated or truncated
                self.memory.add(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                for _ in range(self.train_per_step):
                    self.train()

                self.total_steps += 1

            decay_rate = (1.0 - self.epsilon_min) / self.epsilon_decay_steps
            self.epsilon = max(self.epsilon_min, 1.0 - decay_rate * self.total_steps)


            print(f"Episode {episode} - Total Reward: {total_reward} - Epsilon: {self.epsilon:.3f} - Steps: {self.total_steps}")
            wandb.log({
                "Episode": episode,
                "Total Reward": total_reward,
                "Epsilon": self.epsilon,
                "Env Steps": self.total_steps
            })

            if episode % 50 == 0:
                self.evaluate()

def evaluate(self, episodes=10):
    avg_reward = 0
    seeds_used = []
    for _ in range(episodes):
        eval_seed = random.randint(0, 999999)
        seeds_used.append(eval_seed)
        obs, _ = self.test_env.reset(seed=eval_seed)
        state = self.preprocessor.reset(obs)
        total_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            next_state = self.preprocessor.step(next_obs)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        avg_reward += total_reward

    avg_reward /= episodes
    wandb.log({"Evaluation Avg Reward (multi-seed)": avg_reward}, step=self.total_steps)
    print(f"[Evaluate] Multi-seed avg reward: {avg_reward:.2f} using seeds {seeds_used}")

    if avg_reward > self.best_eval_reward:
        self.best_eval_reward = avg_reward
        model_path = os.path.join(self.save_dir, f"pong_best_model_multiseed_{self.total_steps}.pt")
        torch.save(self.q_net.state_dict(), model_path)
        print(f" New best model (multi-seed) saved with avg reward {avg_reward:.2f}")

# --- main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-model-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="pong-task2-run")
    parser.add_argument("--num-episodes", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay-steps", type=int, default=1_000_000)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    agent = DQNAgent(args)
    agent.run(num_episodes=args.num_episodes)
    agent.evaluate(episodes=10)
