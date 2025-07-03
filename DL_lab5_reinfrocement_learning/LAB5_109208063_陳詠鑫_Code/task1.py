
##  use this command to run this code
# python task1.py --wandb-run-name "cartpole-better" --epsilon-decay 0.995 --lr 0.0005 --batch-size 64 --replay-start-size 1000 --num-episodes 2000


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
from collections import deque
import argparse
import wandb
import os

# ----- Q-network 定義 -----
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

# ----- DQN Agent 定義 -----
class DQNAgent:
    def __init__(self, args, env_name="CartPole-v1"):
        self.env = gym.make(env_name)
        self.test_env = gym.make(env_name)
        self.num_actions = self.env.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.q_net = DQN(self.num_actions).to(self.device)
        self.target_net = DQN(self.num_actions).to(self.device)
        if args.load_model_path is not None:
            print(f"Loading model from {args.load_model_path}")
            self.q_net.load_state_dict(torch.load(args.load_model_path))
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.memory = deque(maxlen=args.memory_size)
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.target_update_freq = args.target_update_frequency
        self.train_start = args.replay_start_size
        self.max_episode_steps = args.max_episode_steps
        self.train_per_step = args.train_per_step

        self.total_steps = 0

        wandb.init(project="DLP-Lab5-CartPole", name=args.wandb_run_name, config=vars(args))

        os.makedirs(args.save_dir, exist_ok=True)
        self.save_dir = args.save_dir
        self.best_eval_reward = -float('inf')

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state)
        return q_values.argmax().item()

    def train(self):
        if len(self.memory) < self.train_start:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def run(self, num_episodes=500):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.memory.append((state, action, reward, next_state, done))

                state = next_state
                total_reward += reward

                for _ in range(self.train_per_step):
                    self.train()

                self.total_steps += 1

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            print(f"Episode {episode} - Total Reward: {total_reward} - Epsilon: {self.epsilon:.3f}")

            wandb.log({
                "Episode": episode,
                "Total Reward": total_reward,
                "Epsilon": self.epsilon,
                "Env Steps": self.total_steps
            })

            if episode % 20 == 0:
                self.evaluate()

    def evaluate(self, episodes=10):
        avg_reward = 0
        for _ in range(episodes):
            state, _ = self.test_env.reset()
            done = False
            total_reward = 0
            while not done:
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.q_net(state).argmax().item()
                next_state, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = next_state
            avg_reward += total_reward
        avg_reward /= episodes
        print(f"Average Evaluation Reward over {episodes} episodes: {avg_reward}")
        wandb.log({"Evaluation Avg Reward": avg_reward})

        if avg_reward > self.best_eval_reward:
            self.best_eval_reward = avg_reward
            model_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(self.q_net.state_dict(), model_path)
            print(f"New best model saved with average reward {avg_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-model-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--num-episodes", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    args = parser.parse_args()

    agent = DQNAgent(args)
    agent.run(num_episodes=args.num_episodes)
    agent.evaluate(episodes=10)
