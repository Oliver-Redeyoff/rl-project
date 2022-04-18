import os
from collections import deque
import pickle

import gym
import random
import numpy as np
import torch
from torch import nn
import torch.autograd as autograd
import torch.optim as optim
from torchvision.transforms import functional


class Cnn(nn.Module):

    def __init__(self, classes_count):
        super().__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.LeakyReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.LeakyReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.LeakyReLU(),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.LeakyReLU(),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(512), out_features=4096),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.LeakyReLU(),
            nn.Linear(in_features=4096, out_features=classes_count),
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(-1, 512)  # reduce the dimensions for linear layer input
        return self.classifier(x)

class ReplayBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add_replay(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:

    def __init__(self, actions):
        self.classes = actions
        self.net = Cnn(len(actions))
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=0.0001)
        self.replay_buffer = ReplayBuffer(1000)
        self.gamma = 0.9
        self.epsilon = 0.5
        self.MSE_loss = nn.MSELoss()

    def get_action(self, state, actions):
        if (len(state) == 4 and random.random() < self.epsilon):
            state = torch.Tensor(state)
            qvals = self.net(state)
            action = np.argmax(qvals.detach().numpy())
        else:
            action = random.randint(0, len(actions)-1)

        return action

    def compute_loss(self, experience):
        state, action, reward, next_state, done = experience
        state = torch.Tensor(state)
        next_state = torch.Tensor(next_state)

        curr_Q = self.net(state)
        next_Q = self.net(next_state)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = reward + self.gamma * max_next_Q

        loss = self.MSE_loss(curr_Q, expected_Q.detach())
        return loss

    def train(self, batch_size):
        if (len(self.replay_buffer.buffer) >= batch_size):
            batch = self.replay_buffer.sample(batch_size)
            for experience in batch:
                loss = self.compute_loss(experience)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

class Environment:

    def __init__(self):
        self.gym_env = gym.make('ALE/Pong-v5', frameskip=4) #, render_mode="human"
        self.gym_env.seed(0)
        self.actions = [i for i in range(0, self.gym_env.action_space.n)]
        self.state = []

    def reset(self):
        frame = self.gym_env.reset()
        self.add_frame_to_state(frame)

    def take_action(self, action):
        frame, reward, done, _ = self.gym_env.step(action)
        self.add_frame_to_state(frame)
        return (reward, done)

    def add_frame_to_state(self, frame):
        transposed = frame.transpose(2, 0, 1)
        as_tensor = torch.Tensor(transposed)
        grey = functional.rgb_to_grayscale(as_tensor)
        downsampled = functional.resize(grey, [110, 84])
        self.state.append(downsampled)
        if (len(self.state) > 4):
            self.state.pop(0)

    def get_state(self):
        return torch.stack(self.state).squeeze(1)
    

def main():

    if os.path.exists("rewards.txt"):
        os.remove("rewards.txt")

    env = Environment()
    agent = DQNAgent(env.actions)

    episode_count = 100
    for _ in range(episode_count):
        episode_reward = 0
        env.reset()

        while (True):
            action = agent.get_action(env.get_state(), env.actions)
            old_state = env.get_state()
            reward, done = env.take_action(action)
            
            episode_reward += reward
            if (len(old_state) == 4):
                agent.replay_buffer.add_replay(old_state, action, reward, env.get_state(), done)
                agent.train(1)

            if done:
                with open('rewards.txt', 'a') as f:
                    f.write(str(episode_reward))
                    f.write('\n')

                with open('agent.pkl', 'wb') as outp:
                    pickle.dump(agent, outp, pickle.HIGHEST_PROTOCOL)

                episode_reward = 0
                agent.epsilon *= 0.99
                break
    

if __name__ == '__main__':
    main()