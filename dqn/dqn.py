import os
from collections import deque
import pickle
import time

import gym
import random
import numpy as np
import torch
from torch import nn
import torch.autograd as autograd
import torch.optim as optim
from torchvision.transforms import functional


class MemoryNetwork(nn.Module):

    def __init__(self, classes_count):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=classes_count),
        )

    def forward(self, x):
        # x = self.conv_net(x)
        # x = x.view(-1, 512)  # reduce the dimensions for linear layer input
        return self.classifier(x)

class ImageNetwork(nn.Module):
    def __init__(self, classes_count) -> None:
        super().__init__()

        self._cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self._action_network = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=classes_count),
        )

    @classmethod
    def preprocess_image(cls, image):
        """Prepare an image for input to the network"""
        # print(type(image))
        transposed = image.transpose(1, 3)
        # print(transposed.shape)
        # grey = functional.rgb_to_grayscale(transposed)
        # downsampled = functional.resize(grey, [110, 84])

        return transposed
        return image

    def forward(self, input_image):
        convoluted_image = self._cnn(input_image)

        # print("Convoluted: ", convoluted_image.shape)

        new_view = convoluted_image.view(-1, 512)  # reduce the dimensions for linear layer input
        return self._action_network(new_view)

class ReplayBuffer:

    def __init__(self, max_size):
        # self.max_size = max_size
        self.buffer = deque()

    def add_replay(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, actions, epsilon=0.5, gamma=0.9):
        self.classes = actions
        self.net = MemoryNetwork(len(self.classes))
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=0.0001)
        self.replay_buffer = ReplayBuffer(1000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.MSE_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss()

    def get_action(self, state, actions):
        should_explore = random.random() < self.epsilon

        if should_explore:
            return random.choice(actions)

        result = self.net(state)
        reduced = result.sum(dim=0)
        selected_action = torch.argmax(reduced)

        return selected_action

    def compute_loss(self, experience):
        state, action, reward, next_state, done = experience
        reward = torch.Tensor([reward])

        if done:
            y_j = reward
        
        else:
            state = torch.Tensor(state)
            next_state = torch.Tensor(next_state)
            result = self.net(next_state)
            max_next_reward = torch.max(result.sum(dim=0))

            y_j = reward + self.gamma * max_next_reward

        loss = self.huber_loss(y_j, reward)
        return loss

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        self.optimizer.zero_grad()

        for experience in batch:
            loss = self.compute_loss(experience)
            if loss.requires_grad:
                # loss.retain_grad()
                loss.backward()

        self.optimizer.step()

class ImageDQNAgent(DQNAgent):

    def __init__(self, actions, epsilon=0.5, gamma=0.9):
        self.classes = actions
        self.net = ImageNetwork(len(actions))
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=0.0001)
        self.replay_buffer = ReplayBuffer(1000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.MSE_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss()

    def get_action(self, state, actions):
        should_explore = random.random() < self.epsilon

        if should_explore:
            return random.choice(actions)

        result = self.net(state)
        reduced = result.sum(dim=0)
        selected_action = torch.argmax(reduced)

        return selected_action

    def compute_loss(self, experience):
        state, action, reward, next_state, done = experience
        state = torch.Tensor(state)
        next_state = torch.Tensor(next_state)

        reward = torch.Tensor([reward])

        if done:
            y_j = reward
        
        else:
            result = self.net(next_state)
            max_next_reward = torch.max(result.sum(dim=0))

            y_j = reward + self.gamma * max_next_reward

        loss = self.huber_loss(y_j, reward)
        return loss

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        for experience in batch:
            loss = self.compute_loss(experience)
            if loss.requires_grad:
                self.optimizer.zero_grad()
                loss.retain_grad()
                loss.backward()
                self.optimizer.step()

class Environment:

    def __init__(self, frame_skip, environment_name='ALE/Pong-ram-v5'):
        self.frame_skip = frame_skip
        self.gym_env = gym.make(environment_name) #, render_mode="human"
        self.gym_env.seed(0)
        self.actions = [i for i in range(self.gym_env.action_space.n)]
        self.state = []

    def reset(self):
        """Reset for new episode"""
        self.state = []

        frame = self.gym_env.reset()
        self.add_frame_to_state(frame)

    def take_action(self, action):
        frame, reward, done, _ = self.gym_env.step(action)
        self.add_frame_to_state(frame)
        return (reward, done)

    def add_frame_to_state(self, frame):
        frame = torch.Tensor(frame)
        self.state.append(frame)

        # Keep state length at 4
        if (len(self.state) > 4):
            self.state.pop(0)

    def get_state(self):
        return torch.stack(self.state).squeeze(1)

    
    def _pre_process(self, frame):
        transposed = frame.transpose(2, 0, 1)
        as_tensor = torch.Tensor(transposed)
        grey = functional.rgb_to_grayscale(as_tensor)
        downsampled = functional.resize(grey, [110, 84])

        return downsampled

def image_main(frame_skip=4, batch_size=4, num_episodes=1000):
    if os.path.exists("rewards_image.txt"):
        os.remove("rewards_image.txt")

    env = Environment(frame_skip, environment_name='ALE/Pong-v5')
    agent = ImageDQNAgent(env.actions)

    for _ in range(int(num_episodes)):
        episode_reward = 0
        env.reset()

        done = False
        while not done:
            current_state = env.get_state()
            # print(current_state.shape)
            current_state = ImageNetwork.preprocess_image(current_state)
            action = agent.get_action(current_state, env.actions)
            reward, done = env.take_action(action)
            
            episode_reward += reward

            if (len(current_state) == frame_skip):
                next_state = env.get_state()
                next_state = ImageNetwork.preprocess_image(next_state)
                agent.replay_buffer.add_replay(current_state, action, reward, next_state, done)
                agent.train(batch_size)

            if done:
                print(episode_reward)
                with open('rewards_image.txt', 'a') as f:
                    f.write(str(episode_reward))
                    f.write('\n')

                with open('agent_image.pkl', 'wb') as outp:
                    pickle.dump(agent, outp, pickle.HIGHEST_PROTOCOL)

                episode_reward = 0
                agent.epsilon *= 0.99


def main(frame_skip=4, batch_size=4, num_episodes=1000):

    if os.path.exists("rewards.txt"):
        os.remove("rewards.txt")

    env = Environment(frame_skip)
    agent = DQNAgent(env.actions, epsilon=0.75, gamma=0.2)

    for episode_num in range(int(num_episodes)):
        start_time = time.perf_counter()
        episode_reward = 0
        env.reset()

        done = False
        while not done:
            current_state = env.get_state()
            action = agent.get_action(current_state, env.actions)
            reward, done = env.take_action(action)
            
            episode_reward += reward

            if (len(current_state) == frame_skip):
                agent.replay_buffer.add_replay(current_state, action, reward, env.get_state(), done)
                agent.train(batch_size)

            if done:
                end_time = time.perf_counter()
                print(f"episode: {episode_num+1} reward {episode_reward} time taken: {end_time-start_time}")
                with open('rewards.txt', 'a') as f:
                    f.write(str(episode_reward))
                    f.write('\n')

                if episode_num % 50 == 0:
                    with open('agent.pkl', 'wb') as outp:
                        pickle.dump(agent, outp, pickle.HIGHEST_PROTOCOL)

                if episode_num % 10:
                    agent.epsilon *= 0.9

if __name__ == '__main__':
    main(num_episodes=2e7)


# class DQNAgent:

#     def __init__(self, actions):
#         self.classes = actions
#         self.net = Cnn(len(actions))
#         self.optimizer = optim.Adam(params=self.net.parameters(), lr=0.0001)
#         self.replay_buffer = ReplayBuffer(1000)
#         self.gamma = 0.9
#         self.epsilon = 0.5
#         self.MSE_loss = nn.MSELoss()

#     def get_action(self, state, actions):
#         print("state shape: ", state.shape)
#         if (len(state) == 4 and random.random() < self.epsilon):
#             state = torch.Tensor(state)
#             qvals = self.net(state)
#             print("qvals: ", qvals)
#             action = np.argmax(qvals.detach().numpy())
#         else:
#             action = random.choice(actions) # (0, len(actions)-1)

#         return action

#     def compute_loss(self, experience):
#         state, action, reward, next_state, done = experience
#         state = torch.Tensor(state)
#         next_state = torch.Tensor(next_state)

#         curr_Q = self.net(state)
#         next_Q = self.net(next_state)
#         max_next_Q = torch.max(next_Q, 1)[0]
#         expected_Q = reward + self.gamma * max_next_Q

#         print(curr_Q.shape)
#         print(next_Q.shape)
#         print(next_Q)
#         print(expected_Q.shape)

#         loss = self.MSE_loss(curr_Q, next_Q.detach())
#         return loss

#     def train(self, batch_size):
#         if (len(self.replay_buffer.buffer) >= batch_size):
#             batch = self.replay_buffer.sample(batch_size)
#             for experience in batch:
#                 loss = self.compute_loss(experience)
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()