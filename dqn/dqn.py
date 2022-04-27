import os
from collections import deque
import pickle
import time

import gym
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torchvision.transforms import functional

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageNetwork(nn.Module):
    def __init__(self, classes_count, in_channels) -> None:
        super().__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=4, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2160, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=classes_count)
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(-1, 2160)  # reduce the dimensions for linear layer input
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

    def __init__(self, actions, batch_size, in_channels, epsilon=0.5, gamma=0.9, buffer_size=10000):
        self.classes = actions
        self.net = Cnn(len(actions)).to(DEVICE)
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=0.0001)
        self.replay_buffer = ReplayBuffer(10000)
        self.gamma = 0.9
        self.epsilon = 0.5
        self.MSE_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss()

    def get_action(self, state, actions):
        should_explore = random.random() < self.epsilon

        if should_explore:
            return random.choice(actions)

        # print("state shape: ", state.shape)
        result = self.net(state)
        # print("result: ", result)
        reduced = result.sum(dim=0)
        selected_action = torch.argmax(reduced)

        return selected_action

    def compute_loss(self, experience):
        print("===")

        state, action, reward, next_state, done = experience
        state = torch.Tensor(state)
        next_state = torch.Tensor(next_state)
        reward = torch.Tensor([reward])

        expected_result = self.net(state)
        q_value = expected_result[0][action]

        if done:
            y_j = reward
        
        else:
            result = self.net(next_state)
            max_next_reward = torch.max(result.sum(dim=0))

            y_j = reward + self.gamma * max_next_reward

        loss = self.MSE_loss(q_value, y_j)
        return loss

    def train(self, batch_size):
        if (len(self.replay_buffer.buffer) >= batch_size):
            batch = self.replay_buffer.sample(batch_size)
            for experience in batch:
                self.optimizer.zero_grad()
                loss = self.compute_loss(experience)
                loss.backward()

        self.optimizer.step()

class Environment:

    def __init__(self, frame_skip, environment_name='ALE/Pong-v5'):
        self.frame_skip = frame_skip
        self.gym_env = gym.make(environment_name, frameskip=frame_skip, full_action_space=False) #, render_mode="human"
        self.gym_env.seed(0)
        self.actions = [i for i in range(self.gym_env.action_space.n)]
        self.state = []

    def reset(self):
        """Reset for new episode"""

        frame = self.gym_env.reset()
        frame = ImageNetwork.preprocess_image(frame)
        return frame

    def take_action(self, action):
        frame, reward, done, _ = self.gym_env.step(action)
        frame = ImageNetwork.preprocess_image(frame)
        return (frame, reward, done)

    def add_frame_to_state(self, frame):
        transposed = frame.transpose(2, 0, 1)
        as_tensor = torch.Tensor(transposed)
        grey = functional.rgb_to_grayscale(as_tensor)
        downsampled = functional.resize(grey, [110, 84])
        thresh = nn.Threshold(87.3, 0)
        background_removed = thresh(downsampled)
        self.state.append(background_removed)
        if (len(self.state) > 4):
            self.state.pop(0)

    def get_state(self):
        raise NotImplementedError
        return torch.stack(self.state)

def main(frame_skip=4, batch_size=4, num_episodes=1000):
    if os.path.exists("rewards_image.txt"):
        os.remove("rewards_image.txt")

    env = Environment(frame_skip, environment_name='ALE/Pong-v5')
    agent = DQNAgent(env.actions, batch_size, env.gym_env.observation_space.shape[2], epsilon=1, gamma=0.99)

    env = Environment()
    agent = DQNAgent(env.actions)

    episode_count = 100000
    for episode_num in range(episode_count):
        episode_reward = 0
        start_time = time.perf_counter()
        current_state = env.reset()

        batch = Batch(batch_size)

        done = False
        while not done:

            # print("current state: ", current_state.shape)
            action = agent.get_action(current_state, env.actions)
            next_state, reward, done = env.take_action(action)
            
            episode_reward += reward

            if done:
                print("Finished episode {} with reward: {}".format(episode_num, episode_reward))

                with open('rewards.txt', 'a') as f:
                    f.write(str(episode_reward))
                    f.write('\n')

            agent.train()

            current_state = next_state

        print(f"episode: {episode_num+1} reward: {episode_reward} time taken: {time.perf_counter()-start_time} epsilon: {agent.epsilon}")
        
        with open('rewards.txt', 'a') as f:
            f.write(str(episode_reward))
            f.write('\n')

        if episode_num % 50 == 0:
            with open('agent.pkl', 'wb') as outp:
                pickle.dump(agent, outp, pickle.HIGHEST_PROTOCOL)

        agent.epsilon = max(0.05, agent.epsilon * 0.99)


if __name__ == '__main__':
    main(num_episodes=10000)


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