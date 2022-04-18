from collections import deque

import gym
import random
import numpy as np
import torch
from torch import nn
import torch.autograd as autograd
import torch.optim as optim
random.seed(0)


def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    return env.render(mode='rgb_array').transpose((2, 0, 1))


class Cnn(nn.Module):

    def __init__(self, classes_count):
        super().__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
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
            nn.Linear(in_features=(256 * 5 * 3), out_features=4096),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.LeakyReLU(),
            nn.Linear(in_features=4096, out_features=classes_count),
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(-1, 256 * 5 * 3)  # reduce the dimensions for linear layer input
        return self.classifier(x)

class ReplayBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add_replay(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:

    def __init__(self, actions):
        self.classes = actions
        self.net = Cnn(len(actions))
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=0.0001)
        self.replay_buffer = ReplayBuffer(1000)
        self.gamma = 0.99
        self.epsilon = 0.5
        self.MSE_loss = nn.MSELoss()

    def get_action(self, state):
        state = autograd.Variable(torch.from_numpy(state).float().unsqueeze(0))
        qvals = self.net(state)
        if (random.random() < self.epsilon):
            action = np.argmax(qvals.detach().numpy())
            print("Choosed best action {}".format(action))
        else:
            action = random.randint(0, len(qvals.tolist()[0])-1)
            print("Choosed random action {}".format(action))

        return action

    def compute_loss(self, batch):
        b_states, b_actions, b_rewards, b_next_states, b_dones = batch
        states = torch.FloatTensor(b_states)
        actions = torch.LongTensor(b_actions)
        rewards = torch.FloatTensor(b_rewards)
        next_states = torch.FloatTensor(b_next_states)
        dones = torch.FloatTensor(b_dones)

        curr_Q = self.net.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.net.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + (1 - dones) * self.gamma * max_next_Q

        loss = self.MSE_loss(curr_Q.clone(), expected_Q.detach())
        return loss

    def train(self, batch_size):
        if (len(self.replay_buffer.buffer) >= batch_size):
            batch = self.replay_buffer.sample(batch_size)
            loss = self.compute_loss(batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def main():
    env = gym.make('Pong-v0') #, render_mode="human"
    env.seed(0)
    actions = [i for i in range(0, env.action_space.n)]
    agent = DQNAgent(actions)

    episode_count = 100
    rewards = []
    for _ in range(episode_count):
        episode_reward = 0
        env.reset()
        current_state = get_screen(env)

        while (True):
            action = agent.get_action(current_state)
            old_state = current_state
            _, reward, done, _ = env.step(action)
            current_state = get_screen(env)
            episode_reward += reward

            agent.replay_buffer.add_replay(old_state, action, reward, current_state, done)
            agent.train(4)

            print('Took action {} and got reward {}'.format(action, reward))

            if done:
                print('Episode done, rewards from episode: %s' % episode_reward)
                rewards.append(episode_reward)
                break
    


if __name__ == '__main__':
    main()