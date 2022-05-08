import random
from collections import deque

import gym
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Convolutional neural network our RL agent will use
class Cnn(nn.Module):
    def __init__(self, h, w, output_size):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4,  out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        convw, convh = self.conv2d_size_calc(w, h, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=3, stride=1)

        linear_input_size = convw * convh * 64  # Last conv layer's out sizes

        # action layer
        self.Alinear1 = nn.Linear(in_features=linear_input_size, out_features=128)
        self.Alrelu = nn.LeakyReLU()  # Linear 1 activation funct
        self.Alinear2 = nn.Linear(in_features=128, out_features=output_size)

        # state Value layer
        self.Vlinear1 = nn.Linear(in_features=linear_input_size, out_features=128)
        self.Vlrelu = nn.LeakyReLU()  # Linear 1 activation funct
        self.Vlinear2 = nn.Linear(in_features=128, out_features=1)  # Only 1 node

    def conv2d_size_calc(self, w, h, kernel_size=5, stride=2):
        # calcs conv layers output image sizes
        next_w = (w - (kernel_size - 1) - 1) // stride + 1
        next_h = (h - (kernel_size - 1) - 1) // stride + 1
        return next_w, next_h

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # flatten every batch

        Ax = self.Alrelu(self.Alinear1(x))
        Ax = self.Alinear2(Ax)  # no activation on last layer

        Vx = self.Vlrelu(self.Vlinear1(x))
        Vx = self.Vlinear2(Vx)  # no activation on last layer

        q = Vx + (Ax - Ax.mean())

        return q

# Deep reinforcement learning agent
class DQNAgent:
    def __init__(self, environment):
        # state size for breakout env. SS images (210, 160, 3). Used as input size in network
        self.state_size_h = environment.observation_space.shape[0]
        self.state_size_w = environment.observation_space.shape[1]
        self.state_size_c = environment.observation_space.shape[2]

        # activation size for breakout env. Used as output size in network
        self.action_size = environment.action_space.n

        self.gamma = 0.9
        self.alpha = 0.0001
        self.epsilon = 1

        # deque holds replay mem.
        self.memory = deque(maxlen=50000)

        # create two model for DDQN algorithm
        self.online_model = Cnn(h=80, w=64, output_size=self.action_size).to(DEVICE)
        self.target_model = Cnn(h=80, w=64, output_size=self.action_size).to(DEVICE)
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.online_model.parameters(), lr=self.alpha)

    def preProcess(self, image):
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale image
        frame = frame[20:self.state_size_h, 0:self.state_size_w]  # crop 20px from top of image
        frame = cv2.resize(frame, (64, 80))  # resize image
        frame = frame.reshape(64, 80) / 255  # normalize image

        return frame

    def chooseAction(self, state):        
        # decide if we should explore
        explore = random.uniform(0, 1) <= self.epsilon

        if explore:
            action = random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float, device=DEVICE).unsqueeze(0)
                q_values = self.online_model.forward(state)  # (1, action_size)
                action = torch.argmax(q_values).item()  # Returns the indices of the maximum value of all elements

        return action

    def train(self):
        # only start training once we have a certain amount of replay memory
        if len(self.memory) < 40000:
            loss, max_q = [0, 0]
            return loss, max_q
        
        # we get out minibatch and turn it to numpy array
        state, action, reward, next_state, done = zip(*random.sample(self.memory, 64))

        # concat batches in one array
        # (np.arr, np.arr) ==> np.BIGarr
        state = np.concatenate(state)
        next_state = np.concatenate(next_state)

        # convert them to tensors
        state = torch.tensor(state, dtype=torch.float, device=DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float, device=DEVICE)
        action = torch.tensor(action, dtype=torch.long, device=DEVICE)
        reward = torch.tensor(reward, dtype=torch.float, device=DEVICE)
        done = torch.tensor(done, dtype=torch.float, device=DEVICE)

        # make predictions
        state_q_values = self.online_model(state)
        next_states_q_values = self.online_model(next_state)
        next_states_target_q_values = self.target_model(next_state)

        # find selected action's q_value
        selected_q_value = state_q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # get indice of the max value of next_states_q_values
        # use that indice to get a q_value from next_states_target_q_values
        # we use greedy for policy So it called off-policy
        next_states_target_q_value = next_states_target_q_values.gather(1, next_states_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        # use Bellman function to find expected q value
        expected_q_value = reward + self.gamma * next_states_target_q_value * (1 - done)

        # calculate loss with expected_q_value and q_value
        loss = (selected_q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, torch.max(state_q_values).item()

    def storeResults(self, state, action, reward, nextState, done):
        self.memory.append([state[None, :], action, reward, nextState[None, :], done])

    def decayEpsilon(self):
        if self.epsilon > 0.05:
            self.epsilon *= 0.99

# Main method
def main():
    environment = gym.make("Pong-v4")
    agent = DQNAgent(environment)

    last_100_ep_reward = deque(maxlen=100)
    total_steps = 0
    for episode in range(1000):
        frame = agent.preProcess(environment.reset())
        state = np.stack((frame, frame, frame, frame))

        total_max_q_val = 0
        total_reward = 0
        total_loss = 0
        for step in range(10000):
            # get action and perform it
            action = agent.chooseAction(state)
            raw_frame, reward, done, _ = environment.step(action)

            # stack state . Every state contains 4 time contionusly frames
            # we stack frames like 4 channel image
            frame = agent.preProcess(raw_frame)
            next_state = np.stack((frame, state[0], state[1], state[2]))

            # store the experience in memory
            agent.storeResults(state, action, reward, next_state, done)

            # train agent
            loss, max_q_val = agent.train()

            total_loss += loss
            total_max_q_val += max_q_val
            total_reward += reward
            total_steps += 1
            if total_steps % 1000 == 0:
                agent.decayEpsilon()

            if done:
                # update target model
                agent.target_model.load_state_dict(agent.online_model.state_dict())

                # update stats
                last_100_ep_reward.append(total_reward)
                avg_max_q_val = total_max_q_val / step

                outStr = "Episode:{} Reward:{:.2f} Loss:{:.2f} Last_100_Avg_Rew:{:.3f} Avg_Max_Q:{:.3f} Epsilon:{:.2f} Step:{} CStep:{}".format(
                episode, total_reward, total_loss, np.mean(last_100_ep_reward), avg_max_q_val, agent.epsilon, step, total_steps
                )
                print(outStr)

                break
            else:
                # move to the next state
                state = next_state

# Entry point
if __name__ == '__main__':
    main()