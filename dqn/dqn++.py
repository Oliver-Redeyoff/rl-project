from collections import namedtuple, deque
import random
import math
import time


import gym
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class Epsilon:
    def __init__(self, max_eps, min_eps, decay) -> None:
        self.max_eps = max_eps
        self.min_eps = min_eps
        self._eps = max_eps
        self.decay = decay

    @property
    def eps(self):
        to_return = self._eps
        self._step()
        return to_return

    def _step(self):
        self._eps = self.min_eps + (self.max_eps - self.min_eps) * math.exp(-1 / self.decay)

class ReplayMemory:
    def __init__(self, capacity) -> None:
        self._memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self._memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self._memory, batch_size)

    def __len__(self):
        return len(self._memory)  

class DQNNetwork(nn.Module):
    def __init__(self, image_height, image_width, num_actions) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(image_width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(image_height)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, num_actions)

    def forward(self, x) -> torch.Tensor:
        x.to(DEVICE)
        x = functional.relu(self.bn1(self.conv1(x)))
        x = functional.relu(self.bn2(self.conv2(x)))
        x = functional.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class DQNAgent:
    def __init__(self, gamma, epsilon, num_actions, batch_size) -> None:
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_actions = num_actions
        self.batch_size = batch_size

        self.policy_network = DQNNetwork(210, 160, num_actions)
        self.target_network = DQNNetwork(210, 160, num_actions)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.optimiser = optim.Adam(self.policy_network.parameters(), lr=0.001)
        self.memory = ReplayMemory(10000)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=DEVICE, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=DEVICE)
        next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimiser.zero_grad()
        loss.backward()
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimiser.step()


    def get_action(self, state):
        should_explore = random.random() < self.epsilon.eps

        if should_explore:
            return torch.tensor([[random.randrange(self.num_actions)]], device=DEVICE, dtype=torch.long)

        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return self.policy_network(state).max(1)[1].view(1, 1)

def preprocess_image(image):
    tensored = torch.FloatTensor(image).transpose(2, 0)
    return tensored.unsqueeze(0)

def main(num_episodes, batch_size, epsilon_min, epsilon_max, decay, gamma, target_update):
    environment = gym.make('ALE/Pong-v5', frameskip=4)
    num_actions = environment.action_space.n

    epsilon = Epsilon(epsilon_min, epsilon_max, decay)

    agent = DQNAgent(gamma, epsilon, num_actions, batch_size)

    for episode_num in range(num_episodes):
        time_start = time.perf_counter()
        episode_reward = 0

        current_state = environment.reset()
        current_state = preprocess_image(current_state)
        done = False
        while not done:
            action = agent.get_action(current_state)

            next_state, reward, done, _ = environment.step(action)
            next_state = preprocess_image(next_state)

            episode_reward += reward

            agent.memory.push(current_state, action, next_state, torch.FloatTensor([reward]))

            agent.train()

            current_state = next_state
        
        print(f"Episode: {episode_num} Reward: {episode_reward} Time Taken: {time.perf_counter() - time_start}")

        if episode_num % target_update == 0:
            agent.target_network.load_state_dict(agent.policy_network.state_dict())


    

if __name__=="__main__":
    NUM_EPISODES = 1000
    BATCH_SIZE = 128
    EPS_MIN = 0.05
    EPS_MAX = 1
    EPS_DECAY = 200
    GAMMA = 0.9
    TARGET_UPDATE = 10

    main(NUM_EPISODES, BATCH_SIZE, EPS_MIN, EPS_MAX, EPS_DECAY, GAMMA, TARGET_UPDATE)