from collections import deque
import random
import gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Policy(nn.Module):
    def __init__(self, is_actor):
        super(Policy, self).__init__()

        self.isActor = is_actor

        self.layers = nn.Sequential(
            nn.Linear(12000, 512),
            nn.ReLU(),
            nn.Linear(512, 2 if is_actor else 1),
        )

        self.clip_hyperparam = 0.1
        self.gamma = 0.99



    def pre_process_single_img(self, img):
        """ get 210x160x3 frame into length 6000 1D vector. """
        if img is None:
            return torch.zeros(1, 6000)
        img = img[35:185] # crop
        img = img[::2, ::2, 0] # downsample by 2.
        img[img == 109] = 0 # remove bg
        img[img == 144] = 0 # remove bg 2
        img[img != 0] = 1 # paddles+paddle are white

        return torch.from_numpy(img.astype(np.float32).ravel()).unsqueeze(0)

    def get_action(self, action):
        return 2+action

    def pre_process(self, frame, prev_frame):
        return self.pre_process_single_img(frame) - self.pre_process_single_img(prev_frame)

    def forward(self, d_obs):
        with torch.no_grad():
            if self.isActor:
                policyOutput = self.layers(d_obs)
                probabilityDistribution = torch.distributions.Categorical(logits=policyOutput)
                action = int(probabilityDistribution.sample().cpu().numpy()[0])
                action_prob = float(probabilityDistribution.probs[0, action].detach().cpu().numpy())
                return action, action_prob

            vals = self.layers(d_obs)
            return vals

    def get_ppo_loss(self, network_obs, actions, action_probs, advantages):
        # PPO
        action_bools = np.array([[1., 0.], [0., 1.]])
        mask_for_new_policy_probs = torch.FloatTensor(action_bools[actions.cpu().numpy()])

        new_predictions = self.layers(network_obs)
        policy_ratio = torch.sum(functional.softmax(new_predictions, dim=1) * mask_for_new_policy_probs, dim=1) / action_probs
        clamped_loss = torch.clamp(policy_ratio, 1 - self.clip_hyperparam, 1 + self.clip_hyperparam) * advantages
        non_clamped_loss = policy_ratio * advantages
        loss = -torch.min(clamped_loss, non_clamped_loss)
        loss = torch.mean(loss)

        return loss

    def get_mse_loss(self, d_obs, actual_rewards):
        vals = torch.squeeze(self.layers(d_obs))
        loss = nn.MSELoss()(vals, actual_rewards)
        return loss


env = gym.make('ALE/Pong-v5', frameskip=4)
# env = gym.make('ALE/Pong-v5', frameskip=4, render_mode="human")

load = False

env.reset()

actor = Policy(True)
critic = Policy(False)

all_ep_rewards = []

path_actor = f'./actorParams.ckpt'
path_critic = f'./criticParams.ckpt'
path_rewards = f'./episodeRewards.csv'
if load:
    actor.load_state_dict(torch.load(path_actor))
    actor.eval()
    critic.load_state_dict(torch.load(path_critic))
    critic.eval()
    all_ep_rewards = np.loadtxt(path_rewards)
    print(all_ep_rewards)
    all_ep_rewards = all_ep_rewards.tolist()



actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
# actor_lr_scheduler = ReduceLROnPlateau(actor_opt, 'min', patience=100)

critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
# critic_lr_scheduler = ReduceLROnPlateau(critic_opt, 'min', patience=100)

all_ep_rewards = []

last_20_ep_reward = deque(maxlen=20)
iteration = -1
while True:
    iteration += 1
    network_obs_history, action_history, action_probability_history, reward_history = [], [], [], []
    for episode in range(10):
        obs, prev_obs, prev_prev_obs = env.reset(), None, None
        timestep = 0
        while True:
            intermediate_obs1 = actor.pre_process(obs, prev_obs)
            intermediate_obs2 = actor.pre_process(prev_obs, prev_prev_obs)
            network_obs = torch.cat((intermediate_obs1, intermediate_obs2), -1)

            with torch.no_grad():
                action, action_probability = actor(network_obs)

            prev_prev_obs = prev_obs
            prev_obs = obs
            obs, reward, done, info = env.step(actor.get_action(action))

            network_obs_history.append(network_obs)
            action_probability_history.append(action_probability)
            action_history.append(action)
            reward_history.append(reward)

            timestep += 1

            if done:
                episode_reward = sum(reward_history[-timestep:])
                all_ep_rewards.append(episode_reward)
                last_20_ep_reward.append(episode_reward)
                reward_sum_avg = np.mean(last_20_ep_reward)
                print('Iteration', iteration, 'episode', episode, 'timesteps:', timestep, 'reward_sum:', episode_reward, 'running_avg:', reward_sum_avg)
                break

    discounted_reward = 0
    discounted_rewards = []
    for r in reward_history[::-1]:
        if r != 0:
            discounted_reward = 0 # point scored, reset reward sum
        discounted_reward = r + actor.gamma*discounted_reward
        discounted_rewards.append(discounted_reward)

    discounted_rewards.reverse()

    d_obs_history_tensor = torch.cat(network_obs_history, 0)
    discounted_rewards_tensor = torch.FloatTensor(discounted_rewards)

    critic_predicted_values = torch.squeeze(critic(d_obs_history_tensor))
    advantages = discounted_rewards_tensor - critic_predicted_values.detach()
    # Normalizing advantages isn't necessary. But it decreases the variance of the advantages
    # and makes convergence faster
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

    # update policy
    numPolicyIters = 5
    for _ in range(numPolicyIters):
        n_batch = math.floor(0.6*len(action_history))

        sample_idxs = random.sample(range(len(action_history)), n_batch)
        network_obs_batch = torch.cat([network_obs_history[idx] for idx in sample_idxs], 0)
        action_probs_batch = torch.FloatTensor([action_probability_history[idx] for idx in sample_idxs])
        actions_batch = torch.LongTensor([action_history[idx] for idx in sample_idxs])
        advantages_batch = torch.FloatTensor([advantages[idx] for idx in sample_idxs])
        discounted_rewards_batch = torch.FloatTensor([discounted_rewards[idx] for idx in sample_idxs])

        actor_loss = actor.get_ppo_loss(network_obs_batch, actions_batch, action_probs_batch, advantages_batch)
        critic_loss = critic.get_mse_loss(network_obs_batch, discounted_rewards_batch)

        actor_opt.zero_grad()
        actor_loss.backward(retain_graph=True)
        actor_opt.step()
        # actor_lr_scheduler.step(actor_loss)

        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
        # critic_lr_scheduler.step(critic_loss)


    if iteration % 5 == 0:
        torch.save(actor.state_dict(), 'actorParams.ckpt')
        torch.save(critic.state_dict(), 'criticParams.ckpt')
        np.savetxt('episodeRewards.csv', np.array(all_ep_rewards))


env.close()

