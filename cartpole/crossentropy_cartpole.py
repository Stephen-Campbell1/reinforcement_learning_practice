from dataclasses import dataclass, field
import gymnasium as gym
from typing import List, Tuple, Dict, Any
import numpy as np
from VideoEncoder import VideoFrameEncoder
from typing import List
import torch
from torch import nn
import os

# takes in 4 floats in a list
# outputs an action


class CartpoleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pipe = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, data):
        return self.pipe(data)

@dataclass
class Episode:
    total_reward: float = 0
    total_steps: int = 0
    # Maps observations -> action
    events: List[Tuple[torch.tensor, int]] = field(default_factory=list)

    def record_step(self, observation: List, action: int, reward: float):
        self.total_steps += 1
        self.total_reward += reward
        self.events.append((observation, action))


@dataclass
class EpisodeBatch:
    episodes: List[Episode] = field(default_factory=list)

    def get_unzipped(self) -> Tuple[List[torch.tensor], List[int]]:
        observations = []
        actions = []
        for episode in self.episodes:
            for event in episode.events:
                observations.append(event[0])
                actions.append(event[1])
        return (observations, actions)

    def record_episode(self, e: Episode):
        self.episodes.append(e)

    def get_percentile_and_up(self, percentile: float = 50) -> List[Episode]:
        reward_boundary = self.get_reward_boundary_for_percentile(percentile)
        return EpisodeBatch(episodes=list(filter(lambda episode: episode.total_reward >= reward_boundary, self.episodes)))

    def get_reward_boundary_for_percentile(self, percentile: float = 50) -> List[Episode]:
        episode_rewards = list(map(
            lambda episode: episode.total_reward, self.episodes))
        reward_boundary = np.percentile(episode_rewards, percentile)
        return reward_boundary

    def get_mean_reward(self):
        episode_rewards = list(map(
            lambda episode: episode.total_reward, self.episodes))
        return np.mean(episode_rewards)


def select_action_with_model(net: nn.Module, observation: List[float]) -> int:
    observation_tensor = torch.tensor(observation)
    unbounded_action_distribution = net(observation_tensor)
    soft_max = nn.Softmax()
    softmaxed_action_distribution = soft_max(
        unbounded_action_distribution)

    softmaxed_action_distribution_np = softmaxed_action_distribution.data.numpy()
    # softmaxed_action_distribution now represents a probablility distribution
    # that indicates what the model thinks we should do based off of our learnings
    # Based of this distrubtion, we pick an action at random
    selected_action = np.random.choice(
        [0, 1], p=softmaxed_action_distribution_np)
    return selected_action


def get_batch(net: nn.Module, batch_size: int = 16) -> EpisodeBatch:
    env = gym.make('CartPole-v1')
    episode_batch: EpisodeBatch = EpisodeBatch()
    for _ in range(batch_size):
        # Play out the episode
        episode = Episode()
        # observation is a python list of 4 elements
        # info is an empty dict
        observation, _ = env.reset()
        while True:
            selected_action = select_action_with_model(net, observation)
            observation_for_selected_action = observation
            observation, reward,  terminated, truncated, info, = env.step(
                selected_action)
            episode.record_step(
                observation_for_selected_action, selected_action, reward)
            if terminated or truncated:
                episode_batch.record_episode(episode)
                break
    return episode_batch


# net should take in a tensor of shape (4,) and output a tensor of shape (2,) (not softmaxed)
def batch_generator(net: nn.Module, batch_size: int = 16) -> EpisodeBatch:
    while True:
        yield get_batch(net, batch_size)


def train_net_with_episode_batch(criterion: nn.Module, optimizer: torch.optim.Optimizer, net: nn.Module, episode_batch: EpisodeBatch):
    observation_batch, selected_action_batch = episode_batch.get_unzipped()
    observation_batch_tensor = torch.tensor(observation_batch)
    selected_action_batch_tensor = torch.tensor(selected_action_batch)
    action_batch_tensor = net(observation_batch_tensor)
    optimizer.zero_grad()
    loss = criterion(action_batch_tensor, selected_action_batch_tensor)
    loss.backward()
    optimizer.step()
    return loss


def train_loop():
    MAX_ITERATIONS = 1000
    MAX_MEAN_REWARD = 250
    iterations = 0
    mean_reward = 0
    model = CartpoleNet()
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for batch in batch_generator(model):
        if iterations >= MAX_ITERATIONS:
            print("Reached Maximum number of iterations")
            break
        if mean_reward >= MAX_MEAN_REWARD:
            print("Acheived Mean Reward Goal")
            break
        iterations += 1
        elite_batch = batch.get_percentile_and_up(75)
        reward_boundary = batch.get_reward_boundary_for_percentile(75)
        mean_reward = batch.get_mean_reward()
        loss = train_net_with_episode_batch(
            criterion, optimizer, model, elite_batch)
        print(f"s={len(elite_batch.episodes)/len(batch.episodes)} loss={loss.item()} reward_boundary={reward_boundary} mean_reward={mean_reward}")
    return model


# %%
model = train_loop()
# %%
"""
This function spawns up a cartpole environment and selects a
random action each step.
This also has the side effect of saving all trials to a ./trials/episodes.mp4

"""


def record_episode_batch(net: nn.Module):
    cartpole_env = gym.make('CartPole-v1', render_mode='rgb_array')
    NUM_EPISODES = 30
    __file__
    os.makedirs('./trials', exist_ok=True)
    # os.mkdirs('./trials', )
    # pathlib.Path.mkdir("./trials",exist_ok=True)
    with VideoFrameEncoder((600, 400), './trials/episodes.mp4') as encoder:
        for episode_index in range(NUM_EPISODES):
            observation, info = cartpole_env.reset()
            while True:
                selected_action = select_action_with_model(net, observation)
                observation, reward,  terminated, truncated, info, = cartpole_env.step(
                    selected_action)
                frame_hwc = cartpole_env.render()
                encoder.write_frame(frame_hwc)
                if terminated or truncated:
                    break


# %%
record_episode_batch(model)
