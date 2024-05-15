import datetime

import cv2
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

DEATH_PENALTY = -200
NUM_FRAMES = 4
SAVE_INTERVAL = 250
OPTIMIZE_TICK_RATE = 10

PY_FILE_NAME = "qbert_more.pytorch"

parser = argparse.ArgumentParser(
    prog='q-learning',
    description='learns to play a game'
)

parser.add_argument('-s', '--save', default=PY_FILE_NAME, help="file to save model to", type=str)
parser.add_argument('-l', '--load', default=PY_FILE_NAME, help="file to load model from", type=str)
parser.add_argument('-t', '--train', default="False", help="if want to train the model", type=str)
parser.add_argument('-f', '--frame_skip', default="False", help="if want to have frame skips", type=str)

args = parser.parse_args()

frame_skip = args.frame_skip == "True"

GAME_ID = "Qbert-v4" if frame_skip else "QbertNoFrameskip-v4"

env = gym.make(GAME_ID)
if not frame_skip:
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False,
                                          grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
env = gym.wrappers.FrameStack(env, 4)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

preprocess_method = None


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.flatten = nn.Flatten()

        conv_out_size = self._get_conv_out(input_shape)
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def _get_conv_out(self, shape):
        with torch.no_grad():
            o = self._forward_conv(torch.zeros(1, *shape))
            return o.view(o.size(0), -1).size(1)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x.view(x.size(0), -1)


# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, _ = env.reset()
input_shape = (NUM_FRAMES, 84, 84)

policy_net = DQN(input_shape, n_actions).to(device)
target_net = DQN(input_shape, n_actions).to(device)
if args.load is not None:
    print(f"loading {args.load}")
    policy_net.load_state_dict(torch.load(args.save, map_location=device))
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def preprocess_observation(observation):
    """
    Preprocesses the observation to match the network's first input size and adds a batch.
    :param observation: Observation to preprocess.
    :return: tensor with a processed observation and a batch.
    """
    return torch.tensor(np.array(observation), dtype=torch.float32, device=device).unsqueeze(0)


def preprocessBigObservation(big_observation):
    """
    Preprocesses the observation to match the network's first input size and adds a batch.
    :param big_observation: "Big Observation" which is probably a non-frame-skipping observation.
    :return: tensor with a processed observation and a batch.
    """
    frames = [np.array(frame, copy=False) for frame in big_observation]
    return preprocess_observation(
        np.array([cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (84, 84)) for frame in frames]))


def run_model(count=100):
    """You should probably not modify this, other than
    to load qbert.
    """
    env = gym.make(GAME_ID, render_mode="human")
    if not frame_skip:
        env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84,
                                              terminal_on_life_loss=False,
                                              grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
    env = gym.wrappers.FrameStack(env, 4)
    # Initialize the environment and get it's state
    state, _ = env.reset()
    state = preprocess_method(state)
    for t in range(count):
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())

        if terminated:
            state = None
        else:
            state = preprocess_method(observation)
        env.render()
        if terminated or truncated:
            break


def train_model():
    """ You may want to modify this method: for instance,
    you might want to skip frames during training."""

    start = datetime.datetime.now()
    episode_count = 0
    try:
        while True:
            # Initialize the environment and get it's state
            state, info = env.reset()
            state = preprocess_method(state)
            prev_lives = info["lives"]
            for t in count():
                action = select_action(state)
                observation, reward, terminated, truncated, info = env.step(action.item())
                reward += (prev_lives - info["lives"]) * DEATH_PENALTY  # Give negative reward for dying!!
                reward = torch.tensor([reward], device=device)
                prev_lives = info["lives"]
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = preprocess_method(observation)

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                if t % OPTIMIZE_TICK_RATE == 0:
                    # Perform one step of the optimization (on the policy network)
                    optimize_model()

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = target_net.state_dict()
                    policy_net_state_dict = policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (
                                1 - TAU)
                    target_net.load_state_dict(target_net_state_dict)

                if done:
                    # episode_durations.append(t + 1)
                    # plot_durations()
                    break
            if episode_count % SAVE_INTERVAL == 0:
                torch.save(policy_net.state_dict(), args.save)
            episode_count += 1
    except KeyboardInterrupt:
        print("Interrupt")
    print("Elapsed:", datetime.datetime.now() - start, "Episodes:", episode_count)
    env.close()
    torch.save(policy_net.state_dict(), args.save)


preprocess_method = preprocessBigObservation if frame_skip else preprocess_observation

if __name__ == '__main__':
    if args.train == "True":
        print("Training")
        train_model()
        # plot_durations(True)
        print('Time to play')

    policy_net.eval()
    EPS_START = EPS_END = 0

    run_model(10000)
    env.close()
