import datetime

import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium import logger

from Block import Block
from EntityMapper import EntityMapper
from qbert_rl import preprocess_observation, DEATH_PENALTY, memory, optimize_model, target_net, policy_net, \
    episode_durations, plot_durations, OPTIMIZE_TICK_RATE, PY_FILE_NAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCENARIOS = 1000
MAX_TRAIN_EPISODE = 2000
TAU = 0.01
SAVE_INTERVAL = 250


def preprocessBigObservation(obs):
    frames = [np.array(frame, copy=False) for frame in obs]
    return preprocess_observation(
        np.array([cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (84, 84)) for frame in frames]))


if __name__ == '__main__':
    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make("Qbert-v4")
    env = gym.wrappers.FrameStack(env, 4)
    print("Beginning Training")
    start = datetime.datetime.now()
    episode_count = 0
    try:
        while True:
            observation, info = env.reset()
            prev_lives = info["lives"]
            state = preprocessBigObservation(observation)
            Block.resetBlocks()
            start_time = datetime.datetime.now()
            for t in range(MAX_TRAIN_EPISODE):
                action = torch.tensor([[EntityMapper.updateAll(observation[-1])]], device=device, dtype=torch.long)
                observation, reward, terminated, truncated, info = env.step(action.item())
                reward += (prev_lives - info["lives"]) * DEATH_PENALTY  # Give negative reward for dying!!
                reward = torch.tensor([reward], device=device)
                prev_lives = info["lives"]
                if terminated:
                    next_state = None
                else:
                    next_state = preprocessBigObservation(observation)

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                if t % OPTIMIZE_TICK_RATE == 0:
                    optimize_model()

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = target_net.state_dict()
                    policy_net_state_dict = policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (
                                1 - TAU)
                    target_net.load_state_dict(target_net_state_dict)

                if terminated or truncated or t == MAX_TRAIN_EPISODE - 1:
                    # episode_durations.append(t + 1)
                    # plot_durations()
                    break
            # print("Elapsed", datetime.datetime.now() - start_time)
            if episode_count % SAVE_INTERVAL == 0:
                torch.save(policy_net.state_dict(), PY_FILE_NAME)
            episode_count += 1
    except KeyboardInterrupt:
        print("Interrupt.")

    print("Elapsed:", datetime.datetime.now() - start, "Episodes:", episode_count)

    env.close()
    torch.save(policy_net.state_dict(), PY_FILE_NAME)
