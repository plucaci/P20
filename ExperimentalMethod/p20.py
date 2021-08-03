import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

from collections import deque

from linear_atari import LinearAtariWrapper
from running_utils import Checkpoint, Metrics
from baselines.common.atari_wrappers import EpisodicLifeEnv, FireResetEnv, WarpFrame, ScaledFloatFrame, ClipRewardEnv

### Adapted from https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/atari_wrappers.py#L275
def wrap_atariari(env, width=160, height=210, grayscale=True, episode_life=False, clip_rewards=True, scale=True):
    """Configure environment for AtariARI-style Atari. """

    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, width, height, grayscale, dict_space_key=None)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    return env

### Adapted from https://github.com/mila-iqia/atari-representation-learning/blob/08165740a7688853c6315751003aa4dee9901073/README.md#L125
class BaseEncoder(nn.Module):
    # Network defined by the Deepmind paper (Mnih, et al., 2013)
    ## Modified variant
    def __init__(self):
        super().__init__()
        self.final_conv_size = 64 * 9 * 6
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, stride=1),
            nn.ReLU()
        )

    def forward(self, inputs):
        x = self.cnn(inputs)
        x = x.view(x.size(0), -1)
        return x

class P20:

    '''
    The P20 object is the main class for training the RL agent for both of the Experimental Method trials

    Parameters
    ----------
    env : LinearAtariWrapper object
        Environment wrapper with the environment passed. See class LinearAtariWrapper for more
    theta : NumPy object
        The theta vector of weights to be optimised for training the RL agent
        In the Experimental Method it has shape (number_of_features * number_of_actions_in_env)
    metrics : Metrics object
        Metrics object for measurements. See class Metrics for more
    '''
    def __init__(self, env=None, theta=None, metrics=None, checkpoint=None):
        self.env = env
        self.theta = theta

        self.checkpoint = checkpoint
        self.metrics = metrics

    def get_action(self, Q, epsilon, random):
        ''' Epsilon-Greedy Policy and Random Tie-Breaking

        Returns an integer corresponding with an action from the space range(number_of_actions_in_env)

        Parameters
        ----------
        Q : numpy.ndarray
            Action-values obtained following linear value function approximation
            Q-values array of shape (4, 1)
        epsilon : float
            Linearly annealed epsilon value
            The probability of choosing a greedy action to facilitate the exploitation of learned weights
        random : numpy.random.RandomState
            Random state with environment seed for choosing a random action to facilitate environment exploration

        Returns
        ----------
        int
            random integer from set, if random.rand() < epsilon, or if epsilon >= random.rand() but all values in Q are close to max(Q);
            Otherwise, the action whose value is maximum: argmax(Q) for exploitation of learned weights with probability epsilon
        '''

        if random.rand() < epsilon:
            a = random.choice(self.env.num_actions)
        else:
            Q_max = max(Q)
            best = [a for a in range(self.env.num_actions) if np.allclose(Q_max, Q[a])]
            if best:
                a = random.choice(best)
            else:
                a = np.argmax(Q)
        return a

    def train(self, start_episode, max_episodes, solved_at, lr, gamma, epsilon, min_epsilon, render, seed):
        if self.checkpoint.has_checkpoint:
            start_episode, frame_count, highest_score, rolling_reward_window100 = self.checkpoint.get_counters()
            start_episode += 1
        else:
            frame_count, highest_score, rolling_reward_window100 = 0, 0, deque(maxlen=100)

        random_state = np.random.RandomState(seed)
        epsilon = np.linspace(epsilon, min_epsilon, max_episodes)

        for episode in range(start_episode, max_episodes + 1):
            features = self.env.reset()
            frame_count += 1
            if render: self.env.render()

            Q = features.dot(self.theta)
            action = self.get_action(Q, epsilon[episode - 1], random_state)

            ep_score = 0
            done = False
            while not done:
                next_features, reward, done = self.env.step(action)
                frame_count += 1
                if render: self.env.render()

                next_Q = next_features.dot(self.theta)
                next_action = self.get_action(next_Q, epsilon[episode - 1], random_state)

                temp_diff = reward + (gamma * next_Q[next_action]) - Q[action]
                self.theta += lr * temp_diff * features[action]

                features = next_features
                Q = features.dot(self.theta)
                action = next_action

                ep_score += reward

            rolling_reward_window100.append(ep_score)
            if len(rolling_reward_window100) == 100:
                rolling_reward = np.mean(rolling_reward_window100)
                highest_score = max(rolling_reward, highest_score)

                print(f"{episode}/{max_episodes} done \tEpisode Score: {ep_score}"
                      f"\t\tAvg Score 100 Episodes: {rolling_reward:2f}"
                      f"\tHighest Avg Score: {highest_score:2f}"
                      f"\t\tFrame count: {frame_count}")

                if episode % 50 == 0:
                    self.checkpoint.collect(episode, frame_count, self.theta, highest_score, rolling_reward_window100)
                    self.metrics.collect(episode, frame_count, highest_score, rolling_reward)

                if rolling_reward >= solved_at:
                    print('Solved environment in', episode, 'episodes')
                    self.checkpoint.collect(episode, frame_count, self.theta, highest_score, rolling_reward_window100)
                    self.metrics.collect(episode, frame_count, highest_score, rolling_reward)
                    break
            else:
                print(f"{episode}/{max_episodes} done \tEpisode Score: {ep_score} \tFrame count: {frame_count}")

        return self.metrics, self.theta


def training_p20(env=None, seed=0, solved_at=40, render=False,
                 max_episodes=55000, lr=0.00025, gamma=0.99, max_epsilon=1, min_epsilon=0.1, p20_model=None,
                 metrics_filename=None, checkpoint_filename=None, theta_filename=None):

    metrics = Metrics(metrics_filename)
    training = Checkpoint(checkpoint_filename)
    if training.has_checkpoint:
        seed, theta, num_features  = training.get_training_properties()
        assert num_features == p20_model.final_conv_size, "Size of model output does not match size of feature from checkpoint file given"
    else:
        num_features = p20_model.final_conv_size
        theta = np.zeros(num_features * env.action_space.n)
    training.checkpoint["seed"] = seed
    training.checkpoint["num_features"] = num_features

    linear_env = LinearAtariWrapper(env=env, p20_model=p20_model, num_features=num_features)

    train_p20 = P20(env=linear_env, theta=theta, metrics=metrics, checkpoint=training)
    metrics, theta = train_p20.train(
        start_episode = 1,
        max_episodes  = max_episodes,
        solved_at     = solved_at,
        lr            = lr,
        gamma         = gamma,
        epsilon       = max_epsilon,
        min_epsilon   = min_epsilon,
        render        = render,
        seed          = seed
    )

    if theta_filename is None:
        theta_filename = time.strftime('%d-%m-%Y_%H-%M-%S') +'_theta.npy'
        print('Saved theta to file', theta_filename, 'instead')
    with open(theta_filename, 'wb') as theta_file:
        np.save(theta_file, theta)
    theta_file.close()

    df = pd.DataFrame.from_dict(metrics.metrics, orient='index')
    print('\nCollected metrics during training')
    print(df)
    print('')

    plt.plot(df['episode'], df['highest_score'])
    plt.plot(df['episode'], df['rolling_reward'])
    plt.show()

    plt.plot(df['frame'], df['highest_score'])
    plt.plot(df['frame'], df['rolling_reward'])
    plt.show()
