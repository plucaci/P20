import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

from collections import deque

from running_utils import Checkpoint, Metrics

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class P20:
    def __init__(self, env=None, metrics=None, checkpoint=None, p20_model=None, optimizer=None, criterion=None):
        self.env = env

        self.checkpoint = checkpoint
        self.metrics = metrics

        self.p20_model = p20_model
        self.optimizer = optimizer
        self.criterion = criterion

    def get_action(self, Q, epsilon, random):
        if random.rand() < epsilon:
            a = random.choice(self.env.action_space.n)
        else:
            Q_max = max(Q)
            best = [a for a in range(self.env.action_space.n) if np.allclose(Q_max, Q[a])]
            if best:
                a = random.choice(best)
            else:
                a = np.argmax(Q)
        return a

    def train(self, start_episode, max_episodes, solved_at, gamma, epsilon, min_epsilon, render, seed):
        if self.checkpoint.has_checkpoint:
            start_episode, frame_count, highest_score, rolling_reward_window100 = self.checkpoint.get_counters()
            start_episode += 1
        else:
            frame_count, highest_score, rolling_reward_window100 = 0, 0, deque(maxlen=100)

        random_state = np.random.RandomState(seed)
        epsilon = np.linspace(epsilon, min_epsilon, max_episodes)

        for episode in range(start_episode, max_episodes + 1):
            state = self.env.reset()
            state = torch.unsqueeze(torch.Tensor(state).permute(2,0,1).float().to(device), dim=0)
            frame_count += 1
            if render: self.env.render()

            Q = self.p20_model(state)
            with torch.no_grad():
                action = self.get_action(torch.squeeze(Q).cpu().numpy(), epsilon[episode - 1], random_state)

            ep_score = 0
            done = False
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_state = torch.unsqueeze(torch.Tensor(next_state).permute(2, 0, 1).float().to(device), dim=0)
                frame_count += 1
                if render: self.env.render()

                next_Q = self.p20_model(next_state).detach()
                with torch.no_grad():
                    next_action = self.get_action(torch.squeeze(next_Q).cpu().numpy(), epsilon[episode - 1], random_state)

                temp_diff = reward + (gamma * next_Q[0, next_action])
                loss = self.criterion(Q[0, action], temp_diff)

                self.optimizer.zero_grad()
                loss.backward()
                for param in self.p20_model.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()

                state = next_state
                Q = self.p20_model(state)
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
                    self.checkpoint.collect(episode, frame_count, highest_score, rolling_reward_window100)
                    self.metrics.collect(episode, frame_count, highest_score, rolling_reward)

                if rolling_reward >= solved_at:
                    print('Solved environment in', episode, 'episodes')
                    self.checkpoint.collect(episode, frame_count, highest_score, rolling_reward_window100)
                    self.metrics.collect(episode, frame_count, highest_score, rolling_reward)
                    break
            else:
                print(f"{episode}/{max_episodes} done \tEpisode Score: {ep_score} \tFrame count: {frame_count}")

        return self.metrics


def training_p20(env=None, seed=0, solved_at=40, render=False,
                 max_episodes=55000, lr=0.00025, gamma=0.99, max_epsilon=1, min_epsilon=0.1, p20_model=None,
                 metrics_filename=None, checkpoint_filename=None, theta_filename=None):

    optimizer = optim.Adam(p20_model.parameters(), lr=lr)

    metrics = Metrics(metrics_filename)
    training = Checkpoint(checkpoint_filename, theta_filename, model=p20_model, optimizer=optimizer)
    if training.has_checkpoint:
        seed = training.get_training_properties()
        p20_model.load_state_dict(torch.load("./models/model_"+theta_filename))
        optimizer.load_state_dict(torch.load("./models/optim_"+theta_filename))
    training.checkpoint["seed"] = seed

    train_p20 = P20(env=env, metrics=metrics, checkpoint=training, p20_model=p20_model, optimizer=optimizer, criterion=nn.SmoothL1Loss())
    metrics = train_p20.train(
        start_episode = 1,
        max_episodes  = max_episodes,
        solved_at     = solved_at,
        gamma         = gamma,
        epsilon       = max_epsilon,
        min_epsilon   = min_epsilon,
        render        = render,
        seed          = seed
    )

    if theta_filename is None:
        theta_filename = time.strftime('%d-%m-%Y_%H-%M-%S') +'_theta.npy'
        print('Saved theta to file', theta_filename, 'instead')
    torch.save(p20_model.state_dict(), "./models/model_"+theta_filename)
    torch.save(optimizer.state_dict(), "./models/optim_"+theta_filename)

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