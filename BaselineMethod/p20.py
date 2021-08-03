import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

from collections import deque

from linear_atari import LinearAtariWrapper
from running_utils import Checkpoint, Metrics


class P20:
    '''
    The P20 object is the main object for training the RL agent for both of the Baseline Method variants

    Parameters
    ----------
    env : LinearAtariWrapper
        Environment wrapper passed with a seed-initialised environment. See class LinearAtariWrapper for more.
    theta : np.ndarray
        The theta vector of weights to be optimised for training the RL agent
        In the Baseline Method it has shape (1, number_of_features * number_of_actions_in_env)
    metrics : Metrics
        Metrics object for measurements. See class Metrics for more
    checkpoint : Checkpoint
        Checkpoint object for the checkpoint-saving system. See class Checkpoint for more
    '''
    def __init__(self, env=None, theta=None, metrics=None, checkpoint=None):
        self.env = env
        self.theta = theta

        self.checkpoint = checkpoint
        self.metrics = metrics

    def get_action(self, Q, epsilon, random):
        ''' Epsilon-Greedy Policy and Random Tie-Breaking
        (Sutton & Barto, 1998; Sutton & Barto, 2018), Epsilon-Greedy Policy

        Returns an integer corresponding with an action from the space range(number_of_actions_in_env)

        Parameters
        ----------
        Q : np.ndarray
            Action-values obtained following linear value function approximation
            Q-values array of shape (4, 1)
        epsilon : float
            Linearly annealed epsilon value
            The probability of choosing a greedy action to facilitate the exploitation of learned weights
        random : np.random.RandomState
            Random state with environment seed for choosing a random action to facilitate environment exploration

        Returns
        ----------
        action : int
            With probability 1-epsilon, for environment exploration
             -- If random.rand() < epsilon; Or, for random tie-breaking, if random.rand() >= epsilon & all values in Q are too close to max(Q): a random action for environment exploration. <br>
            Otherwise, with probability epsilon, for exploitation of learned weights
             -- If random.rand() >= epsilon and in Q at least one action not too close to Q(max): argmax(Q), that action whose value is maximum from Q.
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
        ''' On-Policy Sarsa Algorithm for Control. Training of the RL agent for the Baseline Method.
        (Sutton & Barto, 1998; Sutton & Barto, 2018), Sarsa for Control

        Parameters
        ----------
        start_episode : int
            The episode to begin training from
        max_episodes : int
            Maximum number of training episodes
        solved_at : int
            Condition for solving the environment: a value less than or equal to max_episode.
        lr : float
            The learning rate alpha for optimisation of vector theta
        gamma : float
            The discount factor for the expected return
        epsilon : float
            Maximum value for epsilon linear annealing
        min_epsilon : float
            Minimum value for epsilon linear annealing
        render : bool
            Whether to render the video game frames while training
            If True, only works on local machine if ran from Jupyter Notebook; env.render() raises exceptions otherwise.
        seed : int
            Seed the environment P20.env has been initialised with; Also the seed for initialising the random state for the Epsilon-Greedy Policy

        Returns
        ----------
        tuple[Metrics, np.ndarray]
            metrics - RL agent training performance dictionary of metrics. See Metrics for more. <br>
            theta - Learned vector of weights for the RL agent. In the Baseline Method it has shape (1, number_of_features * number_of_actions_in_env)
        '''

        # If checkpoint exists, it reloads the counters from the last episode to continue training with the next
        if self.checkpoint.has_checkpoint:
            start_episode, frame_count, highest_score, rolling_reward_window100 = self.checkpoint.get_counters()
            start_episode += 1
        else:
            # Counter initialisation if no checkpoints exists
            frame_count, highest_score, rolling_reward_window100 = 0, 0, deque(maxlen=100)

        # Random state for the epsilon-greedy policy
        # Initialisation with the same seed as the environment
        random_state = np.random.RandomState(seed)
        # Linearly annealed epsilon
        epsilon = np.linspace(epsilon, min_epsilon, max_episodes)

        for episode in range(start_episode, max_episodes + 1):
            features = self.env.reset()
            frame_count += 1
            if render: self.env.render()

            # Linear function approximation of the action values Q(s, a)
            Q = features.dot(self.theta)
            # Selecting next action with the Epsilon-Greedy Policy
            action = self.get_action(Q, epsilon[episode - 1], random_state)

            ep_score = 0
            done = False
            while not done:
                next_features, reward, done = self.env.step(action)
                frame_count += 1
                if render: self.env.render()

                # Linear value function approximation of the next action values Q(s+1, a+1)
                next_Q = next_features.dot(self.theta)
                # Selecting next action with the Epsilon-Greedy Policy
                next_action = self.get_action(next_Q, epsilon[episode - 1], random_state)

                # The Temporal Difference error between the discounted immediate expected return and the prediction
                temp_diff = reward + (gamma * next_Q[next_action]) - Q[action]
                # Optimisation of theta with Stochastic Gradient Descent (SGD)
                self.theta += lr * temp_diff * features[action]

                features = next_features
                Q = features.dot(self.theta)
                action = next_action

                ep_score += reward

            # Retaining the cumulative rewards received from the last 100 episodes for metrics and determining convergence
            rolling_reward_window100.append(ep_score)
            # Computing the rolling average of the last 100 cumulative rewards received once the 100th episode is done
            # Beginning to dump metrics and the checkpoint to disk when the 100th episode is done
            if len(rolling_reward_window100) == 100:
                rolling_reward = np.mean(rolling_reward_window100)
                # Computing the maximum highest rolling average to present for metrics and determining if environment is solved
                highest_score = max(rolling_reward, highest_score)

                print(f"{episode}/{max_episodes} done \tEpisode Score: {ep_score}"
                      f"\t\tAvg Score 100 Episodes: {rolling_reward:2f}"
                      f"\tHighest Avg Score: {highest_score:2f}"
                      f"\t\tFrame count: {frame_count}")

                # Dumping metrics and checkpoint to disk every 50 episodes as part of performance metrics collection and checkpoint-saving systems
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
    ''' Use this method for resuming training, or for starting up new training. <br>
    Unless the following parameters have been passed to the P20 class as such:
    environment wrapped with class linear_atari.LinearAtariWrapper, theta is a Numpy array with the shape indicated in the report,
    and the objects metrics & checkpoints have been created for a P20 object,
    this method is recommended instead for handling all the above. <br>
    Class P20 can still be instantiated and P20.train(..) can still be called, without using training_p20(..).

    Parameters
    ----------
    env : Any
        Atari environment for training
    seed : int
        The same seed the environment has been initialised with.
        It also becomes the same seed the random state will be initialised with for the Epsilon-Greedy Policy
    solved_at : int
        Condition for solving the environment: a value less than or equal to max_episode.
    render : bool
        Whether to render the video game frames while training
        If True, only works on local machine if ran from Jupyter Notebook; env.render() raises exceptions otherwise.
    max_episodes : int
        Maximum number of training episodes
    lr : float
        The learning rate alpha for optimisation of vector theta
    gamma : float
        The discount factor for the expected return
    max_epsilon : float
        Maximum value for epsilon linear annealing
    min_epsilon : float
        Minimum value for epsilon linear annealing
    p20_model : tensorflow.keras.models.Model
        This is the model resulting from the the 2 ablations to the architecture. The resulting model must be of type tensorflow.keras.models.Model <br>
        It cannot ever be None if training_p20(..) is meant to be used. <br>
        Also, the size of the model output must match the number of features given in the checkpoint file, if checkpoint file given to resume training.
    metrics_filename : str
        The path of the metrics file with the .pkl extension <br>
        If None, new file is created at the next checkpoint, with a timestamped filename under format '%d-%m-%Y_%H-%M-%S_metrics.pkl' on the path given by your machine's Python configuration <br>
        If set, it always attempts to read the dictionary with previous metrics saved in the .pkl file and continues to append newly collected metrics every 50 episodes from episode 100 onwards <br>
        To see what the structure of this file is and how the metrics collection system is implemented, refer to the ./BaselineMethod/running_utils.py file.
    checkpoint_filename : str
        The path of the checkpoint file with the .pkl extension <br>
        If None, new file is created at the next checkpoint, with a timestamped filename under format '%d-%m-%Y_%H-%M-%S_checkpoint.pkl' on the path given by your machine's Python configuration <br>
        If set, it always attempts to read the file contents and replaces with a new checkpoint every 50 episodes (only if after episode 100 due to the rolling window of 100 rewards being stored as well) <br>
        To see what the structure of this file is and how the checkpoint-saving system is implemented, refer to the ./BaselineMethod/running_utils.py file.
    theta_filename : str
        The Numpy vector of weights theta saved in this file with the .npy extension. <br>
        If None, it is stored in a new file created with a name timestamped under format '%d-%m-%Y_%H-%M-%S_theta.npy' on the path given by your machine's Python configuration. <br>
        It only saves theta at the end of training; And if training is stopped early, theta can be found in the checkpoint file.
    '''
    metrics = Metrics(metrics_filename)
    training = Checkpoint(checkpoint_filename)
    # Checkpoint objects have a flag, the has_checkpoint. If True, a checkpoint exists after having successfully read the contents of the file; False otherwise.
    # The size of the model output must match the number of features given in the checkpoint file, if checkpoint file given to resume training.
    # The number of features is given by the last layer in the model, in order to wrap the environment with the linear_atari.LinearAtariWrapper class
    # and to determine the size of vector of weights theta.
    # The next if-else structure is part of the checkpoint-saving system to resume training if checkpoint file is given.
    if training.has_checkpoint:
        seed, theta, num_features  = training.get_training_properties()
        assert num_features == p20_model.layers[-1].output_shape[1], "Size of model output does not match size of feature from checkpoint file given"
    else:
        num_features = p20_model.layers[-1].output_shape[1]
        theta = np.zeros(num_features * env.action_space.n)
    training.checkpoint["seed"] = seed
    training.checkpoint["num_features"] = num_features

    # Environment must be wrapped using the linear_atari.LinearAtariWrapper, as described by the methodology from the final report.
    # Refer to linear_atari.LinearAtariWrapper for a summary.
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