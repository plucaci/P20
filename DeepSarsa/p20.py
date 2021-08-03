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

    '''
    The P20 object is the main object for training the RL agent for both, Deep Sarsa & Deep Linear Sarsa

    Parameters
    ----------
    env : gym.Wrapper
        Environment wrapper passed with a seed-initialised environment. <br>
        Wrappers used must be returned by baselines.common.atari_wrappers.wrap_deepmind(..) method
        Expected DeepMind-style Atari: 84x84, grayscale, last 4 frames stacked.
    metrics : Metrics
        Metrics object for measurements. See class Metrics for more
    checkpoint : Checkpoint
        Checkpoint object for the checkpoint-saving system. See class Checkpoint for more
    p20_model : torch.nn.Module
        PyTorch model
    optimizer : torch.optim.optimizer.Optimizer
        PyTorch optimizer
    criterion : torch.nn.modules.loss._Loss
        PyTorch loss function
    '''
    def __init__(self, env=None, metrics=None, checkpoint=None, p20_model=None, optimizer=None, criterion=None):
        self.env = env

        self.checkpoint = checkpoint
        self.metrics = metrics

        self.p20_model = p20_model
        self.optimizer = optimizer
        self.criterion = criterion

    def get_action(self, Q, epsilon, random):
        ''' Epsilon-Greedy Policy and Random Tie-Breaking
        (Sutton & Barto, 1998; Sutton & Barto, 2018), Epsilon-Greedy Policy

        Returns an integer corresponding with an action from the space range(number_of_actions_in_env)

        Parameters
        ----------
        Q : np.ndarray
            Action-values obtained following value function approximation
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
        ''' On-Policy Sarsa Algorithm for Control. Training of the RL agent for the Deep Sarsa & Deep Linear Sarsa Method.
        (Sutton & Barto, 1998; Sutton & Barto, 2018) (Sarsa for Control)
        (Elfwing et al., 2017), deep Sarsa(lambda)

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
        metrics : Metrics
            RL agent training performance dictionary of metrics. See Metrics for more. <br>
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
            state = self.env.reset()
            state = torch.unsqueeze(torch.Tensor(state).permute(2,0,1).float().to(device), dim=0)
            frame_count += 1
            if render: self.env.render()

            # Value function approximation with a Deep Neural Network of the action values Q(s, a)
            # As in Temporal Difference Learning, we estimate a value on the basis of another estimate, we assume that the action-values Q are our predictions
            # Therefore, as with any kind of predictions in neural networks, we are not being detaching,
            # to allow the backpropagation of gradients while optimising theta (i.e., the last layer in this case) along with the other weights
            Q = self.p20_model(state)
            # Selecting next action with the Epsilon-Greedy Policy
            # As we have not detached, we must prevent gradients to propagate any forward when using Q to prepare for the selection!
            with torch.no_grad():
                action = self.get_action(torch.squeeze(Q).cpu().numpy(), epsilon[episode - 1], random_state)

            ep_score = 0
            done = False
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_state = torch.unsqueeze(torch.Tensor(next_state).permute(2, 0, 1).float().to(device), dim=0)
                frame_count += 1
                if render: self.env.render()

                # Value function approximation of the next action values Q(s+1, a+1)
                # Here, although we make a prediction, this is that of the target in the context of temporal difference learning
                # As with any targets, they are not supposedly predictions, therefore we detach
                # This is essentially 1-step bootstrapping but on the same network
                next_Q = self.p20_model(next_state).detach()
                with torch.no_grad():
                    next_action = self.get_action(torch.squeeze(next_Q).cpu().numpy(), epsilon[episode - 1], random_state)

                # This is essentially preparing the calculation of the loss, between the discounted immediate expected return and the prediction.
                temp_diff = reward + (gamma * next_Q[0, next_action])
                # As mentioned above, Q has not been detached as it is considered the prediction;
                # Whereas temp_diff used the target next_Q to calculate the discounted immediate expected return.
                # This is then helping us to calculate the actual temporal difference using a loss function from PyTorch.
                loss = self.criterion(Q[0, action], temp_diff)

                self.optimizer.zero_grad()
                # The next line is equivalent with hard-coding: (temp_diff - Q[action]) * convolutional_features,
                # to calculate the gradients of the loss function with respect to the weights theta
                loss.backward()
                # Clipping the gradients in place
                for param in self.p20_model.parameters():
                    param.grad.data.clamp_(-1, 1)
                # This is equivalent to updating theta using the learning rate step as a percentage of the sampled gradient:
                # theta = theta + [learning_rate * (temp_diff - Q[action]) * convolutional_features]
                self.optimizer.step()

                state = next_state
                # Same as before the inner while loop
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
    ''' Use this method for resuming training, or for starting up new training. <br>
    Unless the following parameters have been passed to the P20 class as such:
    environment wrapped, and the objects metrics & checkpoints have been created for a P20 object,
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
        The learning rate alpha for the PyTorch optimiser
    gamma : float
        The discount factor for the expected return
    max_epsilon : float
        Maximum value for epsilon linear annealing
    min_epsilon : float
        Minimum value for epsilon linear annealing
    p20_model : torch.nn.Module
        This is the model resulting from the the 2 ablations to the architecture. The resulting model must be of type torch.nn.Module <br>
        It cannot ever be None if training_p20(..) is meant to be used. <br>
        Also, the size of the model output must match the number of features given in the checkpoint file, if checkpoint file given to resume training.
    metrics_filename : str
        The path of the metrics file with the .pkl extension <br>
        If None, new file is created at the next checkpoint, with a timestamped filename under format '%d-%m-%Y_%H-%M-%S_metrics.pkl' on the path given by your machine's Python configuration <br>
        If set, it always attempts to read the dictionary with previous metrics saved in the .pkl file and continues to append newly collected metrics every 50 episodes from episode 100 onwards <br>
        To see what the structure of this file is and how the metrics collection system is implemented, refer to the ./DeepSarsa/running_utils.py file.
    checkpoint_filename : str
        The path of the checkpoint file with the .pkl extension <br>
        If None, new file is created at the next checkpoint, with a timestamped filename under format '%d-%m-%Y_%H-%M-%S_checkpoint.pkl' on the path given by your machine's Python configuration <br>
        If set, it always attempts to read the file contents and replaces with a new checkpoint every 50 episodes (only if after episode 100 due to the rolling window of 100 rewards being stored as well) <br>
        To see what the structure of this file is and how the checkpoint-saving system is implemented, refer to the ./DeepSarsa/running_utils.py file.
    theta_filename : str
        Name of files for optimiser and models, without extension or paths. <br>
        Optimiser is saved as "./models/optim_" + theta_filename. Model is saved as "./models/model_" + theta_filename.
        If None, it includes the pattern above and timestamps as below.
        If None, it is stored in a new file created with a name timestamped under format '%d-%m-%Y_%H-%M-%S' on the path "./models/" where your machine's Python configuration indicates. <br>
    '''

    optimizer = optim.Adam(p20_model.parameters(), lr=lr)

    metrics = Metrics(metrics_filename)
    training = Checkpoint(checkpoint_filename, theta_filename, model=p20_model, optimizer=optimizer)
    # Checkpoint objects have a flag, the has_checkpoint. If True, a checkpoint exists after having successfully read the contents of the file; False otherwise.
    # The size of the model output must match the number of features given in the checkpoint file, if checkpoint file given to resume training.
    # The next if-else structure is part of the checkpoint-saving system to resume training if checkpoint file is given.
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
        theta_filename = time.strftime('%d-%m-%Y_%H-%M-%S')
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