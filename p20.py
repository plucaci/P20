import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import deque

from baselines.common.atari_wrappers import make_atari, wrap_deepmind

from linear_atari import LinearAtariWrapper
from running_utils import Checkpoint, Metrics

"""
  Input shape K_FRAMES_STACKED x FRAME_SIZE x FRAME_SIZE
  As proposed in Chapter 2.2.1.2.1, this is 84 x 84 x 4
"""
# Number of consecutive frames stacked
K_FRAMES_STACKED = 4
# The size of the frame, after downsampling and cropping
FRAME_SIZE = 84

def q_model(num_actions=4):
    # Network defined by the Deepmind paper
    inputs = tf.keras.layers.Input(shape=(FRAME_SIZE, FRAME_SIZE, K_FRAMES_STACKED,))

    # Convolutions on the frames on the screen
    layer1 = tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu", data_format="channels_last")(inputs)
    layer2 = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu", data_format="channels_last")(layer1)
    layer3 = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu", data_format="channels_last")(layer2)

    layer4 = tf.keras.layers.Flatten()(layer3)

    # The adjusted output layer, with no activation="relu" of shape 'output_shape'
    layer5 = tf.keras.layers.Dense(units=512)(layer4)

    action = tf.keras.layers.Dense(num_actions, activation="linear")(layer5)

    return tf.keras.Model(inputs=inputs, outputs=action)


class P20:
    def __init__(self, env=None, theta=None, metrics=None, checkpoint=None):
        self.env = env
        self.theta = theta

        self.checkpoint = checkpoint
        self.metrics = metrics

    def get_action(self, Q, epsilon, random):
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

    def linear_sarsa_p20(self, start_episode, max_episodes, solved_at, lr, gamma, epsilon, min_epsilon, render, seed):
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

    def play(self, theta, episodes, render):
        frame_count = 0
        highest_score = 0

        for episode in range(1, episodes + 1):
            ep_score = 0

            features = self.env.reset()
            frame_count += 1
            if render:
                self.env.render()
            Q = features.dot(theta)
            action = np.argmax(Q)

            done = False
            while not done:
                next_features, reward, done, _ = self.env.step(action)
                frame_count += 1
                if render:
                    self.env.render()
                next_Q = next_features.dot(theta)
                next_action = np.argmax(next_Q)

                action = next_action
                ep_score += reward
            self.metrics.collect(episode, frame_count, highest_score, ep_score)
            print(f"{episode}/{episodes} done \tEpisode Score: {ep_score}")
        return self.metrics


def training_p20(game_name="BreakoutNoFrameskip-v4", seed=0, solved_at=40,
                 max_episodes=55000, lr=0.00025, gamma=0.99, max_epsilon=1, min_epsilon=0.1, render=False,
                 model_weights=None, remove_layers=-2, force_model=False,
                 metrics_filename='metrics_breakout_loaded.pkl', checkpoint_filename='checkpoint_breakout_loaded.pkl', theta_filename='theta_breakout_loaded.npy'):

    if remove_layers not in [-1, -2]:
        print('Can only remove the remaining Dense layer of size 512. HINT: Use remove_layers=-2 to remove this, or -1 to keep')
        return

    metrics = Metrics(metrics_filename)
    training = Checkpoint(checkpoint_filename).restore()
    if training.has_checkpoint:
        seed, theta, pretrained, num_features, remove_layers = training.get_training_properties()
    else:
        if model_weights is not None: pretrained = True
        else: pretrained = False

    # Warp and stack the frames, preprocessing: stack four frames and scale to smaller ratio
    env = wrap_deepmind(make_atari(game_name), frame_stack=True, scale=True)
    env.seed(seed=seed)
    p20_model = q_model(num_actions=env.action_space.n)

    if pretrained:
        try:
            p20_model.load_weights(model_weights)
        except:
            print(f'\nCould not find file for model weights: {model_weights}')
            if force_model:
                pretrained = False
                print('Using random weights instead')
            else:
                print('HINT (irreversible!): Set force_model=True in training_p20(..) to bypass using random weights instead')
                return

    # Stripping away the last remove_layers layers, and the layer for the actions
    p20_model = tf.keras.models.Model(inputs=p20_model.input, outputs=p20_model.layers[remove_layers - 1].output)
    if not training.has_checkpoint:
        # Setting the number of features according to the last layer
        num_features = p20_model.layers[-1].output_shape[1]
        theta = np.zeros(num_features * env.action_space.n)

    wrapped_env = LinearAtariWrapper(env=env, p20_model=p20_model, num_features=num_features)
    training.redefine(pretrained, num_features, remove_layers, seed)

    train_p20 = P20(env=wrapped_env, theta=theta, metrics=metrics, checkpoint=training)
    metrics, theta = train_p20.linear_sarsa_p20(
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
        theta_filename = game_name+'_theta.npy'
        print('Saved theta to file', theta_filename, 'instead')
    with open(theta_filename, 'wb') as theta_file:
        np.save(theta_file, theta)
    theta_file.close()

    df = pd.DataFrame.from_dict(metrics, orient='index')
    print(f'\nCollected metrics during training for game {game_name}:')
    print(df)
    print('')

    plt.plot(df['episode'], df['highest_score'])
    plt.plot(df['episode'], df['rolling_reward'])
    plt.show()

    plt.plot(df['frame'], df['highest_score'])
    plt.plot(df['frame'], df['rolling_reward'])
    plt.show()

def playing_p20(game_name="BreakoutNoFrameskip-v4", seed=0, episodes=100, render=True, model_weights=None, remove_layers=-2, force_model=False,
                theta_filename='theta_breakout_loaded.npy', metrics_filename='metrics_breakout_loaded.pkl'):

    if remove_layers not in [-1, -2]:
        print('Can only remove the remaining Dense layer of size 512. HINT: Use remove_layers=-2 to remove this, or -1 to keep')
        return

    # Warp and stack the frames, preprocessing: stack four frames and scale to smaller ratio
    env = wrap_deepmind(make_atari(game_name), frame_stack=True, scale=True)
    env.seed(seed=seed)

    metrics = Metrics(metrics_filename)
    p20_model = q_model(num_actions=env.action_space.n)

    if model_weights is not None:
        try:
            p20_model.load_weights(model_weights)
        except:
            print(f'\nCould not find file for model weights: {model_weights}')
            if force_model:
                print('Using random weights instead')
            else:
                print('HINT (irreversible!): Set force_model=True in training_p20(..) to bypass using random weights instead')
                return

    # Stripping away the last remove_layers layers, and the layer for the actions
    p20_model = tf.keras.models.Model(inputs=p20_model.input, outputs=p20_model.layers[remove_layers - 1].output)

    # Setting the number of features according to the last layer
    num_features = p20_model.layers[-1].output_shape[1]
    try:
        theta = np.load(theta_filename)
    except:
        print('Could not load theta. Abort')
        return
    if theta.shape[0] != p20_model.layers[-1].output * env.action_space.n:
        print('Theta could not be used with this architecture. Abort')
        return

    wrapped_env = LinearAtariWrapper(env=env, p20_model=p20_model, num_features=num_features)
    play_p20 = P20(env=wrapped_env, theta=theta, metrics=metrics, checkpoint=None)
    metrics = play_p20.play(theta, episodes, render)

    df = pd.DataFrame.from_dict(metrics, orient='index')
    print(f'\nCollected metrics during training for game {game_name}:')
    print(df)
    print('')

    plt.plot(df['episode'], df['highest_score'])
    plt.plot(df['episode'], df['rolling_reward'])
    plt.show()

    plt.plot(df['frame'], df['highest_score'])
    plt.plot(df['frame'], df['rolling_reward'])
    plt.show()