import tensorflow as tf

import numpy as np

import pickle

import pandas as pd
import matplotlib.pyplot as plt

from collections import deque

from baselines.common.atari_wrappers import make_atari, wrap_deepmind

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
    def __init__(self, env=None, game_name="BreakoutNoFrameskip-v4", seed=0,
                 p20=None, remove_layers=-1, pretrained=False, theta=None, num_features=None, null_features=None,
                 metrics_filename='metrics_breakout_loaded.pkl', checkpoint=None, checkpoint_filename='checkpoint_breakout_loaded.pkl'):

        self.game_name = game_name
        self.env = env

        self.seed = seed

        self.p20 = p20
        self.pretrained = pretrained
        self.remove_layers = remove_layers

        self.theta = theta
        self.num_features = num_features
        self.null_features = null_features
        # self.actions_identity = np.eye(self.env.action_space.n, self.env.action_space.n) # Identity matrix for one-hot encoded actions

        self.checkpoint = checkpoint
        self.checkpoint_filename = checkpoint_filename

        self.metrics = dict()
        self.metrics_filename = metrics_filename
        try:
            self.metrics = pd.read_pickle(self.metrics_filename)
            print(f'\nFound and using previously collected metrics during training for game {self.game_name}:')
            print(pd.DataFrame.from_dict(self.metrics))
            print('')
        except:
            if self.metrics_filename is None:
                self.metrics_filename = game_name+'_metrics.pkl'
                print('Saving metrics to file', self.metrics_filename, 'instead')


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

    def get_features(self, state):
        conv_features = self.p20(np.expand_dims(np.asarray(state).astype(np.float64), axis=0))
        conv_features = np.squeeze(conv_features)

        ### Null-Feature Encoding ###
        features = np.zeros(shape=(self.env.action_space.n, self.num_features * self.env.action_space.n))
        features[0] = np.concatenate((conv_features, self.null_features, self.null_features, self.null_features))
        features[1] = np.concatenate((self.null_features, conv_features, self.null_features, self.null_features))
        features[2] = np.concatenate((self.null_features, self.null_features, conv_features, self.null_features))
        features[3] = np.concatenate((self.null_features, self.null_features, self.null_features, conv_features))
        return features
        #############################

        ### One-Hot Encoding ###
        # conv_features = np.tile(conv_features, (self.env.action_space.n, 1))
        # return np.concatenate((conv_features, self.actions_identity), axis=1)
        ########################

    def linear_sarsa_p20(self, start_episode, max_episodes, solved_at, lr, gamma, epsilon, min_epsilon, render):
        if self.checkpoint is None:
            frame_count = 1
            highest_score = 0
            rolling_reward_window100 = deque(maxlen=100)
        else:
            start_episode = self.checkpoint['episode'] + 1
            frame_count = self.checkpoint['frame_count']
            highest_score = self.checkpoint['highest_score']
            rolling_reward_window100 = deque(iterable=self.checkpoint['rolling_reward_window100'], maxlen=100)

        random_state = np.random.RandomState(self.seed)
        # lr = np.linspace(lr, 0, max_episodes)
        epsilon = np.linspace(epsilon, min_epsilon, max_episodes)

        for episode in range(start_episode, max_episodes + 1):
            state = self.env.reset()
            frame_count += 1
            if render: self.env.render()

            features = self.get_features(state)
            Q = features.dot(self.theta)
            action = self.get_action(Q, epsilon[episode - 1], random_state)

            ep_score = 0
            done = False
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                frame_count += 1
                if render: self.env.render()

                next_features = self.get_features(next_state)
                next_Q = next_features.dot(self.theta)
                next_action = self.get_action(next_Q, epsilon[episode - 1], random_state)

                temp_diff = reward + (gamma * next_Q[next_action]) - Q[action]
                self.theta += lr * temp_diff * features[action]

                state = next_state
                features = self.get_features(state)
                Q = features.dot(self.theta) # Q for current state, from dot product with re-fined theta from above, in training
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
                    self.collect(episode, frame_count, highest_score, rolling_reward, rolling_reward_window100)

                if rolling_reward >= solved_at:
                    print('Solved environment', self.game_name, 'in', episode, 'episodes')
                    break
            else:
                print(f"{episode}/{max_episodes} done \tEpisode Score: {ep_score} \tFrame count: {frame_count}")

        return self.theta

    def collect(self, episode, frame_count, highest_score, rolling_reward, rolling_reward_window100):
        self.metrics[str(episode)] = dict()
        self.metrics[str(episode)]['episode'] = episode
        self.metrics[str(episode)]['frame_count'] = frame_count
        self.metrics[str(episode)]['highest_score'] = highest_score
        self.metrics[str(episode)]['rolling_reward'] = rolling_reward
        with open(self.metrics_filename, "wb") as metrics_file:
            pickle.dump(self.metrics, metrics_file)
        metrics_file.close()

        self.checkpoint = dict()
        self.checkpoint['episode'] = episode
        self.checkpoint['frame_count'] = frame_count
        self.checkpoint['theta'] = self.theta
        self.checkpoint['pretrained'] = self.pretrained
        self.checkpoint['num_features'] = self.num_features
        self.checkpoint['remove_layers'] = self.remove_layers
        self.checkpoint['highest_score'] = highest_score
        self.checkpoint['rolling_reward_window100'] = rolling_reward_window100
        self.checkpoint['seed'] = self.seed
        with open(self.checkpoint_filename, "wb") as checkpoint_file:
            pickle.dump(self.checkpoint, checkpoint_file)
        checkpoint_file.close()

def training_p20(game_name="BreakoutNoFrameskip-v4", seed=0, solved_at=40,
                           max_episodes=55000, lr=0.00025, gamma=0.99, max_epsilon=1, min_epsilon=0.1, render=False,
                           model_weights=None, remove_layers=-2, force_model=False,
                           metrics_filename='metrics_breakout_loaded.pkl', checkpoint_filename='checkpoint_breakout_loaded.pkl', theta_filename='theta_breakout_loaded.npy'):

    if remove_layers not in [-1, -2]:
        print('Can only remove the remaining Dense layer of size 512. HINT: Use remove_layers=-2 to remove this, or -1 to keep')
        return

    # Warp and stack the frames, preprocessing: stack four frames and scale to smaller ratio
    env = wrap_deepmind(make_atari(game_name), frame_stack=True, scale=True)
    p20_model = q_model(num_actions=env.action_space.n)

    try:
        checkpoint = pd.read_pickle(checkpoint_filename)
        print(f'\nFound saved checkpoint during training for game {game_name}:')

        seed = checkpoint['seed']
        theta = checkpoint['theta']
        pretrained = checkpoint['pretrained']
        num_features = checkpoint['num_features']
        null_features = np.zeros(shape=(num_features,))
        remove_layers = checkpoint['remove_layers']
        print(checkpoint)
    except:
        checkpoint = None

        if model_weights is not None:
            pretrained = True
        else:
            pretrained = False

        if checkpoint_filename is None:
            checkpoint_filename = game_name + '_checkpoint.pkl'
            print('Saving checkpoint to file', checkpoint_filename, 'instead')

    try:
        if pretrained:
            p20_model.load_weights(model_weights)
    except:
        print(f'\nCould not find file for model weights: {model_weights}')
        if checkpoint is not None and pretrained:
            print('Checkpoint was saved while evaluating with a pretrained convolutional network')
        if force_model:
            pretrained = False
            print('Using random weights instead')
        else:
            print('HINT (irreversible!): Set force_model=True in training_p20(..) to bypass using random weights instead')
            return

    # Stripping away the last remove_layers layers, and the layer for the actions
    p20_model = tf.keras.models.Model(inputs=p20_model.input, outputs=p20_model.layers[remove_layers - 1].output)

    if checkpoint is None:
        # Setting the number of features according to the last layer
        num_features = p20_model.layers[-1].output_shape[1]

        theta = np.zeros(num_features * env.action_space.n)
        null_features = np.zeros(shape=(num_features,))

    train_p20 = P20(env, game_name, seed, p20_model, remove_layers, pretrained, theta, num_features, null_features, metrics_filename, checkpoint, checkpoint_filename)
    print(train_p20.p20.summary())
    print('')

    theta = train_p20.linear_sarsa_p20(
        start_episode = 1,
        max_episodes  = max_episodes,
        solved_at     = solved_at,
        lr            = lr,
        gamma         = gamma,
        epsilon       = max_epsilon,
        min_epsilon   = min_epsilon,
        render        = render
    )

    if theta_filename is None:
        theta_filename = train_p20.game_name+'_theta.npy'
        print('Saved theta to file', theta_filename, 'instead')
    with open(theta_filename, 'wb') as theta_file:
        np.save(theta_file, theta)
    theta_file.close()

    df = pd.DataFrame.from_dict(pd.read_pickle(metrics_filename), orient='index')
    print(f'\nCollected metrics during training for game {train_p20.game_name}:')
    print(df)
    print('')

    plt.plot(df['episode'], df['highest_score'])
    plt.plot(df['episode'], df['rolling_reward'])
    plt.show()

    plt.plot(df['frame_count'], df['highest_score'])
    plt.plot(df['frame_count'], df['rolling_reward'])
    plt.show()

with tf.device('/device:GPU:0'):
    training_p20(game_name="BreakoutNoFrameskip-v4", seed=0, solved_at=40,
                    max_episodes=1000, lr=0.00025, gamma=0.99, max_epsilon=1, min_epsilon=0.1, render=False,
                    model_weights=None, remove_layers=-2, force_model=False,
                    metrics_filename='./P20/metrics_breakout.pkl', checkpoint_filename='./P20/checkpoint_breakout.pkl', theta_filename='./P20/theta_breakout.npy')

"""
with tf.device('/device:GPU:0'):
    training_p20(game_name="BreakoutNoFrameskip-v4", seed=0, solved_at=40,
                 max_episodes=55000, lr=0.00025, gamma=0.99, max_epsilon=1, min_epsilon=0.1, render=False,
                 model_weights=None, remove_layers=-2, force_model=False,
                 metrics_filename='/content/drive/MyDrive/new_metrics_breakout.pkl', checkpoint_filename='/content/drive/MyDrive/new_checkpoint_breakout.pkl',
                 theta_filename='/content/drive/MyDrive/new_theta_breakout.npy')

    training_p20(game_name="BreakoutNoFrameskip-v4", seed=0, solved_at=40,
                 max_episodes=55000, lr=0.00025, gamma=0.99, max_epsilon=1, min_epsilon=0.1, render=False,
                 model_weights='/content/drive/MyDrive/model_breakout.h5', remove_layers=-2, force_model=False,
                 metrics_filename='/content/drive/MyDrive/new_metrics_breakout_loaded.pkl', checkpoint_filename='/content/drive/MyDrive/new_checkpoint_breakout_loaded.pkl',
                 theta_filename='/content/drive/MyDrive/new_theta_breakout_loaded.npy')
"""