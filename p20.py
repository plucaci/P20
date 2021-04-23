import tensorflow as tf

import numpy as np
import pickle
import pandas as pd

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


def p20_model():
    input_states = tf.keras.Input(shape=(FRAME_SIZE, FRAME_SIZE, K_FRAMES_STACKED,))

    layer1 = tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu", data_format="channels_last")(input_states)
    layer2 = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu", data_format="channels_last")(layer1)
    layer3 = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu", data_format="channels_last")(layer2)

    layer4 = tf.keras.layers.Flatten()(layer3)

    output_features = tf.keras.layers.Dense(units=512)(layer4)
    # The adjusted output layer, with no activation="relu" of shape 'output_shape'

    model = tf.keras.Model(inputs=input_states, outputs=output_features)
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.SGD())

    return model


def q_model(num_actions=4):
    # Network defined by the Deepmind paper
    inputs = tf.keras.layers.Input(shape=(FRAME_SIZE, FRAME_SIZE, K_FRAMES_STACKED,))

    # Convolutions on the frames on the screen
    layer1 = tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu", data_format="channels_last")(inputs)
    layer2 = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu", data_format="channels_last")(layer1)
    layer3 = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu", data_format="channels_last")(layer2)

    layer4 = tf.keras.layers.Flatten()(layer3)

    layer5 = tf.keras.layers.Dense(512, activation="relu")(layer4)
    action = tf.keras.layers.Dense(num_actions, activation="linear")(layer5)

    return tf.keras.Model(inputs=inputs, outputs=action)


class P20:
    def __init__(self, game_name="BreakoutNoFrameskip-v4"):

        # Warp the frames, grey scale, stack four frames and scale to smaller ratio
        self.env = wrap_deepmind(make_atari(game_name), frame_stack=True, scale=True)
        # self.actions_identity = np.eye(self.env.action_space.n, self.env.action_space.n)
        self.null_features = np.zeros(shape=(512, ))

        self.p20 = p20_model()

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
        features = np.zeros(shape=(self.env.action_space.n, 512 * self.env.action_space.n))
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

    def linear_sarsa_p20(self, start_episode, max_episodes, solved_at, theta, lr, gamma, epsilon, seed, render=True):
        highest_score = 0

        random_state = np.random.RandomState(seed)

        # lr = np.linspace(lr, 0, max_episodes)
        epsilon = np.linspace(epsilon, 0, max_episodes)

        frame_count = 0
        rolling_reward_window100 = deque(maxlen=100)
        for episode in range(start_episode, max_episodes):
            if render: self.env.render()
            state = self.env.reset()
            features = self.get_features(state)

            Q = features.dot(theta).reshape(-1, 1)
            action = self.get_action(Q, epsilon[episode], random_state)

            ep_score = 0
            done = False
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                if render: self.env.render()
                next_features = self.get_features(next_state)

                next_Q = next_features.dot(theta).reshape(-1, 1)
                next_action = self.get_action(next_Q, epsilon[episode], random_state)

                temp_diff = reward + (gamma * next_Q[next_action]) - Q[action]
                theta += lr * temp_diff * features[action]

                state = next_state
                features = self.get_features(state)

                # Q for current state, from dot product with re-fined theta from above, in training
                Q = features.dot(theta).reshape(-1, 1)
                action = next_action

                frame_count += 1
                ep_score += reward

            rolling_reward_window100.append(ep_score)
            rolling_reward = np.mean(rolling_reward_window100)
            highest_score = max(rolling_reward, highest_score)

            if len(rolling_reward_window100) == 100:
                print(f"{episode + 1}/{max_episodes} done \tEpisode Score: {ep_score}"
                      f"\t\tAvg Score 100 Episodes: {rolling_reward:2f}"
                      f"\tHighest Avg Score: {highest_score:2f}"
                      f"\t\tFrame count: {frame_count}")
            else:
                print(f"{episode + 1}/{max_episodes} done \tEpisode Score: {ep_score}"
                      f"\tFrame count: {frame_count}")

            if episode % 500 == 0:
                checkpoint = dict()
                checkpoint['episode'] = episode
                checkpoint['theta'] = theta

                with open("checkpoint.pkl", "wb") as cpoint:
                    pickle.dump(checkpoint, cpoint)
                cpoint.close()

            if rolling_reward >= solved_at:
                print(f"Solved in {episode + 1} episodes")
                break


def training_p20(game="BreakoutNoFrameskip-v4", seed=0, solved=40, theta_filename='theta_breakout.npy'):
    p20_train = P20(game_name=game)

    theta = np.zeros(512 + p20_train.env.action_space.n)
    p20_train.linear_sarsa_p20(
        start_episode= 0,
        max_episodes = 100000,
        solved_at    = solved,
        theta        = theta,
        lr           = 0.5,
        gamma        = 0.99,
        epsilon      = 0.5,
        seed         = seed,
        render       = False
    )

    with open(theta_filename, 'wb') as f:
        np.save(f, theta)
    f.close()


def preloaded_training_p20(game="BreakoutNoFrameskip-v4", seed=0, solved=40, num_actions=4,
                           model_weights='./model_breakout.h5', theta_filename='theta_loaded_breakout.npy'):
    ## Loading the the weights
    model = q_model(num_actions)
    model.load_weights(model_weights)

    ## Stripping away the last layer
    model2 = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
    model2.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.SGD())

    ## Using the loaded model without the last layer
    loaded_p20 = P20(game_name=game)
    loaded_p20.p20 = model2
    print(loaded_p20.p20.summary())

    try:
        checkpoint = pd.read_pickle('checkpoint.pkl')
        episode = checkpoint['episode'] +1
        theta = checkpoint['theta']
    except:
        theta = np.zeros(512 * loaded_p20.env.action_space.n)
        episode = 0
    loaded_p20.linear_sarsa_p20(
        start_episode= episode,
        max_episodes = 500000,
        solved_at    = solved,
        theta        = theta,
        lr           = 0.00025,
        gamma        = 0.99,
        epsilon      = 0.5,
        seed         = seed,
        render       = False
    )

    with open(theta_filename, 'wb') as f:
        np.save(f, theta)
    f.close()


def testing_p20(game="BreakoutNoFrameskip-v4", seed=0, num_actions=4,
                model_weights='./model_breakout.h5', theta_filename='theta_loaded_breakout.npy'):
    ## Loading theta
    with open(theta_filename, 'rb') as f:
        theta = np.load(f)
    f.close()

    ## Loading the the weights
    model = q_model(num_actions)
    model.load_weights(model_weights)

    ## Stripping away the last layer
    model2 = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
    model2.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.SGD())

    ## Using the loaded model without the last layer
    p20_test = P20(game_name=game)
    p20_test.p20 = model2
    print(p20_test.p20.summary())

    p20_test.linear_sarsa_p20(
        start_episode= 0,
        max_episodes = 2000,
        solved_at    = -1,
        theta        = theta,
        lr           = 0.5,
        gamma        = 0.99,
        epsilon      = 0.5,
        seed         = seed,
        render       = True
    )


with tf.device('/device:GPU:0'):
    preloaded_training_p20(
        game = "BreakoutNoFrameskip-v4",
        seed = 0,
        solved = 40,
        num_actions = 4,
        model_weights = './P20/model_breakout.h5',
        theta_filename = './P20/theta_loaded_breakout.npy'
    )

    """
    testing_p20(
        game = "BreakoutNoFrameskip-v4",
        seed = 0,
        num_actions = 4,
        model_weights = '/content/drive/MyDrive/model_breakout.h5',
        theta_filename = '/content/drive/MyDrive/theta_loaded_breakout.npy'
    )
    """