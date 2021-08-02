import tensorflow as tf
from p20 import training_p20, playing_p20
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

"""
TODO List:

0. Randomly trained -- DONE:
       episode    frame  highest_score  rolling_reward
100        100    18249           1.26            1.26
150        150    27299           1.32            1.24
200        200    36839           1.44            1.38
250        250    46302           1.48            1.46
300        300    55495           1.52            1.37
        ...      ...            ...             ...
42650    42650  7780467           1.77            1.09
42700    42700  7791255           1.77            1.34
42750    42750  7801288           1.77            1.39
42800    42800  7811610           1.77            1.34
42850    42850  7821402           1.77            1.21

1. use channels first for pre-trained from (Surma, 2018) -- DONE
2. use Adam as the optimiser for theta
3. use e-greedy for frames, 2 million frames, 1 million greedy
"""

def q_model(num_actions=4):
    # Network defined by the Deepmind paper
    inputs = tf.keras.layers.Input(shape=(84, 84, 4,))

    # Convolutions on the frames on the screen
    layer1 = tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu", data_format="channels_last", kernel_initializer='he_normal', use_bias=True)(inputs)
    layer2 = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu", data_format="channels_last", kernel_initializer='he_normal', use_bias=True)(layer1)
    layer3 = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu", data_format="channels_last", kernel_initializer='he_normal', use_bias=True)(layer2)

    layer4 = tf.keras.layers.Flatten()(layer3)

    # The adjusted output layer, with no activation="relu" of shape 'output_shape'
    layer5 = tf.keras.layers.Dense(units=512, kernel_initializer='he_normal')(layer4)

    action = tf.keras.layers.Dense(num_actions, activation="linear", kernel_initializer='he_normal')(layer5)

    return tf.keras.Model(inputs=inputs, outputs=action)

def main(game_name, seed):

    with tf.device('/device:GPU:0'):
        env = wrap_deepmind(make_atari(game_name), episode_life=False, frame_stack=True, scale=True)
        env.seed(seed)
        DQN_model = q_model(num_actions=env.action_space.n)
        p20_model = tf.keras.models.Model(inputs=DQN_model.input, outputs=DQN_model.layers[-3].output) # Ablation of all Dense (i.e., linear) layers

        training_p20(env=env, seed=seed, solved_at=40, p20_model=p20_model,
                     max_episodes=60000, lr=0.00025, gamma=0.99, max_epsilon=1, min_epsilon=0.1, render=False,
                     metrics_filename='keras-he_normal+bias-rand_metrics_breakout.pkl',
                     checkpoint_filename='keras-he_normal+bias-rand_checkpoint_breakout.pkl',
                     theta_filename='keras-he_normal+bias-rand_theta_breakout.npy')

if __name__ == "__main__":
    main(game_name="BreakoutNoFrameskip-v4", seed=0)

    import pandas as pd
    import matplotlib.pyplot as plt

    metrics = './pretrained_metrics_breakout_last_to_last_channels.pkl'
    df = pd.DataFrame.from_dict(pd.read_pickle(metrics), orient='index')


    print(f'\nCollected metrics during training for game:')
    print(df)
    print('')

    plt.plot(df['episode'], df['highest_score'])
    plt.plot(df['episode'], df['rolling_reward'])
    plt.show()

    plt.plot(df['frame'], df['highest_score'])
    plt.plot(df['frame'], df['rolling_reward'])
    plt.show()