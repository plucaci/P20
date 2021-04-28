import tensorflow as tf
from p20 import training_p20, playing_p20


def main(kind='training'):
    with tf.device('/device:GPU:0'):
        if kind == 'training' or kind == 'both':
            training_p20(game_name="BreakoutNoFrameskip-v4", seed=0, solved_at=40,
                         max_episodes=1000, lr=0.00025, gamma=0.99, max_epsilon=1, min_epsilon=0.1, render=False,
                         model_weights=None, remove_layers=-2, force_model=False,
                         metrics_filename='./P20/metrics_breakout.pkl', checkpoint_filename='./P20/checkpoint_breakout.pkl', theta_filename='./P20/theta_breakout.npy')
            training_p20(game_name="BreakoutNoFrameskip-v4", seed=0, solved_at=40,
                         max_episodes=1000, lr=0.00025, gamma=0.99, max_epsilon=1, min_epsilon=0.1, render=False,
                         model_weights=None, remove_layers=-2, force_model=False,
                         metrics_filename='./P20/metrics_breakout.pkl', checkpoint_filename='./P20/checkpoint_breakout.pkl', theta_filename='./P20/theta_breakout.npy')
        if kind == 'playing' or kind == 'both':
            playing_p20(game_name="BreakoutNoFrameskip-v4", seed=0, episodes=100, render=True,
                        model_weights=None, remove_layers=-2, force_model=False,
                        theta_filename='./new_theta_breakout_loaded.npy')


if __name__ == "__main__":
    main()