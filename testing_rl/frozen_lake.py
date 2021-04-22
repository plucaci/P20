from sys import path

path.append("frozen_lake.py")
path.append("tabular.py")

from tabular import FrozenLake, LinearWrapper
from model_free import linear_sarsa, linear_sarsa_p20

import numpy as np

def main():
    seed = 0

    # Big lake
    # lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
    #         ['.', '.', '.', '.', '.', '.', '.', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '.'],
    #         ['.', '.', '.', '.', '.', '#', '.', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '.'],
    #         ['.', '#', '#', '.', '.', '.', '#', '.'],
    #         ['.', '#', '.', '.', '#', '.', '#', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '$']]

    # Small lake
    lake =   [['&', '.', '.', '.'],
              ['.', '#', '.', '#'],
              ['.', '.', '.', '#'],
              ['#', '.', '.', '$']]


    print('# Model-free algorithms')
    max_episodes = 2000
    eta = 0.5
    epsilon = 0.5
    gamma = 0.9

    print('## Linear Sarsa (origin): Frozen Lake')
    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    linear_env = LinearWrapper(env)

    parameters = linear_sarsa(linear_env, max_episodes, eta, gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)
    print('')


    print('## Linear Sarsa (p20''s RL testing): Frozen Lake')
    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    linear_env = LinearWrapper(env)

    theta = np.zeros(linear_env.n_features)
    linear_sarsa_p20(env=linear_env, theta=theta,
                     max_episodes=max_episodes, lr=eta, gamma=gamma, epsilon=epsilon, seed=seed, training=True)
    policy, value = linear_env.decode_policy(theta)
    linear_env.render(policy, value)
    print('')

if __name__ == "__main__":
    main()