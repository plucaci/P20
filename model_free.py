import numpy as np


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()

        q = features.dot(theta)

        if random_state.rand() < epsilon[i]:
            a = random_state.choice(env.n_actions)
        else:
            qmax = max(q)
            best = [a for a in range(env.n_actions) if np.allclose(qmax, q[a])]
            a = random_state.choice(best)

        done = False
        while not done:
            next_features, r, done = env.step(a)

            next_q = next_features.dot(theta)

            if random_state.rand() < epsilon[i]:
                next_action = random_state.choice(env.n_actions)
            else:
                qmax = max(next_q)
                best = [na for na in range(env.n_actions) if np.allclose(qmax, next_q[na])]
                next_action = random_state.choice(best)

            td = r + gamma*next_q[next_action] - q[a]
            theta = theta + eta[i]*td*features[a]

            features = next_features
            q = features.dot(theta)
            a = next_action

    return theta

### P20's RL Testing ###

def get_action(env, Q, epsilon, random):
    if random.rand() < epsilon:
        a = random.choice(env.n_actions)
    else:
        Q_max = max(Q)
        best = [a for a in range(env.n_actions) if np.allclose(Q_max, Q[a])]
        a = random.choice(best)
    return a


def linear_sarsa_p20(env, max_episodes, theta, lr, gamma, epsilon, seed, training=True):

    random_state = np.random.RandomState(seed)

    lr = np.linspace(lr, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    for episode in range(max_episodes):
        features = env.reset()

        Q = features.dot(theta).reshape(-1, 1)
        action = get_action(env, Q, epsilon[episode], random_state)

        ep_score = 0
        done = False
        while not done:
            next_features, reward, done = env.step(action)

            next_Q = next_features.dot(theta).reshape(-1, 1)
            next_action = get_action(env, next_Q, epsilon[episode], random_state)

            if training:
                temp_diff = reward + (gamma * next_Q[next_action]) - Q[action]
                theta += lr[episode] * temp_diff * features[action]

            features = next_features
            Q = features.dot(theta).reshape(-1, 1) # Q for current state,
            # from dot product with re-fined theta from above, in training
            action = next_action

            ep_score += reward

    return theta