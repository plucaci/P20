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
