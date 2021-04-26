import numpy as np


class LinearAtariWrapper:
    def __init__(self, env, p20_model, num_features):
        self.env = env
        self.num_actions = self.env.action_space.n

        self.p20_model = p20_model
        print(self.p20_model.summary())
        print('')

        self.features_shape = (self.env.action_space.n, num_features * self.env.action_space.n)
        self.null_features = np.zeros(shape=(num_features,))

    def get_features(self, state):
        conv_features = self.p20_model(np.expand_dims(np.asarray(state).astype(np.float64), axis=0))
        conv_features = np.squeeze(conv_features)

        ### Null-Feature Encoding ###
        features = np.zeros(shape=self.features_shape)
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

    def reset(self):
        return self.get_features(self.env.reset())

    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        return self.get_features(state), reward, done

    def render(self):
        self.env.render()