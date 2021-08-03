import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinearAtariWrapper:
    def __init__(self, env, p20_model, num_features):
        self.env = env
        self.num_actions = self.env.action_space.n

        self.p20_model = p20_model
        print(self.p20_model)
        print('')

        self.features_shape = (self.env.action_space.n, num_features * self.env.action_space.n)
        self.null_features = np.zeros(shape=(num_features,))

    def get_features(self, state):
        conv_features = torch.unsqueeze(torch.Tensor(state).permute(2,0,1).float().to(device), dim=0)
        with torch.no_grad():
            conv_features = self.p20_model(conv_features).detach().cpu().numpy()
        conv_features = np.squeeze(conv_features)

        features = np.zeros(shape=self.features_shape)
        features[0] = np.concatenate((conv_features, self.null_features, self.null_features, self.null_features))
        features[1] = np.concatenate((self.null_features, conv_features, self.null_features, self.null_features))
        features[2] = np.concatenate((self.null_features, self.null_features, conv_features, self.null_features))
        features[3] = np.concatenate((self.null_features, self.null_features, self.null_features, conv_features))
        return features

    def reset(self):
        return self.get_features(self.env.reset())

    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        return self.get_features(state), reward, done

    def render(self):
        self.env.render()