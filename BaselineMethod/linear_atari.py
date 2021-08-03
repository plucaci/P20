import numpy as np


class LinearAtariWrapper:
    '''
    Environment wrapper for transforming the convolutional features such that,
    upon linear value function approximation between theta and these features, only a subset of weights from theta
    is used to approximate the action values.

    It is only used in the Baseline Method and the Experimental Method, for the reasons explained above aligned with the methodology discussed in the final report.

    It must be used on top of the layers of wrappers implemented using gym.Wrapper and returned by the baselines.common.atari_wrappers(..) method.


    It is essentially a facade class to hide the computations done upon calling environment.reset() and environment.step() each time, to keep the training algorithm clean.

    Parameters
    ----------
    env : gym.Wrapper
        An environment wrapped with the layers of wrappers returned by baselines.common.atari_wrappers
    p20_model : tensorflow.keras.models.Model
        This is the model resulting from the 2 ablations to the architecture. The resulting model must be of type tensorflow.keras.models.Model <br>
        It cannot ever be None. p20_model can be considered function phi() described in the report. <br>
        Also, the size of the model output must match the number of features given in the next parameter.
    num_features : int
        The number of convolutional features extracted by the model passed in the previous parameter. <br>
        The wrapper does not automatically retrieve this from the last layer of the model, thus it must match with the size of the last layer in the model.
    '''
    def __init__(self, env, p20_model, num_features):
        self.env = env
        self.num_actions = self.env.action_space.n

        self.p20_model = p20_model
        print(self.p20_model.summary())
        print('')

        self.features_shape = (self.env.action_space.n, num_features * self.env.action_space.n)
        self.null_features = np.zeros(shape=(num_features,))

    def get_features(self, state):
        '''
        Performs one-hot encoding on the convolutional features received as a set, with 3 sets of zeroes each the size of the features received.
        This method is essentially computing tau(phi(state)), as discussed in the methodology from the final report.

        Parameters
        ----------
        state : baselines.common.atari_wrappers.LazyFrames
            State of the environment at some time step, represented by frames of the video game. <br>
            In the Baseline Method, there are 4 stacked frames of 84x84, and the final shape of parameter 'state' must be (84, 84, 4)

        Returns
        ----------
        features : np.ndarray
            The set of features of shape (number_of_actions_in_env, number_of_actions_in_env * number_of_features),
            after performing the tau(phi(state)) set of array transformations described in the methodology from the final report
        '''
        conv_features = self.p20_model(np.expand_dims(np.asarray(state).astype(np.float64), axis=0))
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