import pickle
import pandas as pd
import time
from collections import deque


class Checkpoint:
    def __init__(self, checkpoint_filename):
        self.checkpoint_filename = checkpoint_filename

        self.has_checkpoint = False

        self.start_episode = None
        self.frame_count = None
        self.theta = None
        self.pretrained = None
        self.num_features = None
        self.remove_layers = None
        self.highest_score = None
        self.rolling_reward_window100 = None
        self.seed = None

    def redefine(self, pretrained, num_features, remove_layers, seed):
        self.pretrained = pretrained
        self.num_features = num_features
        self.remove_layers = remove_layers
        self.seed = seed

    def get_counters(self):
        return self.start_episode, self.frame_count, self.highest_score, self.rolling_reward_window100

    def get_training_properties(self):
        return self.seed, self.theta, self.pretrained, self.num_features, self.remove_layers

    def restore(self):
        try:
            checkpoint = pd.read_pickle(self.checkpoint_filename)
            self.has_checkpoint = True

            print(f'\nUsing saved checkpoint during training:')
            print(checkpoint)
            print('')

            self.start_episode = checkpoint['start_episode']
            self.frame_count = checkpoint['frame_count']
            self.theta = checkpoint['theta']
            self.pretrained = checkpoint['pretrained']
            self.num_features = checkpoint['num_features']
            self.remove_layers = checkpoint['remove_layers']
            self.highest_score = checkpoint['highest_score']
            self.rolling_reward_window100 = deque(iterable=checkpoint['rolling_reward_window100'], maxlen=len(checkpoint['rolling_reward_window100']))
            self.seed = checkpoint['seed']
        except:
            if self.checkpoint_filename is None:
                self.checkpoint_filename = time.strftime('%d-%m-%Y_%H-%M-%S') + '_checkpoint.pkl'
                print('Saving checkpoint to file', self.checkpoint_filename, 'instead')

        return self

    def collect(self, episode, frame_count, theta, highest_score, rolling_reward_window100):
        checkpoint = dict()
        checkpoint['start_episode'] = episode
        checkpoint['frame_count'] = frame_count
        checkpoint['theta'] = theta
        checkpoint['pretrained'] = self.pretrained
        checkpoint['num_features'] = self.num_features
        checkpoint['remove_layers'] = self.remove_layers
        checkpoint['highest_score'] = highest_score
        checkpoint['rolling_reward_window100'] = rolling_reward_window100
        checkpoint['seed'] = self.seed
        with open(self.checkpoint_filename, "wb") as checkpoint_file:
            pickle.dump(checkpoint, checkpoint_file)
        checkpoint_file.close()


class Metrics:
    def __init__(self, metrics_filename):
        self.metrics_filename = metrics_filename

        self.metrics = dict()
        try:
            self.metrics = pd.read_pickle(self.metrics_filename)

            print(f'\nUsing previously collected metrics during training:')
            print(pd.DataFrame.from_dict(self.metrics))
            print('')
        except:
            if self.metrics_filename is None:
                metrics_filename = time.strftime('%d-%m-%Y_%H-%M-%S') + '_metrics.pkl'
                print('Saving metrics to file', metrics_filename, 'instead')

    def collect(self, episode, frame_count, highest_score, rolling_reward):
        self.metrics[str(episode)] = dict()
        self.metrics[str(episode)]['start_episode'] = episode
        self.metrics[str(episode)]['frame_count'] = frame_count
        self.metrics[str(episode)]['highest_score'] = highest_score
        self.metrics[str(episode)]['rolling_reward'] = rolling_reward
        with open(self.metrics_filename, "wb") as metrics_file:
            pickle.dump(self.metrics, metrics_file)
        metrics_file.close()