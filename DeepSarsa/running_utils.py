import pickle
import pandas as pd
import time
from collections import deque

import torch

class Checkpoint:
    def __init__(self, checkpoint_filename, theta_filename, model, optimizer):
        self.checkpoint_filename = checkpoint_filename
        self.theta_filename = theta_filename
        self.model = model
        self.optimizer = optimizer

        self.has_checkpoint = False
        self.checkpoint = dict()
        try:
            self.checkpoint = pd.read_pickle(self.checkpoint_filename)
            self.checkpoint['rolling_reward_window100'] = deque(iterable=self.checkpoint['rolling_reward_window100'], maxlen=len(self.checkpoint['rolling_reward_window100']))

            print(f'\nUsing previously saved checkpoint:')
            print(pd.DataFrame.from_dict(self.checkpoint, orient="index"))
            print('')

            self.has_checkpoint = True
        except:
            if self.checkpoint_filename is None:
                self.checkpoint_filename = time.strftime('%d-%m-%Y_%H-%M-%S') + '_checkpoint.pkl'
                print('Saving checkpoint to file', self.checkpoint_filename, 'instead')

    def get_counters(self):
        return self.checkpoint["start_episode"], self.checkpoint["frame_count"], self.checkpoint["highest_score"], self.checkpoint["rolling_reward_window100"]

    def get_training_properties(self):
        return self.checkpoint["seed"]

    def collect(self, episode, frame_count, highest_score, rolling_reward_window100):
        self.checkpoint['start_episode'] = episode
        self.checkpoint['frame_count'] = frame_count
        self.checkpoint['highest_score'] = highest_score
        self.checkpoint['rolling_reward_window100'] = rolling_reward_window100

        torch.save(self.model.state_dict(), "./models/model_"+self.theta_filename)
        torch.save(self.optimizer.state_dict(), "./models/optim_"+self.theta_filename)
        
        with open(self.checkpoint_filename, "wb") as checkpoint_file:
            pickle.dump(self.checkpoint, checkpoint_file)
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
                self.metrics_filename = time.strftime('%d-%m-%Y_%H-%M-%S') + '_metrics.pkl'
                print('Saving metrics to file', self.metrics_filename, 'instead')

    def collect(self, episode, frame_count, highest_score, rolling_reward):
        self.metrics[str(episode)] = dict()
        self.metrics[str(episode)]['episode'] = episode
        self.metrics[str(episode)]['frame'] = frame_count
        self.metrics[str(episode)]['highest_score'] = highest_score
        self.metrics[str(episode)]['rolling_reward'] = rolling_reward
        with open(self.metrics_filename, "wb") as metrics_file:
            pickle.dump(self.metrics, metrics_file)
        metrics_file.close()