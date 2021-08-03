import pickle
import pandas as pd
import time
from collections import deque

import torch

'''

For this method, the num_features to set the shape of theta is no longer necessary, as theta is a layer of the model. 
Therefore, the state of the model and that of the optimiser are saved at the same time with the rest of the following data structure,
and loaded again when training is resumed with episode start_episode+1. 
PyTorch saves the model as "./DeepSarsa/models/model_"+theta_filename and the optimiser as "./DeepSarsa/models/optim_"+theta_filename with .pt extensions. 
Nothing else changes. This is part of the checkpoint system only for this method. See ./DeepSarsa/running_utils.py for more details.
---------------------------------------------------------------------------
Checkpoint Dictionary Data Structure: Deep Sarsa & Deep Linear Sarsa Method
---------------------------------------------------------------------------
{
  "start_episode": numeric,
  "frame_count": numeric,
  "highest_score": numeric,
  "rolling_reward_window100": list,
  "seed": numeric,
}

------------------------------------------------------------------------
Metrics Dictionary Data Structure: Deep Sarsa & Deep Linear Sarsa Method
------------------------------------------------------------------------
The tag <episode> is a key and the episode which holds each tuple in a larger dictionary.

"<episode>": {
  "episode": numeric,
  "frame": numeric,
  "highest_score": numeric,
  "rolling_reward": numeric,
}
'''

class Checkpoint:
    '''
    The Checkpoint-saving system to aid resuming training if stopped.

    Checkpoint is done every 50 episodes, beginning with episode 100, and the contents are saved in a dictionary then dumped to disk in a .pkl file.

    There is no history of checkpoints.

    For the data collected at checkpoint, see the data structure listed "Checkpoint Dictionary Data Structure" at the top of ./ExperimentalMethod/running_utils.py

    Saving occurs only after an episode has finished!
    Hence, training should be resumed from episode start_episode+1 using these properties.

    Upon instantiation, it creates a new dictionary object in which to store the state of the previous training, and checks if parameter checkpoint_filename is None.
    If the checkpoint_filename is None, the flag remains set to False, but it will get ready its own timestamped filename to create a file for and at the next checkpoint.
    See the parameter checkpoint_filename for the timestamp format and the path.

    Otherwise, it attempts to read the contents of the file.
    If it succeeds in reading the contents, the object's has_checkpoint flag is set to True, and stores the contents in the dictionary object it had just created.
    To restore the checkpoint, call get_training_properties() and get_counters().

    However, if it fails to read the contents, in case file is corrupt or not on disk, new training is assumed and the has_checkpoint flag remains set to False by default.
    As such, before attempting to restore the state of the previous training, you must always check if the object's flag has_checkpoint is set to True.

    If the checkpoint_filename parameter is set, it is always assumed throughout the lifecycle of a Checkpoint object as the preferred path in which to dump a new checkpoint (by overwriting).

    This is part of the checkpoint system implemented to resume the training in case of execution being terminated, for any reasons.

    Parameters
    ----------
    checkpoint_filename : str
        The path of the checkpoint file with the .pkl extension <br>
        If None, new file is created at the next checkpoint, with a timestamped filename under format '%d-%m-%Y_%H-%M-%S_checkpoint.pkl' on the path given by your machine's Python configuration <br>
        If set, it always attempts to read the file contents and replaces with a new checkpoint every 50 episodes (only if after episode 100 due to the rolling window of 100 rewards being stored as well) <br>
        To see what the structure of this file is and how the checkpoint-saving system is implemented, refer to the ./ExperimentalMethod/running_utils.py file.
    '''
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
        '''
        Getter to return the counters for initialisation before resuming training. Use only if there is training to be resumed.

        Returns
        ----------
        tuple[int, int, int, deque]
            start_episode - the episode completed before checkpoint. start_episode+1 is the episode training is meant to be resumed from <br>
            frame_count - the last time step completed before checkpoint. frame_count+1 is the time step training is meant to be resumed from <br>
            highest_score - the highest average of cumulative scores over the last 100 episodes, prior to the checkpoint <br>
            rolling_reward_window100 - the cumulative scores of the last 100 episodes before checkpoint (as a deque object with rolling window of size 100)
        '''
        return self.checkpoint["start_episode"], self.checkpoint["frame_count"], self.checkpoint["highest_score"], self.checkpoint["rolling_reward_window100"]

    def get_training_properties(self):
        '''
        Getter to return initialisation properties of the environment: the seed.

        For Deep Sarsa & Deep Linear Sarsa, the number of features and theta became redundant due to approximating with the whole model.

        Returns
        ----------
        seed : int
            seed to initialise the environment with; Also the seed for initialising the random state for the Epsilon-Greedy Policy <br>
        '''
        return self.checkpoint["seed"]

    def collect(self, episode, frame_count, highest_score, rolling_reward_window100):
        '''
        Collector of last episode completed, last time step completed, theta after its last optimisation update, the highest score, and last 100 episode rewards.

        The method stores the values of these parameters in a dictionary whose structure can be found listed "Checkpoint Dictionary Data Structure" at the top of ./ExperimentalMethod/running_utils.py;
        Then, it dumps the dictionary to disk in a file of format .pkl whose filename was either, specified in the checkpoint_filename attribute of class Checkpoint, or timestamped.
        The method closes the file soon after.

        Additionally, in place of theta, the collector also saves the model's weights and the optimiser's state, as "./models/model_"+theta_filename & "./models/optim_"+theta_filename

        See the Checkpoint class for how files are handled during collections.

        Parameters
        ----------
        start_episode : int
            The episode completed before checkpoint. start_episode+1 is the episode training is meant to be resumed from <br>
        frame_count : int
            The last time step completed before checkpoint. frame_count+1 is the frame training is meant to be resumed from <br>
        highest_score : int
            The highest rolling average of episode scores over the last 100 episodes, prior to the checkpoint <br>
        rolling_reward_window100 : deque
            The cumulative scores of the last 100 episodes (as a deque object with rolling window of size 100)
        '''
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
    '''
    The metrics collection system to for training performance inspection
    As such, data is always collected, even if previous metrics are missing but a checkpoint is available.

    Metrics are collected every 50 episodes, beginning with episode 100, and the contents are saved in a dictionary then dumped to disk in a .pkl file.
    In contrast to saving a checkpoint, every set of metrics collected is appended to the previous in a larger dictionary then dumped to disk.
    Saving occurs only after an episode has finished!

    For the data collected per a single episode, see the data structure listed "Metrics Dictionary Data Structure" at the top of ./ExperimentalMethod/running_utils.py

    Upon instantiation, it creates a new dictionary object in which to store all previous sets of metrics saved, and checks if parameter checkpoint_filename is None.
    If the checkpoint_filename is None, it will get ready its own timestamped filename to create a file for and at the next collection.
    See the parameter metrics_filename for the timestamp format and the path.

    Otherwise, it attempts to read the contents of the file.
    If it succeeds in reading the contents, it stores all previous metrics saved in the dictionary object it had just created.

    However, if it fails to read the contents, in case file is corrupt or not on disk, it will always collect the last set of metrics generated in training.

    If the metrics_filename parameter is set, it is always assumed throughout the lifecycle of a Metrics object as the preferred path in which to newly collected metrics (by overwriting).

    This is part of the data collection system implemented to save metrics for analysis.

    Parameters
    ----------
    metrics_filename : str
        The path of the metrics file with the .pkl extension <br>
        If None, new file is created at the next checkpoint, with a timestamped filename under format '%d-%m-%Y_%H-%M-%S_metrics.pkl' on the path given by your machine's Python configuration <br>
        If set, it always attempts to read the dictionary with previous metrics saved in the .pkl file and continues to append newly collected metrics every 50 episodes from episode 100 onwards <br>
        To see what the structure of this file is and how the metrics collection system is implemented, refer to the ./ExperimentalMethod/running_utils.py file.
    '''
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
        '''
        Collector of last episode completed, last time step completed, the highest average of 100 episode scores, and average over 100 episode scores.

        The method appends, using the episode as key, all the values of these parameters in a nested dictionary.
        The data structure of the dictionary holding the values can be found listed "Metrics Dictionary Data Structure" at the top of ./ExperimentalMethod/running_utils.py;

        Then, it dumps the nested dictionary with the update to the disk in a file of format .pkl whose filename was either, specified in the metrics_filename attribute of class Metrics, or timestamped.
        The method closes the file soon after.

        See the Metrics class for how files are handled during collections.

        Parameters
        ----------
        episode : int
            The episode completed before collection. Also the key to append the values to the nested dictionary
        frame_count : int
            The last time step completed before collection.
        highest_score : int
            The highest rolling average of episode scores over the last 100 episodes, prior to the collection
        rolling_reward : int
            The rolling average over 100 episode scores
        '''
        self.metrics[str(episode)] = dict()
        self.metrics[str(episode)]['episode'] = episode
        self.metrics[str(episode)]['frame'] = frame_count
        self.metrics[str(episode)]['highest_score'] = highest_score
        self.metrics[str(episode)]['rolling_reward'] = rolling_reward
        with open(self.metrics_filename, "wb") as metrics_file:
            pickle.dump(self.metrics, metrics_file)
        metrics_file.close()