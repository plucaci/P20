# <center>On learning to play ATARI games using convolutional feature vectors for linear function approximation </center>

## Installing Guide
> Move the **P20 directory** in **/content/drive/MyDrive/** to obtain the path **/content/drive/MyDrive/P20/** <br>
> This path is hard-coded in the notebooks to automatically any dependencies needed when you run the code cells. <br>
> **You must follow this convention**. <br><br>
>  Alternatively, should you wish to use your own machine, you will have to comment out the lines importing google.colab libraries. <br>
> Google Colaboratory has a free service offering GPU.

> **You will not be able to run the code without a GPU, as the models have been trained on a GPU.** 
## P20 Project Structure
> Each applied method is implemented in separate directories. <br>
> The following sub-directories exist for all 3 applied methods: <br>
> 1. **checkpoints**<br>To save **.pkl files** as checkpoint for state of training. <br>
> 2. **metrics**<br>Data collected following training saved as **.pkl files.** <br>
> 3. **theta / models** (as applicable)<br>Directory **models** hosts **PyTorch** models only! <br>
Directory **theta** stores the vector of weights **theta** as an independent NumPy array only for the Experimental Method and the Baseline Method, for each of their variants.
> 4. **Atari-Roms** <br>
This folder contains the games used by the benchmark. It is a dependency, do not dispose. <br> For more details, read this post: https://stackoverflow.com/a/67668461 <br><br>
**Training was done on Google Colab and, on some occasions, on Datalore. The computational resources offered by Colab can only be accessed through ipynb notebooks. <br>
IMPORTANT: Please note, given the above, the best design pattern decision was to opt for having multiple launchers per applied method for the possibility of training the models in concurrently and finding the right hyperparameters.**
## Applied Methods
### BaselineMethod
> The Baseline Method described in the final report. <br>
> **Encoder architecture from (Mnih et al., 2015) implemented using Keras.** <br>
> **2 different variants of the encoder, both without the last linear layers: with linear function approximation, and with nonlinear.**
> #### Nonlinear_Baseline_Method.ipynb
> 1. Launcher of the Nonlinear Baseline Method variant. <br>
> 2. Architecture definition in Keras for the pretrained DQN model (Mnih et al., 2015), with ablation for the non-linear baseline variant. <br>
> 3. Configuration of hyperparameters used in the (nonlinear) Baseline Method included in this notebook.
> #### Linear_Baseline_Method.ipynb
> 1. Launcher of the Linear Baseline Method variant. <br>
> 2. Architecture definition in Keras for the pretrained DQN model (Mnih et al., 2015), with ablation for the linear baseline variant.<br>
> 3. Configuration of hyperparameters used in the (linear) Baseline Method included in this notebook.
> #### keras-dqn-model_breakout.h5
> Model in Keras of the pretrained DQN for feature extraction. <br> Trained using the TensorFlow DQN implementation for Breakout from https://keras.io/examples/rl/deep_q_network_breakout/
> #### p20.py
> The RL agent training algorithm implemented in TensorFlow.
> #### linear_atari.py
> Wrapper of the Atari environment. Implemented as a facade design pattern. This keeps the RL training algorithm clean.<br>
> Calls through Keras the pretrained encoder to extract features each time the RL training algorithm calls env.step() & env.reset().<br>
> When the features are returned, it applies a mask to these so that the RL training algorithm only updates the weights corresponding to the action chosen by the epsilon-greedy policy.
### ExperimentalMethod
> The Experimental Method described in the final report. <br>
> **Encoder architecture from (Mnih et al., 2013) implemented using PyTorch. This forces the rest of the P20 implementation for this method to accommodate PyTorch over Keras.** <br>
> **Same encoder used in all trials**
> #### Probe-Supervised-Encoding AtariARI.ipynb <br>
> Notebook for Supervised State Representation Learning to pretrain an encoder to extract features from environment observations. <br>
> 1. Implements the supervised learning methodology from (Anand et al., 2020) with as many linear probes as state-variables chosen but with 1 encoder;
And 1 RL agent taking random steps in an environment to collect the observations for training. <br>
> 2. Configuration of hyperparameters for the encoder's supervised training included in this notebook.
> #### ./ExperimentalMethod/models/3-probes_breakout_supervised_encoder.pt
> PyTorch model for the encoder saved by Probe-Supervised-Encoding AtariARI.ipynb & trained using 3 state-variables available in Breakout to extract features.
> #### ./ExperimentalMethod/models/all-probes_breakout_supervised_encoder.pt
> PyTorch model for the encoder saved by Probe-Supervised-Encoding AtariARI.ipynb & trained using all state-variables available in Breakout to extract features.
> #### All-Probes-Supervised.ipynb
> 1. Launcher of the Experimental Method with the encoder trained using all state-variables available in Breakout.
> 2. Configuration of hyperparameters for the (all labels) Experimental Method included in this notebook.
> #### 3-Probes-Supervised.ipynb
> 1. Launcher of the Experimental Method with the encoder trained using only 3 state-variables (ball_x, ball_y, player_x) in Breakout.
> 2. Configuration of hyperparameters for the (3 labels) Experimental Method included in this notebook.
> #### p20.py
> 1. The RL agent training algorithm implemented in PyTorch. <br>
> 2. Architecture definition in PyTorch for the model from (Mnih et al., 2013), as the same encoder is used in both trials. <br>
> 3. AtariARI-style environment configuration function wrap_atariari
> #### linear_atari.py
> Same as the Baseline Method but accommodating PyTorch over Keras.
### DeepSarsa
> The Deep Sarsa & Deep Linear Sarsa Methods described in the final report. <br>
> **Encoder architecture from (Mnih et al., 2015) implemented using PyTorch.** <br>
> **2 different variants of the encoder: with linear function approximation, and with nonlinear.**
> #### Deep-Sarsa.ipynb
> 1. Launcher for Deep Sarsa. <br>
> 2. Architecture definition in PyTorch for the model from (Mnih et al., 2015). <br>
> 3. Configuration of hyperparameters for Deep Sarsa is included in this notebook
> #### Deep-Linear-Sarsa.ipynb
> 1. Launcher for Deep Linear Sarsa. <br>
> 2. Architecture definition in PyTorch for the model from (Mnih et al., 2015), with ablation of the last activation function ReLU. <br>
> 3. Configuration of hyperparameters for Deep Linear Sarsa is included in this notebook
> #### p20.py
> The RL agent training algorithm implemented in PyTorch.
> #### linear_atari.py
> Removed, as masking is done implicitly through backpropagation.
> #### ./DeepSarsa/models/
> Theta being the last layer from the encoder, the model is saved instead of a NumPy array, along with the state of the PyTorch optimiser.
### testing_rl
> Python files for validating the implementation of linear Sarsa on a toy example, the FrozenLake. <br>
> Credits to Dr Paulo Rauber for suggesting this and for providing these 3 resources. <br>
> #### model_free.py
> Implements 3 linear Sarsa variants, of which 1 has been provided and the remaining 2 are implementations used in this project. <br>
`linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None)` is a method confirmed to be working and provided by Dr Paulo Rauber as a testing guideline for convergence on the toy example. <br>
`linear_sarsa_p20(env, max_episodes, theta, lr, gamma, epsilon, seed, training=True)` is one of the two methods being validated and used in this project, with classic Stochastic Gradient Descent (SGD) optimisation for `theta`. The other method being validated and used in this project is doing SGD optimisation through the SGD PyTorch optimiser,  `linear_sarsa_p20_optim(env, start_episode, max_episodes, theta, lr, gamma, epsilon, seed)`.  <br>
Both methods converge on the toy example, validating the correctness of their implementations. `get_action(env, Q, epsilon, random)` is an auxiliary method for the epsilon-greedy policy, being called within the body functions of both methods being validated. <br>
> #### tabular.py
> File containing environment wrappers. This file has been provided and confirmed to be working by Dr Paulo Rauber. <br>
> #### frozen_lake.py
> Contains the definition of the `main()`, as a launcher for calling the tabular wrappers from tabular.py to prepare the FrozenLake environment for trainign; The 3 methods from above,  `linear_sarsa`, `linear_sarsa_p20`, and `linear_sarsa_p20_optim` have their invokation calls also in the body of `main()`. <br>
> **For running the validation process, simply execute the frozen_lake.py file and observe the convergence** on the "small lake" as the values should increase towards the exit point on the lake (bottom-right) and the arrows indicating the the path towards the exit. <br>
It is certain that such observation can be made about the results tagged "(original)" (i.e., corresponding to `linear_sarsa(..)`, the provided working implementation); And for validating the 2 implementations used in the project -- their results too should follow suite!
### DQN
> Model used for pretraining the encoder in the Baseline Methods, to extract convolutional features.
> #### deep_q_network_breakout.ipynb 
> **Adapted from https://keras.io/examples/rl/deep_q_network_breakout/** <br>
> Separately from the source, it implements a best-effort checkpoint system to resume training in case execution is terminated. <br>
> Configuration of hyperparameters for pretraning DQN is included in this notebook
> #### .DQN/h5_weights/keras-model_breakout.h5 & ./BaselineMethod/keras-dqn-model_breakout.h5
> The file containting the pretrained model used in the Baseline Method to extract convolutional features. 
> Same files with different names. Renamed to follow the convention in the notebooks - should not be changed!
> Saved along with the target model, every 50 episodes in case execution crashes. Recovery is not automatic, and requires a few function calls before resuming training DQN with the same files. No such occurrance recorded.
> #### ./DQN/h5_weights/all_episode_rewards.npy
> Each episodic reward is saved in a NumPy array, and dumped to disk every 50 episodes in .npy format. <br>
> Contents of the file are imported at the end of training to report performance on charts. <br>
> As with all other notebooks, including for DQN's, execute **only the first code cell** of the notebook to inspect the data reported. <br>

## Data Inspection
> Prior to release, all notebooks have been carefully tided up, so you can easily access their training output and inspect the data collected. <br>
> **To inspect the data collected**, simply open the notebook corresponding to the data you want to see, and execute **only the first code cell**. <br>
> **To inspect the training**, look up the output of the last code cell in a notebook.

## Reproducing Results and Executing Code Cells
> With the exception for Deep Sarsa and Deep Linear Sarsa, where training has been early stopped owing to computational constraints; Training has been completed in all other cases. <br>
You will have to change the values passed to training_p20() method (the checkpoint_filename, metrics_filename, and theta_filename) called in a launcher notebook, to start training from scratch. <br>
Instead, if you want to verify the algorithms, you can do so by running all code cells, beginning with the first, in any notebook. This will only show you the plots, checkpoints, and metrics, indicating the code is working. <br>
To verify Deep Sarsa and Deep Linear Sarsa, you can do the same as above, and then terminate the execution before completing 50 more episodes to avoid saving new weights and metrics. <br> <br>
**For purposes of reproducibility, the hyperparameters reported for RL training can be found in the launcher notebooks, and in rare cases in the p20.py files if a function. Other sets of hyperparameters can be found in the Probe-Supervised-Encoding notebook, or in the notebook for pretraining the ./DQN**

## Checkpoint System and Metrics Collection
> Computational constraints are in place for anyone using Google Colab, Datalore, AWS, etc. The consequences that follow can be catastrophic for the training progress, or validation & evaluation. Consider the platform suddenly stops the code execution after days of training and few episodes left; How can one then recover the lost learned weights of the model?
> The checkpoint systems exactly as explained below are only available for training the RL agent itself
(i.e., only when using a notebook listed as "launcher" under "Applied Methods". <br> 
**Both systems have been tested thoroughly** through multiple series of training prior to deploying any applied method, and **are highly effective to resume training, and for data collection, respectively**. See running_utils.py in the directory of an applied method for implementation specifics. <br>

### Checkpoint Dictionary Data Structure 
> **Every 50 episodes, beginning with the 100th episode**, the following data structure is saved to disk in a .pkl file. <br>
> **Saving occurs only after an episode has finished!**
> Training is resumed from episode start_episode+1 using these properties. <br>
> There is no history of checkpoints, and only the last checkpoint is saved.<br>
> This is part of the checkpoint system implemented to resume the training in case of execution being terminated, for any reasons. <br>
#### Baseline Method & Experimental Method
```
{
  "start_episode": numeric,
  "frame_count": numeric,
  "theta": list,
  "highest_score": numeric,
  "rolling_reward_window100": list,
  "seed": numeric,
  "num_features": numeric
}
```
#### Deep Sarsa & Deep Linear Sarsa Method
> For this method, the num_features to set the shape of theta is no longer necessary, as theta is a layer of the model. <br>
> Therefore, the state of the model and that of the optimiser are saved at the same time with the rest of the following data structure, and loaded again when training is resumed with episode start_episode+1. <br>
> PyTorch saves the model as "./DeepSarsa/models/model_"+theta_filename and the optimiser as "./DeepSarsa/models/optim_"+theta_filename with .pt extensions. <br>
> Nothing else changes. This is part of the checkpoint system only for this method. See ./DeepSarsa/running_utils.py for more details.<br>
```
{
  "start_episode": numeric,
  "frame_count": numeric,
  "highest_score": numeric,
  "rolling_reward_window100": list,
  "seed": numeric,
}
```
### Metrics Dictionary Data Structure
> **Every 50 episodes, beginning with the 100th episode**, the following data structure is saved to disk in a .pkl file. <br>
> **Saving occurs only after an episode has finished!** <br>
> The tag < episode > is a key and the episode which holds the tuple in a larger dictionary.
> In contrast to saving a checkpoint, this tuple is appended to a larger dictionary containing more such tuples, then saved to disk.<br>
> This is part of the data collection system implemented to save the following metrics for analysis.
```
"<episode>": {
  "episode": numeric,
  "frame": numeric,
  "highest_score": numeric,
  "rolling_reward": numeric,
}
```
___________
## References
*Anand, A. et al., 2020. Unsupervised State Representation Learning in Atari, s.l.: arXiv:1906.08226v6.<br>
Mnih, V. et al., 2013. Playing Atari with Deep Reinforcement Learning, s.l.: arXiv:1312.5602v1.<br>
Mnih, V. et al., 2015. Human-level control through deep reinforcement learning. Nature, 518(7540), pp. 529-533.<br>*
Elfwing, S., Uchibe, E. & Doya, K., 2017. Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning, s.l.: arXiv:1702.03118v3.
