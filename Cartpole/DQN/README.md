# README - DQN for CartPole

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-P1I0lxPf2scs4ZyOFb0tostuokrdm-v?usp=sharing)
[![Python Version](https://img.shields.io/badge/Python-3.6%20|%203.7%20|%203.8-blue)](https://www.python.org/downloads/release/python-380/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)](https://github.com/MahanVeisi8/RL_practices/Cartpole/DQN/requirements.txt)

## Introduction
This repository contains the implementation of the Deep Q-Network (DQN) algorithm applied to the classic "Cart Pole" problem, which is a staple challenge in the field of reinforcement learning. Our project aims to demonstrate how DQN can effectively balance a pole on a moving cart through autonomous learning and decision-making processes.

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
- [Implementing DQN Components](#implementing-dqn-components)
  - [Replay Memory Class](#replay-memory-class)
  - [DQN Network Class](#dqn-network-class)
  - [DQN Agent Class](#dqn-agent-class)
  - [Model TrainTest Class](#model-traintest-class)
- [Results and Performance Analysis](#results-and-performance-analysis)
  - [Training Progress](#training-progress)
  - [Testing Phase](#testing-phase)
  - [Visualizing Agent Performance](#visualizing-agent-performance)
  - [Summary](#summary)

## Setup
The code is designed to run in a Python environment with essential machine learning and simulation libraries. You can execute the notebook directly in Google Colab using the badge link provided, which includes a pre-configured environment with all necessary dependencies.

### Prerequisites
To run this project locally, you need to install the following Python packages. This setup ensures you have all the required libraries:

```bash
pip install gymnasium
pip install torch
pip install matplotlib
pip install renderlab
```


## Implementing DQN Components

This section provides a detailed explanation of the DQN model's architecture, divided into several key components each responsible for a specific part of the learning process:

### Replay Memory Class

The `ReplayMemory` class efficiently manages and stores experiences, minimizing correlations between consecutive learning samples, which is crucial for the stability of our learning algorithms.

```py
    def __init__(self, capacity):
        # Initialize the memory buffer with a fixed size.
    def push(self, state, action, reward, next_state, done):
        # Store new experiences in the memory.
    def sample(self, batch_size):
        # Randomly sample a batch of experiences for training.
    def __len__(self):
        # Return the current number of experiences stored.
```

### DQN Network Class

The `DQN_Network` class defines the neural network architecture for approximating the Q-function, crucial for evaluating the best action to take in a given state.

```py
class DQN_Network(nn.Module):
    def __init__(self, input_dim, num_actions):
        # Setup the layers of the network with specified input dimensions and number of actions.
    def forward(self, x):
        # Define the forward pass to compute Q-values from the state inputs.
    def _initialize_weights(self):
        # Initialize network weights using appropriate schemes to ensure effective learning.
```

### DQN Agent Class

The `DQN_Agent` orchestrates the learning process, manages interactions with the environment, and updates network parameters based on the observed experiences.

```py
class DQN_Agent:
    def __init__(self, env, epsilon_max, epsilon_min, epsilon_decay, learning_rate, discount, memory_capacity):
        # Configure agent with environment, learning parameters, and exploration settings.
    def select_action(self, state):
        # Select an action using epsilon-greedy policy based on current Q-values.
    def learn(self, batch_size):
        # Perform a learning update using a batch of sampled experiences from memory.
    def update_epsilon(self):
        # Adjust the epsilon value for the epsilon-gready policy to balance exploration and exploitation.
    def save(self, path):
        # Save the current state of the network to a file.
    def hard_update(self):
        # Synchronize the weights of the target network with the main network.
```

### Model TrainTest Class

The `Model_TrainTest` class manages the training and testing processes, setting up the environment, and executing the training cycles according to specified hyperparameters.

```py
class Model_TrainTest:
    def __init__(self, agent, env, hyperparams):
        # Initialize with an agent, environment, and training/testing settings.
    def state_preprocess(self, state):
        # Process raw state information from the environment to fit the network input requirements.
    def train(self):
        # Run the training loop, collecting data, updating the agent, and logging results.
    def test(self, max_episodes):
        # Evaluate the agent's performance on unseen data without exploration moves.
    def plot_training(self):
        # Generate and save plots of rewards, losses, and other metrics to visualize the training progress.
```

## Results and Performance Analysis

After intensive training and testing phases, our DQN agent demonstrates remarkable progress and efficiency in solving the CartPole problem. This section outlines the agent's performance across different training epochs and highlights its ability to generalize during testing phases.

### Training Progress

The agent was trained over 1000 episodes with the aim of maximizing the pole's balance duration on the cart. The learning process is quantified through plots that display the evolution of rewards, losses, and the agent's decision-making epsilon parameter over time.

#### Training Plots
<table>
  <tr>
    <td>Loss Plot<br><img src="assets/Loss_plot.png" alt="Loss Plot" width="320px"></td>
    <td>Reward Plot<br><img src="assets/reward_plot.png" alt="Reward Plot" width="320px"></td>
    <td>Epsilon Decay Plot<br><img src="assets/Epsilon_plot.png" alt="Epsilon Decay Plot" width="320px"></td>
  </tr>
</table>

### Testing Phase

Testing of the trained models at different epochs (10, 500, and 1000) showcases the agent's improvement and stability over time. The results from these tests confirm the model's robustness and ability to generalize the learned policy to unseen scenarios.

- **Epoch 10**: Early stages of learning, where the agent's policy is still underdeveloped.
- **Epoch 500**: Midway through training, showing significant improvements in strategy and stability.
- **Epoch 1000**: Fully trained agent demonstrating optimal performance and decision-making capabilities.

### Visualizing Agent Performance

Animated GIFs and video sequences from test runs provide a visual confirmation of the agent's competence. These visual aids illustrate how the agent effectively balances the pole, adapting to different initial conditions and disturbances.

#### Performance Videos

<table>
  <tr>
    <td>Epoch 10<br><img src="assets/10epoch.gif" alt="Epoch 10 Performance" width="240px"></td>
    <td>Epoch 500<br><img src="assets/500epoch.gif" alt="Epoch 500 Performance" width="240px"></td>
    <td>Epoch 1000<br><img src="assets/1000epoch.gif" alt="Epoch 1000 Performance" width="240px"></td>
  </tr>
</table>

### Summary

This project successfully demonstrates the implementation of a Deep Q-Network to solve the Cart Pole challenge, a classic test in reinforcement learning. Our results, presented through detailed plots and animations, confirm the DQN's ability to learn and adapt effectively. Through progressive training, the model showcased significant improvements in strategy and performance, achieving stability and high accuracy in balancing the pole. This work not only validates the DQN approach but also sets the stage for further advancements and applications in dynamic decision-making environments.


