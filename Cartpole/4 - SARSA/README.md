# SARSA for Cart Pole Problem

## Introduction

This project applies the State-Action-Reward-State-Action (SARSA) reinforcement learning algorithm to the classic "Cart Pole" problem. Our objective is to balance a pole, hinged to a moving cart, by strategically moving the cart left or right.


## Objectives

- Implement the SARSA algorithm using PyTorch
![SARSA_algorithm](assets/SARSA.png)
- Train the SARSA model to maintain the balance of the pole on the moving cart for extended periods.
- Evaluate the performance and learning stability of the SARSA algorithm.

## Table of Contents

- [Introduction](#introduction)
- [Objectives](#objectives)
- [Setup](#setup)
- [SARSA Implementation](#sarsa-implementation)
  - [Agent Design](#agent-design)
  - [Network Architecture](#network-architecture)
- [Training Process](#training-process)
- [Testing](#testing)
- [Results](#results)
- [Conclusions](#conclusions)

## Setup

**Running the Notebook in Google Colab**
- The notebook is designed for easy execution in Google Colab, requiring no additional setup other than a Google account and internet access.😊

### Prerequisites
To run this project locally, you need to install the following Python packages. This setup ensures you have all the required libraries:

```bash
pip install gymnasium
pip install torch
pip install matplotlib
pip install renderlab
```

## Implementing SARSA Components

This section outlines the implementation of the SARSA algorithm's architecture, which is structured around several key components that each play a critical role in the learning process:

### Memory Class

The `Memory` class is responsible for storing experiences that the SARSA agent encounters, which are later used to update the agent's learning model, ensuring a diverse set of experiences influence the learning process.

```py
class Memory:
    def __init__(self, capacity):
        # Initialize replay memory with a specified capacity.
    def store(self, state, action, next_state, next_action, reward, done):
        # Store an experience tuple in the replay memory.
    def get_all(self):
        # Retrieve all stored experiences for learning updates.
    def clear(self):
        # Clear all contents of the memory to reset the learning history.
```

### SARSA Network Class

The `SARSA_Network` defines the neural network architecture used to estimate Q-values for each action given a state input. This component is crucial for the SARSA agent to evaluate possible actions from each state.

```py
class SARSA_Network(nn.Module):
    def __init__(self, num_actions, input_dim):
        # Initialize neural network layers and setup architecture.
    def forward(self, x):
        # Process input state to output Q-values for each action.
    def _initialize_weights(self):
        # Initialize weights to facilitate effective learning.
```

### SARSA Agent Class

The `SARSA_Agent` manages the interaction with the environment, decision making based on the current policy, and updates the policy based on observed transitions.

```py
class SARSA_Agent:
    def __init__(self, env, epsilon_max, epsilon_min, epsilon_decay, clip_grad_norm, learning_rate, discount, memory_capacity):
        # Setup agent with environment and learning parameters.
    def select_action(self, state):
        # Select an action using an epsilon-greedy approach for the given state.
    def learn(self, done):
        # Update the policy based on stored experiences and observed rewards.
    def update_epsilon(self):
        # Gradually decrease epsilon to reduce the exploration rate over time.
    def save(self, path):
        # Save the model's state dictionary for later use.
    def hard_update(self):
        # Synchronize the weights of the target network with the main network.
```

### Model TrainTest Class

The `Model_TrainTest` manages the full lifecycle of training and testing the SARSA agent, coordinating environment interactions and systematic improvements based on a set of training hyperparameters.

```py
class Model_TrainTest:
    def __init__(self, hyperparams):
        # Initialize with hyperparameters and set up the environment and agent.
    def state_preprocess(self, state, num_states):
        # Convert state information for network input, potentially normalizing.
    def train(self):
        # Execute the training process, modifying rewards to improve learning efficacy.
    def test(self, max_episodes):
        # Evaluate the agent's policy against new scenarios without further exploration.
    def plot_training(self, episode):
        # Visualize training progress with plots for rewards, losses, and epsilon decay.
```

## Results and Performance Analysis

After extensive training, our SARSA agent has shown impressive progress and efficiency in solving the CartPole problem. This section discusses the agent's performance throughout different stages of training and its ability to generalize during the testing phases.

### Training Progress

The SARSA agent was trained over 30,000 episodes with the goal of maximizing the pole's balance duration on the cart. The learning process is quantified through plots that illustrate the evolution of rewards, losses, and the agent's decision-making epsilon parameter over time.

#### Training Plots
<table>
  <tr>
    <td>Loss Plot<br><img src="assets/Loss_plot.png" alt="Loss Plot" width="320px"></td>
    <td>Reward Plot<br><img src="assets/reward_plot.png" alt="Reward Plot" width="320px"></td>
    <td>Epsilon Decay Plot<br><img src="assets/Epsilon_plot.png" alt="Epsilon Decay Plot" width="320px"></td>
  </tr>
</table>

### Testing Phase

Testing of the trained models at different epochs (10, 15,000, and 30,000) showcases the agent's improvement and stability over time. The results from these tests confirm the model's robustness and ability to generalize the learned policy to unseen scenarios.

- **Epoch 10**: The early stage of learning, where the agent's policy is relatively underdeveloped, achieving only short durations of pole balancing.
- **Epoch 15,000**: Midway through training, the agent shows significant improvements in strategy and stability.
- **Epoch 30,000**: A fully trained agent demonstrating optimal performance and decision-making capabilities.

### Visualizing Agent Performance

Animated GIFs and video sequences from test runs provide visual confirmation of the agent's competence. These visual aids illustrate how the agent effectively balances the pole, adapting to different initial conditions and disturbances.

#### Performance Videos

<table>
  <tr>
    <td>Epoch 10<br><img src="assets/10epoch.gif" alt="Epoch 10 Performance" width="240px"></td>
    <td>Epoch 15000<br><img src="assets/15000epoch.gif" alt="Epoch 15000 Performance" width="240px"></td>
    <td>Epoch 30000<br><img src="assets/30000epoch.gif" alt="Epoch 30000 Performance" width="240px"></td>
  </tr>
</table>

### Summary

The SARSA implementation has successfully demonstrated its ability to learn and adapt effectively to the Cart Pole challenge. The detailed plots and animations confirm the algorithm's learning efficacy and robustness. Through progressive training, the agent showcased significant improvements in strategy and performance, achieving high stability and excellent accuracy in balancing the pole. This success not only underscores the potential of SARSA in dynamic decision-making environments but also sets a benchmark for future enhancements and applications.


## Conclusions

The project demonstrates the application of the SARSA algorithm to a classic problem in reinforcement learning. The results show how the algorithm performs under different conditions and provide insights into its learning capabilities and stability.
