# README - DQN for CartPole

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-P1I0lxPf2scs4ZyOFb0tostuokrdm-v?usp=sharing)
[![Python Version](https://img.shields.io/badge/Python-3.6%20|%203.7%20|%203.8-blue)](https://www.python.org/downloads/release/python-380/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)](https://github.com/your-username/your-repository/blob/main/requirements.txt)

## Introduction
This repository contains the implementation of the Deep Q-Network (DQN) algorithm applied to the classic "Cart Pole" problem, which is a staple challenge in the field of reinforcement learning. The objective is to develop a model that can autonomously balance a pole on a moving cart, demonstrating the capabilities of DQN in maintaining system equilibrium.

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

#### Epsilon Decay Plot
Shows the decrease in epsilon value, reflecting the transition from exploration to exploitation. This plot helps in understanding how the agent gradually shifts its strategy to rely more on learned behaviors rather than random actions.

![Epsilon Decay Plot](path-to-epsilon-plot.png)

#### Loss Plot
Illustrates the changes in learning loss over time, which provides insights into the network's learning efficiency and convergence behavior.

![Loss Plot](path-to-loss-plot.png)

#### Reward Plot
Captures the total rewards accumulated by the agent in each episode, offering a direct measure of performance and the agent's ability to maintain the pole's balance over time.

![Reward Plot](path-to-reward-plot.png)

### Testing Phase

Testing of the trained models at different epochs (10, 500, and 1000) showcases the agent's improvement and stability over time. The results from these tests confirm the model's robustness and ability to generalize the learned policy to unseen scenarios.

- **Epoch 10**: Early stages of learning, where the agent's policy is still underdeveloped.
- **Epoch 500**: Midway through training, showing significant improvements in strategy and stability.
- **Epoch 1000**: Fully trained agent demonstrating optimal performance and decision-making capabilities.

### Visualizing Agent Performance

Animated GIFs and video sequences from test runs provide a visual confirmation of the agent's competence. These visual aids illustrate how the agent effectively balances the pole, adapting to different initial conditions and disturbances.

#### Performance Video at Epoch 1000

A video from the final testing phase, displaying the agent's refined skills in balancing the pole on the cart without human intervention.

[View Performance Video](link-to-video)

### Summary

The graphs and visual content underline the DQN agent's successful learning curve and its efficiency in mastering the CartPole balancing task. This analysis not only confirms the effectiveness of the implemented DQN components but also showcases the potential of reinforcement learning in complex decision-making scenarios.


## Usage
Instructions on how to run the project, train the models, and evaluate them:

```bash
python train_test.py --train
python train_test.py --test
```

## Contributing
We welcome contributions to this project. Please read the contributing guidelines before starting any work.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

### Notes:
- Replace `"Link-to-Colab-notebook"` with the actual link to your Google Colab notebook.
- Update the paths in the badges to match your actual repository structure and URLs.
- Ensure all links within the Table of Contents are correctly anchored to their respective sections for easy navigation.

This structured README closely aligns with the style and detailed descriptions found in your ML repository, offering a cohesive look and thorough documentation across your projects.
