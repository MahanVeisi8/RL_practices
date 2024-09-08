# Lunar Lander D3QN 🛸🔄

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XufOyUU4ag68-N-ige3vxl_p6-fpXZ4s?usp=sharing)
[![Python Version](https://img.shields.io/badge/Python-3.6%20|%203.7%20|%203.8-blue)](https://www.python.org/downloads/release/python-380/)
![Status](https://img.shields.io/badge/status-active-green)

Welcome to the **Lunar Lander D3QN** project! This project builds on our previous **Deep Q-Network (DQN)** work, improving stability and learning efficiency using the **Dueling Deep Q-Network (D3QN)** algorithm. This method is particularly effective for environments like Lunar Lander, where understanding the state’s value and separating it from action advantages is key to faster and more robust learning.

## Table of Contents
- [1 - Introduction](#1---introduction)
- [2 - D3QN Implementation](#2---d3qn-implementation)
  - [Replay Memory](#replay-memory)
  - [D3QN Network](#d3qn-network)
  - [D3QN Agent](#d3qn-agent)
- [3 - Training and Evaluation](#3---training-and-evaluation)
  - [Training Process](#training-process)
  - [Performance Analysis](#performance-analysis)
  - [Visualization](#visualization)
- [4 - Results](#4---results)
- [5 - Future Directions](#5---future-directions)

## 1 - Introduction
The **Lunar Lander** environment challenges an agent to land a spaceship safely on a designated landing pad using a main engine and two side thrusters. The objective is to minimize fuel usage, avoid crashes, and achieve a soft landing.

This project implements the **Dueling Deep Q-Network (D3QN)** architecture, which improves the agent’s ability to learn by distinguishing between state values and action advantages. This architecture helps the agent identify which states are valuable regardless of the action and learn which actions are most advantageous more efficiently.

![D3QN](assets/D3QN_architecture.png)

## Setup

**Running the Notebook in Google Colab**
- This notebook is designed for easy execution in Google Colab. You only need a Google account and internet access. You can execute the notebook by clicking the Colab badge above. 😊

### Prerequisites
To run this project locally, install the following Python packages to ensure all dependencies are met:

```bash
pip install gymnasium
pip install torch
pip install matplotlib
pip install renderlab
```

## 2 - D3QN Implementation
### Replay Memory
The `ReplayMemory` class stores experiences (state, action, reward, next_state, done) during interactions with the environment. This stored data is used to train the D3QN model by sampling mini-batches, which helps in breaking the temporal correlation in the data.

### D3QN Network
The `DuelingDQN_network` class implements the **dueling architecture**, where the network splits into two streams after a few shared layers:
- **State Value Stream (V(s))**: Estimates the value of the current state.
- **Action Advantage Stream (A(s, a))**: Estimates the advantage of each action in the current state.

The final Q-values are calculated by combining these two streams:
\[ Q(s, a) = V(s) + A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a') \]
This structure improves the model’s ability to generalize across states and actions.

### D3QN Agent
The `D3QN_Agent` class manages the key components of the D3QN algorithm, such as action selection (using epsilon-greedy strategy), learning from the replay memory, and performing soft updates for the target network. This agent aims to optimize both the policy and the value estimates of states more efficiently compared to traditional DQNs.

## 3 - Training and Evaluation
### Training Process
The training process involves:
- Interacting with the environment to collect experiences (state, action, reward, next_state, done).
- Storing the experiences in the replay buffer.
- Sampling mini-batches from the buffer to update the D3QN model.
- Gradually reducing the exploration (epsilon decay) to shift from exploration to exploitation as the agent becomes more confident in its learned policy.

Key hyperparameters:
- **Learning rate**: 2e-4
- **Discount factor**: 0.9965
- **Batch size**: 64
- **Epsilon decay**: 0.995

### Performance Analysis
During training, various metrics are tracked, including:
- **Rewards**: The cumulative reward per episode.
- **Loss**: The difference between the predicted and target Q-values.
- **Mean Q-Value**: The average Q-value for states sampled from the replay memory.
- **Epsilon Decay**: Shows how the exploration rate decreases over time.

### Visualization
Several plots are generated to visualize the training progress:
- **Reward Plot**: Tracks the cumulative rewards over episodes.
- **Loss Plot**: Shows the reduction in loss over time.
- **Mean Q-Value Plot**: Visualizes the estimated Q-values during training.
- **Epsilon Decay Plot**: Displays the rate of exploration as epsilon decays over episodes.

![D3QN](assets/plots.png)

## 4 - Results
The agent’s performance improved significantly as training progressed. Below are snapshots of the agent’s performance during different epochs:

<table>
  <tr>
    <td>Epoch 10<br><img src="assets/10epoch.gif" alt="Epoch 10 Performance" width="240px"></td>
    <td>Epoch 750<br><img src="assets/750epoch.gif" alt="Epoch 750 Performance" width="240px"></td>
    <td>Epoch 1500<br><img src="assets/1500epoch.gif" alt="Epoch 1500 Performance" width="240px"></td>
  </tr>
</table>

As shown, the D3QN agent becomes more proficient at landing as the training progresses, demonstrating the advantage of using the dueling architecture for this type of problem.

## 5 - Future Directions
For future work, we plan to:
- Compare the performance of **D3QN** with other advanced RL algorithms such as **Double DQN** and **Dueling DQN with Prioritized Experience Replay (PER)**.
- Experiment with different values for the discount factor (gamma) and analyze how it affects the agent’s learning.

## Colab Links
- [Lunar Lander D3QN](https://colab.research.google.com/drive/1XufOyUU4ag68-N-ige3vxl_p6-fpXZ4s?usp=sharing)

Feel free to explore the code, modify the hyperparameters, and see how it impacts the agent's performance!

Happy coding and learning! 🚀
