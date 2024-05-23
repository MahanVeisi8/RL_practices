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

This notebook is designed for Google Colab. It requires no additional setup from the user's local environment except access to the internet and a Google account.

### Installation of Required Libraries

```bash
pip install gymnasium
pip install torch
pip install matplotlib
```

### Importing Libraries

```python
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
```

### Setting Up GPU and Reproducibility

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## SARSA Implementation

### Agent Design

The SARSA agent class is designed to interact with the environment and update the policy based on a balance between exploration and exploitation, using an epsilon-greedy strategy.

### Network Architecture

The network consists of a simple feedforward neural network with layers designed to process the state inputs and output action values.

## Training Process

The training process involves running the SARSA agent through multiple episodes, updating the policy based on observed rewards and transitions, and adjusting the epsilon value to manage the exploration-exploitation trade-off.

### Hyperparameters

- Learning Rate: 0.001
- Discount Factor: 0.99
- Epsilon Decay: 0.995

## Testing

The model's performance is evaluated by testing it on new episodes and observing its ability to maintain the pole's balance. This phase helps verify the effectiveness of the learned policies.

## Results

Results include plots of rewards, losses, and the epsilon decay over training episodes, illustrating the learning progression and effectiveness of the SARSA algorithm.

## Conclusions

The project demonstrates the application of the SARSA algorithm to a classic problem in reinforcement learning. The results show how the algorithm performs under different conditions and provide insights into its learning capabilities and stability.
