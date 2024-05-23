# CartPole RL Practices 🕹️

![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)
![Status](https://img.shields.io/badge/status-active-green)

Welcome to the CartPole RL Practices repository! This repository is divided into four directories, each focusing on a different Reinforcement Learning (RL) technique applied to the classic Cart Pole problem, a popular test environment in RL.

## Table of Contents
- [1 - DQN (Deep Q-Networks)](#1---dqn-deep-q-networks)
- [2 - Hyperparameter Exploration](#2---hyperparameter-exploration)
- [3 - Boltzmann Exploration](#3---boltzmann-exploration)
- [4 - SARSA (State-Action-Reward-State-Action)](#4---sarsa-state-action-reward-state-action)

## 1 - DQN (Deep Q-Networks)
- **[1 - DQN](1%20-%20DQN/)**
- **Goals:** Implement and evaluate the DQN algorithm. Focus on demonstrating how deep learning can be used to solve reinforcement learning problems efficiently.

![DQN](assets/1-DQN/env.png)

<table>
  <tr>
    <td>Epoch 10<br><img src="assets/1-DQN/10epoch.gif" alt="Epoch 10 Performance" width="240px"></td>
    <td>Epoch 500<br><img src="assets/1-DQN/500epoch.gif" alt="Epoch 500 Performance" width="240px"></td>
    <td>Epoch 1000<br><img src="assets/1-DQN/1000epoch.gif" alt="Epoch 1000 Performance" width="240px"></td>
  </tr>
</table>

## 2 - Hyperparameter Exploration
- **[2 - Hyperparameters](2%20-%20Hyperparameters)**
- **Goals:** Analyze and understand the impact of different hyperparameters on the performance of RL algorithms. Using the DQN setup, explore variations in learning rates, discount factors, and update frequencies to optimize performance.

### Learning Rate Variations
| Learning Rate | Loss Plot                                                    | Reward Plot                                                  |
|---------------|--------------------------------------------------------------|--------------------------------------------------------------|
| **1e-2**      | <img src="assets/2-Hyperparameters/Learning_rate/1e-2/Loss_plot.png" width="220"> | <img src="assets/2-Hyperparameters/Learning_rate/1e-2/reward_plot.png" width="220"> |
| **1e-4**      | <img src="assets/2-Hyperparameters/Learning_rate/1e-4/Loss_plot.png" width="220"> | <img src="assets/2-Hyperparameters/Learning_rate/1e-4/reward_plot.png" width="220"> |
| **1e-6**      | <img src="assets/2-Hyperparameters/Learning_rate/1e-6/Loss_plot.png" width="220"> | <img src="assets/2-Hyperparameters/Learning_rate/1e-6/reward_plot.png" width="220"> |

### Discount Factor Variations
| Discount Factor | Loss Plot                                                          | Reward Plot                                                        |
|-----------------|--------------------------------------------------------------------|--------------------------------------------------------------------|
| **0.997**       | <img src="assets/2-Hyperparameters/Discount_factor/0.997/Loss_plot.png" width="220"> | <img src="assets/2-Hyperparameters/Discount_factor/0.997/reward_plot.png" width="220"> |
| **0.97**        | <img src="assets/2-Hyperparameters/Discount_factor/0.97/Loss_plot.png" width="220">  | <img src="assets/2-Hyperparameters/Discount_factor/0.97/reward_plot.png" width="220">  |
| **0.9**         | <img src="assets/2-Hyperparameters/Discount_factor/0.9/Loss_plot.png" width="220">   | <img src="assets/2-Hyperparameters/Discount_factor/0.9/reward_plot.png" width="220">   |

### Update Frequency Variations
| Update Frequency | Loss Plot                                                             | Reward Plot                                                           |
|------------------|-----------------------------------------------------------------------|-----------------------------------------------------------------------|
| **5**            | <img src="assets/2-Hyperparameters/Update_frequency/5/Loss_plot.png" width="220">       | <img src="assets/2-Hyperparameters/Update_frequency/5/reward_plot.png" width="220">     |
| **50**           | <img src="assets/2-Hyperparameters/Update_frequency/50/Loss_plot.png" width="220">      | <img src="assets/2-Hyperparameters/Update_frequency/50/reward_plot.png" width="220">    |
| **100**          | <img src="assets/2-Hyperparameters/Update_frequency/100/Loss_plot.png" width="220">     | <img src="assets/2-Hyperparameters/Update_frequency/100/reward_plot.png" width="220">   |

## 3 - Boltzmann Exploration
- **[3 - Boltzmann](3%-%Boltzmann)**
- **Goals:** Implement Boltzmann exploration strategy within a DQN setup to compare its effectiveness against the epsilon-greedy approach. Focus on the probabilistic approach to action selection based on Q-values.

![boltzmann_formula](assets/3-Boltzmann/boltzmann_formula.png)

| Set          | Temperature Plot                                                                                                 | Loss Plot                                                                                           | Reward Plot                                                                                         |
|--------------|------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| **Set 1**    | <img src="assets/3-Boltzmann/Hyperparameter_set_1/Temperature_plot.png" width="200">                            | <img src="assets/3-Boltzmann/Hyperparameter_set_1/Loss_plot.png" width="200">                      | <img src="assets/3-Boltzmann/Hyperparameter_set_1/reward_plot.png" width="200">                    |
| **Set 2**    | <img src="assets/3-Boltzmann/Hyperparameter_set_2/Temperature_plot.png" width="200">                            | <img src="assets/3-Boltzmann/Hyperparameter_set_2/Loss_plot.png" width="200">                      | <img src="assets/3-Boltzmann/Hyperparameter_set_2/reward_plot.png" width="200">                    |
| **Set 3**    | <img src="assets/3-Boltzmann/Hyperparameter_set_3/Temperature_plot.png" width="200">                            | <img src="assets/3-Boltzmann/Hyperparameter_set_3/Loss_plot.png" width="200">                      | <img src="assets/3-Boltzmann/Hyperparameter_set_3/reward_plot.png" width="200">                    |
| **Set 4**    | <img src="assets/3-Boltzmann/Hyperparameter_set_4/Temperature_plot.png" width="200">                            | <img src="assets/3-Boltzmann/Hyperparameter_set_4/Loss_plot.png" width="200">                      | <img src="assets/3-Boltzmann/Hyperparameter_set_4/reward_plot.png" width="200">                    |

## 4 - SARSA (State-Action-Reward-State-Action)
- **[4 - SARSA](4%-%SARSA)**
- **Goals:** Apply the SARSA algorithm to the Cart Pole problem to evaluate its performance in a straightforward RL scenario. Focus on how SARSA's on-policy learning compares to other techniques like DQN.

![SARSA](assets/4-SARSA/sarsa2.png)

Each directory contains detailed implementations, experiments, and results that explore various aspects of RL algorithms. Feel free to explore each practice directory for in-depth code, insights, and performance analysis! 😃
