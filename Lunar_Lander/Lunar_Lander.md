# **Lunar Lander RL Practices 🚀**

[![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org/downloads/release/python-380/)
[![Status](https://img.shields.io/badge/status-active-green)]()

Welcome to the **Lunar Lander RL Practices** repository! This repository showcases various implementations of reinforcement learning (RL) algorithms applied to the Lunar Lander environment, a classic RL problem. Here, we explore different methods ranging from **DQN** to **D3QN** with advanced techniques like **adaptive gamma**. The repository is structured into multiple directories, each representing a unique approach.

---

## **Table of Contents**
- [1 - DQN (Deep Q-Networks)](#1---dqn-deep-q-networks)
- [2 - D3QN (Dueling Double DQN)](#2---d3qn-dueling-double-dqn)
- [3 - Adaptive Gamma D3QN](#3---adaptive-gamma-d3qn)
- [Results and Visualizations](#results-and-visualizations)
- [Future Work](#future-work)

---

## **1 - DQN (Deep Q-Networks)**
- **[DQN Directory](DQN/)**

The **Deep Q-Network (DQN)** algorithm is a foundational approach that combines Q-learning with neural networks. This method is widely used to solve problems where the state and action spaces are large and continuous. The DQN implementation here tackles the **Lunar Lander** problem, where an agent learns to control a spaceship and land it safely on a designated pad using reinforcement learning techniques.

### Highlights:
- **Q-Learning with Function Approximation**: Uses neural networks to approximate the Q-function.
- **Replay Memory**: Stores past experiences and trains the agent using mini-batches.
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation for policy learning.

**Performance Visualization**:
![DQN Performance](DQN/assets/plots.png)

---

## **2 - D3QN (Dueling Double DQN)**
- **[D3QN Directory](D3QN/)**

The **Dueling Double Deep Q-Network (D3QN)** extends DQN by addressing **overestimation bias** using a target network and separating **state-value** and **action-advantage** estimations. This combination improves the stability and efficiency of learning policies, making it more robust compared to vanilla DQN.

### Highlights:
- **Double Q-Learning**: Reduces overestimation of Q-values by decoupling action selection from action evaluation.
- **Dueling Architecture**: Estimates the value of being in a state (state-value) and the benefit of each action (action-advantage) separately.
- **Replay Memory and Target Network**: Ensures more stable and efficient learning by reducing correlation in training data.

**Performance Visualization**:
![D3QN Performance](D3QN/asset/plots.png)

---

## **3 - Adaptive Gamma D3QN**
- **[Adaptive Gamma Directory](adaptive_gamma/)**

This project builds upon the **D3QN** algorithm, introducing a **dynamic gamma adjustment** strategy inspired by the paper ["How to Discount Deep Reinforcement Learning"](https://arxiv.org/pdf/1512.02011) by François-Lavet et al. The goal is to enhance convergence speed and improve learning stability by gradually increasing the discount factor (gamma) during training.

### Highlights:
- **Dynamic Gamma**: Adjusts the discount factor over time, starting with short-term rewards and gradually shifting focus to long-term rewards.
- **Incremental Gamma Strategy**: 
  \[
  \gamma_{k+1} = 1 - 0.98 \cdot (1 - \gamma_k)
  \]
  This allows for more stable learning and faster convergence.
- **D3QN Architecture**: Combines Double DQN with a dueling network to separate state-value and action-advantage estimations, further improving performance.

**Performance Visualization**:
![Adaptive Gamma Performance](adaptive_gamma/assets/plots.png)

---

## **Results and Visualizations**

### DQN Results:
<table>
  <tr>
    <td>Epoch 10<br><img src="DQN/assets/10epoch.gif" alt="Epoch 10 Performance" width="240px"></td>
    <td>Epoch 1000<br><img src="DQN/assets/1000epoch.gif" alt="Epoch 1000 Performance" width="240px"></td>
    <td>Epoch 1637<br><img src="DQN/assets/1650epoch.gif" alt="Epoch 1637 Performance" width="240px"></td>
  </tr>
</table>

### D3QN Results:
<table>
  <tr>
    <td>Epoch 10<br><img src="D3QN/asset/10epoch.gif" alt="Epoch 10 Performance" width="240px"></td>
    <td>Epoch 750<br><img src="D3QN/asset/750epoch.gif" alt="Epoch 750 Performance" width="240px"></td>
    <td>Epoch 1500<br><img src="D3QN/asset/1500epoch.gif" alt="Epoch 1500 Performance" width="240px"></td>
  </tr>
</table>

### Adaptive Gamma D3QN Results:
<table>
  <tr>
    <td>Epoch 10<br><img src="adaptive_gamma/assets/10epoch.gif" alt="Epoch 10 Performance" width="240px"></td>
    <td>Epoch 500<br><img src="adaptive_gamma/assets/500epoch.gif" alt="Epoch 500 Performance" width="240px"></td>
    <td>Epoch 1000<br><img src="adaptive_gamma/assets/1000epoch.gif" alt="Epoch 1000 Performance" width="240px"></td>
  </tr>
</table>

---

## **Future Work**

We plan to extend this repository by implementing advanced reinforcement learning methods and techniques, including:
- **Prioritized Experience Replay**: Prioritizing more important transitions for replay, improving learning efficiency.
- **Noisy Networks for Exploration**: Integrating noise into network parameters to enhance exploration without relying on epsilon-greedy methods.
- **Comparison with Actor-Critic Methods**: Analyzing the performance of D3QN and Adaptive Gamma D3QN against popular actor-critic algorithms.
- **Exploring Rainbow DQN**: Incorporating multiple improvements such as Noisy Nets, Prioritized Replay, and Multi-step Learning in a single framework.

---

Feel free to explore the code, experiment with the parameters, and contribute to this repository!

Happy coding and learning! 🚀

---

This README provides a high-level overview of the **Lunar Lander RL Practices** repository, explaining the structure and details of each directory (DQN, D3QN, and Adaptive Gamma). Let me know if you want further refinements!
