Great! Based on your input, here's a structured and comprehensive README for the **Dueling Double Deep Q-Network (D3QN) Lunar Lander** project:

---

# **Lunar Lander D3QN 🛸**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XufOyUU4ag68-N-ige3vxl_p6-fpXZ4s?usp=sharing)
[![Python Version](https://img.shields.io/badge/Python-3.6%20|%203.7%20|%203.8-blue)](https://www.python.org/downloads/release/python-380/)
![Status](https://img.shields.io/badge/status-active-green)

Welcome to the **Lunar Lander D3QN** project! This repository contains the implementation of a **Dueling Double Deep Q-Network (D3QN)** for solving the Lunar Lander problem, a classic reinforcement learning task. D3QN combines both **Double Q-Learning** and **Dueling Network Architectures** to enhance performance and stability over standard DQN.

---

## **1 - Introduction**

The **Lunar Lander** environment involves controlling a spaceship to land safely on a designated pad using a main engine and two side engines. The goal is to minimize fuel consumption and avoid crashing while landing safely.

This project applies the **D3QN algorithm**, which combines **Double DQN** and **Dueling Network Architectures**, to improve learning stability and policy performance over standard DQN.

---

## **2 - Algorithm Overview**

### **Double DQN**

The Double DQN algorithm addresses the **overestimation bias** of Q-learning by decoupling the selection and evaluation of actions:
- The **main DQN** selects the next action \( \argmax_a Q(s', a) \).
- The **target network** computes the Q-value of the selected action.

**Double DQN Update Formula**:
![Double DQN Formula](assets/double.png)

This ensures that the action selection and value estimation are handled separately, reducing the tendency to overestimate action values.

### **Dueling DQN**

The **Dueling Network Architecture** decouples the state value and action advantage, improving stability by allowing the agent to estimate the **state value** \( V(s) \) and the **advantage** of each action \( A(s, a) \) independently:
- \( Q(s, a) = V(s) + A(s, a) - \frac{1}{|A|}\sum A(s, a') \)

This architecture enables the network to distinguish between the importance of the state and the action more effectively, improving learning.

**Dueling DQN Architecture**:
![Dueling DQN Architecture](assets/dueling.png)

---

## **3 - D3QN Implementation**

### **Replay Memory**

The `ReplayMemory` class stores the agent’s experiences, which are then sampled in mini-batches for learning. This helps reduce correlation between sequential experiences, improving learning efficiency.

### **Dueling DQN Network**

The `DuelingDQN_network` class implements the dueling architecture, which separates the estimation of the state value and the action advantage:
- **Shared Network**: Extracts features common to both the value and advantage.
- **Value Stream**: Estimates the value of the current state \( V(s) \).
- **Advantage Stream**: Estimates the advantage of each action \( A(s, a) \).

### **D3QN Agent**

The `D3QN_agent` class orchestrates the training process. Key components include:
- **Epsilon-Greedy Action Selection**: Chooses actions based on Q-values with exploration controlled by epsilon.
- **Learning**: Updates the Q-values using the Double DQN method and the dueling architecture.
- **Target Network**: Soft updates ensure that the target network slowly tracks the main network, improving stability.

---

## **4 - Training and Evaluation**

### **Training Process**

The agent interacts with the environment, collects experiences, and updates its Q-network using mini-batches sampled from the replay memory. The key hyperparameters include:
- **Learning rate**: Controls the step size during learning.
- **Discount factor**: Determines the importance of future rewards.
- **Epsilon decay**: Reduces exploration over time as the agent learns.



### **Performance Analysis**

The performance of the agent is evaluated based on:
- **Reward accumulation**: Tracks the total reward obtained by the agent over episodes.
- **Q-value estimation**: Measures the average Q-values across states.
- **Loss reduction**: Observes how the loss decreases as the agent learns.

### **Visualization**

Multiple plots help visualize the training process:
- **Reward Plot**: Shows the reward progression over episodes.
- **Loss Plot**: Tracks how the model loss changes during training.
- **Mean Q-Value Plot**: Visualizes the mean Q-value estimates.
- **Epsilon Decay Plot**: Illustrates how exploration decreases over time.

![plots](assets/plots.png)
---

## **5 - Results**

The D3QN agent demonstrates significant improvement in landing success as training progresses. Here are the agent’s performances across different training epochs:

<table>
  <tr>
    <td>Epoch 10<br><img src="assets/10epoch.gif" alt="Epoch 10 Performance" width="240px"></td>
    <td>Epoch 750<br><img src="assets/750epoch.gif" alt="Epoch 750 Performance" width="240px"></td>
    <td>Epoch 1500<br><img src="assets/1500epoch.gif" alt="Epoch 1500 Performance" width="240px"></td>
  </tr>
</table>

The training results reveal that the D3QN agent progressively learns more effective landing strategies, particularly after fine-tuning hyperparameters such as epsilon decay and learning rate.

---

## **6 - Future Directions**

In future work, we aim to extend this project by exploring:
- **Prioritized Experience Replay**: Implementing a more sophisticated experience replay strategy where experiences with higher TD errors are replayed more often.
- **Noisy Networks for Exploration**: Replacing epsilon-greedy exploration with parameter-based exploration techniques.
- **Comparison with Other Algorithms**: Evaluating the performance of D3QN against other advanced RL algorithms such as **Rainbow DQN** and **Actor-Critic methods**.

---

## **Colab Links**

- [Lunar Lander D3QN Notebook](https://colab.research.google.com/drive/1XufOyUU4ag68-N-ige3vxl_p6-fpXZ4s?usp=sharing)

Feel free to explore the code, run experiments, and adjust hyperparameters to further optimize the agent’s performance!

---

Happy learning and coding! 🚀

---

This README covers the essential aspects of your project, highlights the advantages of D3QN, and provides useful insights into the implementation. The visuals and future directions help engage other developers and researchers interested in your project. Let me know if you'd like to make further adjustments!
