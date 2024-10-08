# Boltzmann Exploration in DQN Agents

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19wuWHiw_GGcXLqzsA12ljOp44NmNNZiw?usp=sharing)
[![Python Version](https://img.shields.io/badge/Python-3.6%20|%203.7%20|%203.8-blue)](https://www.python.org/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)](https://github.com/MahanVeisi8/RL_practices/Cartpole/3%20-%20Boltzmann/requirements.txt)

## Introduction 🚀
This repository provides an implementation of Boltzmann exploration within Deep Q-Networks (DQN) for the Cart Pole problem. Boltzmann exploration, unlike the epsilon-greedy method, offers a probabilistic approach to action selection. It utilizes the **`temperature`** parameter to adjust the probability of selecting actions based on their Q-values, aiming for a more effective exploration-exploitation balance.

## Table of Contents
- [Setup](#setup-️)
- [Boltzmann Exploration](#boltzmann-exploration-)
- [Hyperparameter Impact](#hyperparameter-impact-)
  - [Experiment Setup](#experiment-setup)
  - [Hyperparameter Settings](#hyperparameter-settings)
  - [Comparative Analysis of Results](#comparative-analysis-of-results)
- [Conclusions](#conclusions-📝)

## Setup 🛠️
**Running the Notebook in Google Colab**
- Simply open the notebook in Google Colab, which requires only a Google account and internet access.😊

**Installation of Required Libraries**
```bash
pip install gymnasium torch matplotlib renderlab
```

**Importing Libraries**
Make sure to import necessary libraries such as PyTorch, Gymnasium, etc.

## Boltzmann Exploration 🔍
![boltzmann_formula](assets/boltzmann_formula.png)
Boltzmann exploration strategically balances exploration and exploitation by modifying action selection probabilities according to their Q-values, contrasting sharply with the fixed exploration rate of epsilon-greedy strategies. Key adjustments include:
- Elimination of epsilon in favor of temperature parameters.
- Action selection now follows the Boltzmann distribution.
  ```py
  def get_action_probs(self, state):
    """
    Get action probabilities using Boltzmann distribution.
    """
    with torch.no_grad():
        Q_values = self.main_network(state)
        action_values = Q_values / self.temperature
        action_probs = torch.softmax(action_values, dim=-1).cpu().numpy()
    return action_probs

  def select_action(self, state):
    """
    Select an action using Boltzmann exploration.
    """
    action_probs = self.get_action_probs(state)
    action = np.random.choice(len(action_probs), p=action_probs)
    return action
  ```
- Temperature is dynamically adjusted to decrease exploration as learning progresses.
  ```py
  def update_temperature(self):
    self.temperature = max(self.temperature * self.temperature_decay, self.temperature_min)
  ```
## Hyperparameter Impact 📉
### Experiment Setup
Exploration of how different temperature settings influence the learning process and agent performance. The key parameters are:

- **`temperature_max`**: The initial temperature value at the start of training, which determines the level of exploration. A higher `temperature_max` encourages more exploratory actions.
- **`temperature_min`**: The minimum temperature value that the system will decay to, ensuring that exploration doesn't cease entirely. This parameter helps maintain a baseline level of exploration throughout training.
- **`temperature_decay`**: The rate at which the temperature decreases over time. This decay rate controls how quickly the exploration level transitions to more exploitative behavior as training progresses.

These parameters collectively influence the agent's ability to balance exploration of new actions and exploitation of known rewarding actions.

### Hyperparameter Settings

Before setting the hyperparameters, let's look at impacts of different temperatures and chances of choosing different values.

![boltzmann_prob_choise_for_different_t_values](assets/boltzmann_prob_choise_for_different_t_values.png)

<table style="border-spacing: 80px; width: 100%;">
  <tr>
    <td style="vertical-align: top;">
      <strong>Set 1: High Exploration</strong>
      <table style="width: 100%; border: 15px solid black; padding: 50px;">
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>temperature_max</td><td>1000</td></tr>
        <tr><td>temperature_min</td><td>0.0001</td></tr>
        <tr><td>temperature_decay</td><td>0.995</td></tr>
      </table>
    </td>
    <td style="vertical-align: top;">
      <strong>Set 2: Moderate Exploration</strong>
      <table style="width: 100%; border: 15px solid black; padding: 50px;">
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>temperature_max</td><td>10</td></tr>
        <tr><td>temperature_min</td><td>0.0001</td></tr>
        <tr><td>temperature_decay</td><td>0.995</td></tr>
      </table>
    </td>
  </tr>
  <tr>
    <td style="vertical-align: top;">
      <strong>Set 3: Low Exploration</strong>
      <table style="width: 100%; border: 15px solid black; padding: 50px;">
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>temperature_max</td><td>0.1</td></tr>
        <tr><td>temperature_min</td><td>0.0001</td></tr>
        <tr><td>temperature_decay</td><td>0.995</td></tr>
      </table>
    </td>
    <td style="vertical-align: top;">
      <strong>Set 4: Constant Temperature</strong>
      <table style="width: 100%; border: 15px solid black; padding: 50px;">
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>temperature_max</td><td>1</td></tr>
        <tr><td>temperature_min</td><td>0.0001</td></tr>
        <tr><td>temperature_decay</td><td>1</td></tr>
      </table>
    </td>
  </tr>
</table>




### Comparative Analysis of Results

The table below presents a detailed comparison of how different temperature settings influence the agent's learning dynamics and performance. Each row corresponds to a hyperparameter set as described above, showcasing the outcomes through visual plots and a brief description of the training behavior and results.

| Set | Description | Temperature Plot | Loss Plot | Reward Plot |
|-----|-------------|------------------|-----------|-------------|
| **Set 1** <br> High Initial Temperature | Starts at 1000, promoting extensive early exploration. As exploration decreases, noticeable stabilization and gradual improvements in performance occur, suggesting effective balancing of exploration with exploitation. | ![Temperature 1](assets/Hyperparameter_set_1/Temperature_plot.png) | ![Loss 1](assets/Hyperparameter_set_1/Loss_plot.png) | ![Reward 1](assets/Hyperparameter_set_1/reward_plot.png) |
| **Set 2** <br> Moderate Initial Temperature | Starts at 10, fostering a balance from the beginning. Results in steady progress with less volatility in performance metrics compared to higher initial temperatures, indicating more consistent learning. | ![Temperature 2](assets/Hyperparameter_set_2/Temperature_plot.png) | ![Loss 2](assets/Hyperparameter_set_2/Loss_plot.png) | ![Reward 2](assets/Hyperparameter_set_2/reward_plot.png) |
| **Set 3** <br> Low Initial Temperature | With an initial temperature of 0.1, the model is highly exploitative, limiting exploration which may hinder learning in complex scenarios and could not solve the problem but accelerates performance in simpler tasks or familiar environments. | ![Temperature 3](assets/Hyperparameter_set_3/Temperature_plot.png) | ![Loss 3](assets/Hyperparameter_set_3/Loss_plot.png) | ![Reward 3](assets/Hyperparameter_set_3/reward_plot.png) |
| **Set 4** <br> Constant Temperature | Maintains a steady temperature of 1, ensuring consistent exploration. But did not resulted well in our problem. | ![Temperature 4](assets/Hyperparameter_set_4/Temperature_plot.png) | ![Loss 4](assets/Hyperparameter_set_4/Loss_plot.png) | ![Reward 4](assets/Hyperparameter_set_4/reward_plot.png) |


## Conclusions 📝

The experiments demonstrate that temperature settings significantly impact the agent's learning dynamics. High initial temperatures facilitate extensive exploration but require careful decay to balance exploitation. Moderate initial temperatures provide a balanced approach, leading to steady and consistent learning. Low initial temperatures promote rapid exploitation but may hinder performance in complex environments. Constant temperatures ensure consistent exploration but may not yield optimal results in all scenarios. Optimal settings depend on the specific task and desired balance between exploration and exploitation.
