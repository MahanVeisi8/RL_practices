# Boltzmann Exploration in DQN Agents

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19wuWHiw_GGcXLqzsA12ljOp44NmNNZiw?usp=sharing)
[![Python Version](https://img.shields.io/badge/Python-3.6%20|%203.7%20|%203.8-blue)](https://www.python.org/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)]()

## Introduction 🚀
This repository provides an implementation of Boltzmann exploration within Deep Q-Networks (DQN) for the Cart Pole problem. Boltzmann exploration, unlike the epsilon-greedy method, offers a probabilistic approach to action selection. It utilizes the **`temperature`** parameter to adjust the probability of selecting actions based on their Q-values, aiming for a more effective exploration-exploitation balance.

## Table of Contents
- [Setup](#setup)
- [Boltzmann Exploration](#boltzmann-exploration)
- [Hyperparameter Impact](#hyperparameter-impact)
- [Conclusions](#conclusions)

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
This section examines how different temperature settings affect the learning dynamics and agent performance. Key hyperparameters include:
- **`temperature_max`**: Determines initial exploration level.
- **`temperature_min`**: Ensures sustained exploration throughout training.
- **`temperature_decay`**: Modulates the reduction in exploration over time.

### Hyperparameter Settings
Before diving into specific settings, it's essential to understand the influence of temperature on the likelihood of selecting actions. Higher temperatures equate to more random selections, enhancing exploration.

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
        <tr><td>temperature_decay</td><td>0.995</td

></tr>
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

## Conclusions 📝
From our experiments with Boltzmann exploration, the key takeaway is that appropriate temperature settings are crucial for balancing exploration and exploitation. High temperatures can be beneficial in complex environments where a broader exploration is necessary, while lower temperatures might suffice for simpler or well-understood environments. The findings emphasize the importance of tuning the temperature parameter to match the specific learning context and goals.
