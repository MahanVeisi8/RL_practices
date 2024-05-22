# Boltzmann Exploration in DQN Agents

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19wuWHiw_GGcXLqzsA12ljOp44NmNNZiw?usp=sharing)
[![Python Version](https://img.shields.io/badge/Python-3.6%20|%203.7%20|%203.8-blue)](https://www.python.org/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)]()

## Introduction 🚀
This repository demonstrates the implementation of Boltzmann exploration (also known as softmax action selection) in Deep Q-Networks (DQN) for solving the Cart Pole problem. Unlike traditional epsilon-greedy strategies, Boltzmann exploration provides a probabilistic approach to action selection that may improve the exploration-exploitation balance by using a temperature parameter to modulate action choice probabilities based on their Q-values.

## Table of Contents
- [Setup](#setup)
- [Boltzmann Exploration](#boltzmann-exploration)
- [Hyperparameter Impact](#hyperparameter-impact)
  - [Experiment Setup](#experiment-setup)
  - [Results](#results)
- [Conclusions](#conclusions)

## Setup 🛠️
**Running the Notebook in Google Colab**
- The notebook is designed for easy execution in Google Colab, requiring no additional setup other than a Google account and internet access.

**Installation of Required Libraries**
```bash
pip install gymnasium torch matplotlib renderlab
```

**Importing Libraries**
Ensure all necessary libraries, including PyTorch, Gymnasium, and others, are imported for the project.

## Boltzmann Exploration 🔍
Detailed explanation of how Boltzmann exploration differs from epsilon-greedy strategies that we used in the previous practices, including:
- Initialization changes: Removal of epsilon parameters and introduction of temperature-related parameters.
- Action selection modifications to utilize the Boltzmann distribution.
  ```py
      def select_action(self, state):
        """
        Select an action using Boltzmann exploration.
        """
        action_probs = self.get_action_probs(state)
        action = np.random.choice(len(action_probs), p=action_probs)
        return action

    def get_action_probs(self, state):
        """
        Get action probabilities using Boltzmann distribution.
        """
        with torch.no_grad():
            Q_values = self.main_network(state)
            action_values = Q_values / self.temperature
            action_probs = torch.softmax(action_values, dim=-1).cpu().numpy()
        return action_probs
  ```
- Temperature decay mechanism to adjust exploration intensity over time.
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



### Results
Visual presentation and analysis of the impacts of various temperature parameters on agent training dynamics and performance.

| Hyperparameter Set | Description | Loss Plot | Reward Plot |
|---------------------|-------------|-----------|-------------|
| **Set 1** <br> `temperature_max`: 1000 <br> `temperature_min`: 0.0001 <br> `temperature_decay`: 0.995 | High initial temperature leads to extensive exploration early in training. As the temperature decays, exploration decreases, shifting towards exploitation. | ![Set 1 Loss](path_to_set1_loss_plot.png) | ![Set 1 Reward](path_to_set1_reward_plot.png) |
| **Set 2** <br> `temperature_max`: 10 <br> `temperature_min`: 0.0001 <br> `temperature_decay`: 0.995 | Moderate initial temperature balances exploration and exploitation throughout training. Decay rate ensures a gradual shift towards exploitation. | ![Set 2 Loss](path_to_set2_loss_plot.png) | ![Set 2 Reward](path_to_set2_reward_plot.png) |
| **Set 3** <br> `temperature_max`: 0.1 <br> `temperature_min`: 0.0001 <br> `temperature_decay`: 0.995 | Low initial temperature promotes early exploitation with limited exploration. Temperature decay has minimal impact due to already low initial value. | ![Set 3 Loss](path_to_set3_loss_plot.png) | ![Set 3 Reward](path_to_set3_reward_plot.png) |
| **Set 4** <br> `temperature_max`: 1 <br> `temperature_min`: 0.0001 <br> `temperature_decay`: 1 (constant) | Constant temperature maintains a consistent balance between exploration and exploitation throughout training. | ![Set 4 Loss](path_to_set4_loss_plot.png) | ![Set 4 Reward](path_to_set4_reward_plot.png) |


## Conclusions 📝
Summary of key findings from the Boltzmann exploration experiments, highlighting optimal settings for different scenarios and the trade-offs between exploration and exploitation based on the temperature parameter.

## How to Run 🏃‍♂️
Instructions on how to clone, set up, and run the project for personal experimentation and learning.
