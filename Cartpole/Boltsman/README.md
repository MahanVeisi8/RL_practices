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

**GPU Setup and Reproducibility**
Setup for GPU utilization (if available) and reproducibility settings are detailed to ensure consistent results across runs.

## Boltzmann Exploration 🔍
Detailed explanation of how Boltzmann exploration differs from epsilon-greedy strategies, including:
- Initialization changes: Removal of epsilon parameters and introduction of temperature-related parameters.
- Action selection modifications to utilize the Boltzmann distribution.
- Temperature decay mechanism to adjust exploration intensity over time.

## Hyperparameter Impact 📉
### Experiment Setup
Exploration of how different temperature settings (`temperature_max`, `temperature_min`, `temperature_decay`) influence the learning process and agent performance.

### Results
Visual presentation and analysis of the impacts of various temperature parameters on agent training dynamics and performance, with plots for:
- Temperature variations over episodes.
- Loss metrics and reward outcomes.

| Temperature Setting | Description | Temperature Plot | Loss Plot | Reward Plot |
|---------------------|-------------|------------------|-----------|-------------|
| **High Temperature** | Leads to more explorative behavior, potentially enhancing state space coverage. | ![High Temp](path_to_high_temp_plot.png) | ![High Temp Loss](path_to_high_temp_loss.png) | ![High Temp Reward](path_to_high_temp_reward.png) |
| **Low Temperature** | Results in more exploitative behavior, possibly speeding up convergence but risking local optima. | ![Low Temp](path_to_low_temp_plot.png) | ![Low Temp Loss](path_to_low_temp_loss.png) | ![Low Temp Reward](path_to_low_temp_reward.png) |

## Conclusions 📝
Summary of key findings from the Boltzmann exploration experiments, highlighting optimal settings for different scenarios and the trade-offs between exploration and exploitation based on the temperature parameter.

## How to Run 🏃‍♂️
Instructions on how to clone, set up, and run the project for personal experimentation and learning.
