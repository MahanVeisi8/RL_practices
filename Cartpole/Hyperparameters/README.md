# DQN Hyperparameter Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JLVm4wdrtjpMiQP_3SmdOhWjX1hhFLP8?usp=sharing)
[![Python Version](https://img.shields.io/badge/Python-3.6%20|%203.7%20|%203.8-blue)](https://www.python.org/downloads/release/python-380/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)](https://github.com/MahanVeisi8/RL_practices/Cartpole/DQN/requirements.txt)

## Introduction
This repository explores the impact of various hyperparameters on the performance of the Deep Q-Network (DQN) applied to the Cart Pole problem. Through systematic experiments, we analyze how learning rate, discount factor, and update frequency affect the model's learning efficiency and policy stability.

## Setup
Details on setting up the environment, installing required libraries, and running the notebook in Google Colab.

### Prerequisites
To run this project locally, ensure you have the following Python packages installed:
```bash
pip install gymnasium torch matplotlib renderlab
```

## Hyperparameter Impact Analysis
Overview of the methodology for testing different hyperparameter settings and the rationale behind selecting specific values.

### Learning Rate Variations
Exploration of how different learning rates affect the speed and stability of learning, with visual comparisons and statistical analysis.

### Discount Factor Adjustments
Investigation into the influence of discount factor variations on the long-term reward strategy of the DQN agent.

### Update Frequency Modifications
Analysis of how different frequencies for updating the target network impact the convergence and performance of the agent.

## Results and Discussion
Detailed presentation of the training and testing results with comprehensive visual data illustrating trends and key findings.

## Conclusions
Summary of the insights gained from the hyperparameter analysis, including recommendations for choosing optimal settings for similar reinforcement learning tasks.


