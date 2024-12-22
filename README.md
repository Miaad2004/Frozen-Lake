# Frozen Taxi-RL
A collection of reinforcement learning algorithms and other methods for solving MDPs, implemented for the stochastic Frozen Lake environment from Gymnasium.

## Algorithms Implemented
- **Value Iteration**: For solving MDPs
- **Policy Iteration**: For solving MDPs
- **Monte Carlo**: Monte Carlo learning method for policy optimization
- **Q-Learning**: Double Q-learning implementation with epsilon-greedy exploration

## Requirements
- Tested on Python 3.13
- These libraries are also required:
```txt
colorama==0.4.6
gymnasium==1.0.0 
numpy==2.2.0
pygame==2.6.1
tqdm==4.67.1
```

## Usage
Just run main.py:
```txt
python main.py
```

## Features:
- Stochastic 8x8 Frozen Lake environment with:
  - Custom reward function based on Manhattan distance
  - Random hole generation in hard mod
  - Action uncertainty in hard mode

> Note: This code is part of a college project, and the source.py file was provided by the course instructor. source.py is not my original work.
