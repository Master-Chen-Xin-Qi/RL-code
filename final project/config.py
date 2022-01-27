# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :final project
# @File     :config
# @Date     :2022/1/7 14:36
# @Author   :Xinqi Chen
# @Software :PyCharm
-------------------------------------------------
"""
models_name = ['DQN', 'DDQN', 'PPO']
env_name = ['PongNoFrameskip-v4', 'Hopper-v2']

# DQN parameters
gamma = 0.99
epsilon_max = 1
epsilon_min = 0.05
eps_decay = 30000
frames = 2000000
USE_CUDA = True
learning_rate = 2e-4
max_buff = 100000
update_tar_interval = 1000
batch_size = 32
print_interval = 1000
learning_start = 10000
win_reward = 18     # Pong-v4
win_break = True

# DDQN parameters
DDQN_epsilon_min = 0.02

# PPO parameters
lr_actor = 0.0003
lr_critic = 0.0003
Iter = 15000
MAX_STEP = 10000
gamma_PPO =0.98
lambd = 0.98
batch_size_PPO = 64
epsilon = 0.2
l2_rate = 0.001
beta = 3