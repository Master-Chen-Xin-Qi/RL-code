# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :final project
# @File     :plot
# @Date     :2022/1/9 9:14
# @Author   :Xinqi Chen
# @Software :PyCharm
-------------------------------------------------
"""
import os
import matplotlib.pyplot as plt


f = open("DQN_reward.txt", 'r')
res = []
for line in f:
    line = line.split(', ')
    if line == ['Save model!\n']:
        continue
    r = line[1].replace('reward: ', '')
    reward = float(r)
    res.append(reward)

plt.plot(res)
plt.title('Reward')
plt.show()

f1 = open('PPO_reward.txt', 'r')
rewards = []
for line in f1:
    line = line.split('\t')
    rewards.append(float(line[2]))
plt.plot(rewards)
plt.title('Reward')
plt.show()

