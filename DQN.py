# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :pythonProject
# @File     :DQN
# @Date     :2021/10/27 14:24
# @Author   :Xinqi Chen
# @Software :PyCharm
-------------------------------------------------
"""

import gym
import torch
import torch.nn as nn
import numpy as np
import random


class my_Net(nn.Module):
    def __init__(self):
        super(my_Net, self).__init__()
        self.layer1 = nn.Linear(2, 16)
        self.layer2 = nn.Linear(16, 32)
        self.layer3 = nn.Linear(32, 3)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.loss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        out = self.layer3(x)
        return out


class DQN:
    def __init__(self, env):
        self.env = env
        self.store_count = 0
        self.store_size = 2000
        self.epsilon = 0.6
        self.learn_time = 0
        self.update_time = 20  # 目标值网络更新步长
        self.gamma = 0.9
        self.batch_size = 1000
        self.store = np.zeros((self.store_size, 6))   # [s,act,s_next,reward]，其中s占两个，act占一个，reward占一个

    def train(self, net_1, net_2):
        for i in range(10000):
            s = self.env.reset()
            while True:
                eps = random.randint(0, 100)
                if eps < 100 * (self.epsilon ** self.learn_time):
                    act = random.randint(0, 2)
                else:
                    out = net_1(torch.Tensor(s)).detach()
                    act = torch.argmax(out).data.item()  # 最大index即为采取的动作
                s_next, reward, done, info = env.step(act)  # 采取动作act
                reward = s_next[0] + 0.5
                if s_next[0] > -0.5:
                    reward = s_next[0] + 0.5
                    if s_next[0] > 0.5:
                        reward = 10
                else:
                    reward = 0
                # 更新
                self.store[self.store_count % self.store_size][0:2] = s
                self.store[self.store_count % self.store_size][2:3] = act
                self.store[self.store_count % self.store_size][3:5] = s_next
                self.store[self.store_count % self.store_size][5:6] = reward
                self.store_count += 1
                s = s_next
                if self.store_count > self.store_size:
                    if self.learn_time % self.update_time == 0:
                        net_2.load_state_dict(net_1.state_dict())  # 延迟更新
                    index = random.randint(0, self.store_size - self.batch_size - 1)
                    batch_s = torch.Tensor(self.store[index:index + self.batch_size, 0:2])
                    batch_act = torch.Tensor(self.store[index:index + self.batch_size, 2:3]).long()
                    batch_s_next = torch.Tensor(self.store[index:index + self.batch_size, 3:5])
                    batch_reward = torch.Tensor(self.store[index:index + self.batch_size, 5:6])

                    q = net_1(batch_s).gather(1, batch_act)
                    q_next = net_2(batch_s_next).detach().max(1)[0].reshape(self.batch_size, 1)
                    target = batch_reward + self.gamma * q_next
                    loss = net_1.loss(q, target)
                    net_1.opt.zero_grad()
                    loss.backward()
                    net_1.opt.step()
                    self.learn_time += 1

                if done:
                    print(r"第%d次成功到达终点，结束" % i)
                    break
                self.env.render()


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    print(env.action_space)  # 每个时刻三种状态：向左加速、向右加速、不加速
    print(env.observation_space.high)  # 右端终点与最大向右速度
    print(env.observation_space.low)  # 左端终点与最大向左速度
    print('位置范围 = {}'.format((env.unwrapped.min_position, env.unwrapped.max_position)))
    print('速度范围 = {}'.format((-env.unwrapped.max_speed, env.unwrapped.max_speed)))
    print('目标位置 = {}'.format(env.unwrapped.goal_position))  # 当小车的最终位置大于0.5即到达终点，任务结束

    net1 = my_Net()
    net2 = my_Net()  # Target Network方法
    env = env.unwrapped
    method = DQN(env)
    method.train(net1, net2)
