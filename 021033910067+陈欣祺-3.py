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
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, 3)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        # xavier init
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)
        torch.nn.init.xavier_uniform_(self.layer3.weight)
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
        self.store_size = 1000
        self.learn_time = 0
        self.update_time = 2000  # 目标值网络更新步长
        self.gamma = 0.99
        self.batch_size = 128
        self.max_epoch = 300  # 最多进行300步
        self.eps_start = 0.95
        self.eps_final = 0.001
        self.eps_decay = 1000
        self.store = np.zeros((self.store_size, 7))   # [s,act,s_next,reward,done]，其中s占两个，act占一个，reward占一个，done占一个

    def get_eps(self):
        return self.eps_final + (self.eps_start - self.eps_final) * np.exp(-1. * self.learn_time / self.eps_decay)

    def train(self, net_1, net_2):
        for i in range(10000):
            s = self.env.reset()
            epoch = 0
            while True:
                epoch += 1
                if self.store_count > self.batch_size:
                    if np.random.rand() <= self.get_eps():
                        act = random.randint(0, 2)
                    else:
                        out = net_1(torch.Tensor(s)).detach()
                        act = torch.argmax(out).data.item()  # 最大index即为采取的动作
                else:
                    act = random.randint(0, 2)
                s_next, reward, done, info = env.step(act)  # 采取动作act
                reward = reward + 300 * (0.99 * abs(s_next[1]) - abs(s[1]))

                # 更新
                self.store[self.store_count % self.store_size][0:2] = s
                self.store[self.store_count % self.store_size][2:3] = act
                self.store[self.store_count % self.store_size][3:5] = s_next
                self.store[self.store_count % self.store_size][5:6] = reward
                self.store[self.store_count % self.store_size][6:7] = done
                self.store_count += 1
                s = s_next
                if self.store_count > self.batch_size:
                    if self.learn_time % self.update_time == 0:
                        net_2.load_state_dict(net_1.state_dict())  # 延迟更新

                    # 随机采样
                    batch_s = torch.Tensor(random.sample(list(self.store[:self.store_size, 0:2]), self.batch_size)).float()
                    batch_act = torch.Tensor(random.sample(list(self.store[:self.store_size, 2:3]), self.batch_size)).long()
                    batch_s_next = torch.Tensor(random.sample(list(self.store[:self.store_size, 3:5]), self.batch_size)).float()
                    batch_reward = torch.Tensor(random.sample(list(self.store[:self.store_size, 5:6]), self.batch_size)).float()
                    batch_done = torch.Tensor(random.sample(list(self.store[:self.store_size, 6:7]), self.batch_size)).float()

                    q = net_1(batch_s).gather(1, batch_act)
                    with torch.no_grad():
                        q_next = net_2(batch_s_next).detach().max(1)[0].unsqueeze(1)
                    target = batch_reward
                    target = target + q_next * (1-batch_done) * self.gamma
                    loss = net_1.loss(q, target)
                    net_1.opt.zero_grad()
                    loss.backward()
                    net_1.opt.step()
                    self.learn_time += 1
                    # if epoch >= self.max_epoch:
                    #     print(r"第%d次超过%d个epoch，重置小车位置" % (i+1, self.max_epoch))
                    #     break
                if done and s[0] >= 0.5:
                    print(r"第%d次成功到达终点，共用了%d个epoch" % (i+1, epoch))
                    break
                if done:
                    print("第%d次超过200个epoch，重置小车" % (i + 1))
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
    # env = env.unwrapped
    method = DQN(env)
    method.train(net1, net2)
