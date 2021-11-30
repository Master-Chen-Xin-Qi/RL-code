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
from typing import Tuple, Union


class ER:
    def __init__(self, max_size: int, batch_size: int):
        self.max_size = max_size
        self.batch_size = batch_size
        self.buffer = np.array([(None, None, None, None, None) for _ in range(max_size)])
        self.cur_ind = 0
        self.cur_size = 0

    def append(self, e: Tuple[np.ndarray, Union[np.ndarray, int], float, np.ndarray, bool]):
        self.buffer[self.cur_ind] = e
        self.cur_ind = (self.cur_ind + 1) % self.max_size
        self.cur_size = max(self.cur_size, self.cur_ind)

    def sample(self):
        indices = random.sample(range(0, self.cur_size), self.batch_size)
        s, a, r, s_next, done = zip(*self.buffer[indices])
        return np.stack(s), np.stack(a), np.stack(r), np.stack(s_next), np.stack(done)


class my_Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(my_Net, self).__init__()
        self.layer1 = nn.Linear(in_dim, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, out_dim)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        # 初始化参数
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
    def __init__(self,
                 n_state: int,
                 n_action: int,
                 eps_start: float = 0.95,
                 eps_final: float = 0.001,
                 eps_decay: int = 2000,
                 gamma: float = 0.99,
                 er_batch_size: int = 128,
                 er_max_size: int = 10000,
                 target_freq: int = 2000,
                 play_before_learn: int = 128):

        self.net = my_Net(n_state, n_action)
        self.target_net = my_Net(n_state, n_action)
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.replay_buffer = ER(er_max_size, er_batch_size)
        self.play_before_learn = play_before_learn
        self.action_space = range(n_action)
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_decay = eps_decay
        self.target_freq = target_freq
        self.frame_ind = 0
        self.loss = nn.MSELoss()

    def get_eps(self):
        return self.eps_final + (self.eps_start - self.eps_final) * np.exp(-1. * self.frame_ind / self.eps_decay)

    def act(self, state):
        if self.replay_buffer.cur_size > self.play_before_learn:
            if np.random.rand() <= self.get_eps():
                action = np.random.choice(self.action_space)
            else:
                state = torch.from_numpy(state).float()
                with torch.no_grad():
                    values = self.net(state).cpu().numpy()
                action = np.argmax(values)
            self.frame_ind += 1
        else:
            # 经验池未满则随机选取
            action = np.random.choice(self.action_space)
        return action

    def append(self, e: Tuple[np.ndarray, Union[np.ndarray, int], float, np.ndarray, bool]):
        self.replay_buffer.append(e)

    def train(self):
        if self.replay_buffer.cur_size > self.play_before_learn:
            if self.frame_ind % self.target_freq == 0:
                self.target_net.load_state_dict(self.net.state_dict())

            s, a, r, s_next, done = self.replay_buffer.sample()
            s = torch.from_numpy(s).float()
            a = torch.from_numpy(a).long()
            r = torch.from_numpy(r).float()
            s_next = torch.from_numpy(s_next).float()
            done = torch.from_numpy(done).float()
            with torch.no_grad():
                s_next_pred = self.target_net(s_next).max(1)[0]
            y = r + self.gamma * s_next_pred * (1 - done)
            pred = self.net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = self.loss(y, pred)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    agent = DQN(env.observation_space.shape[0], env.action_space.n)

    for episode in range(100):
        tot_reward = 0
        state = env.reset()
        while True:
            action = agent.act(state)
            new_state, reward, done, _ = env.step(action)

            # use reward shaping
            modified_reward = reward + 300 * (0.99 * abs(new_state[1]) - abs(state[1]))
            agent.append((state, action, modified_reward, new_state, done))
            agent.train()

            state = new_state
            tot_reward += reward

            env.render()
            if done and state[0] >= 0.5:
                print("第%i次成功到达终点" % (episode+1))
                break
            if done:
                print("第%d次超过200个epoch，重置小车" % (episode+1))
                break
    env.close()
