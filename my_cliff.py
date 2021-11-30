# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :pythonProject
# @File     :my_cliff
# @Date     :2021/10/20 22:48
# @Author   :Xinqi Chen
# @Software :PyCharm
-------------------------------------------------
"""
import random

import numpy as np
import matplotlib.pyplot as plt


class Cliff:
    def __init__(self, alpha, gamma, eps, num_episode):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.num_episode = num_episode
        self.x_length = 12
        self.y_length = 4
        self.q = np.zeros((self.x_length*self.y_length, 4))
        self.actions = ["up", "right", "down", "left"]  # 四个方向分别是0,1,2,3
        self.rewards = np.zeros((self.x_length * self.y_length, 4))
        self.sum_reward = np.zeros([self.num_episode])
        self.epochs = 20
        for i in range(self.y_length * self.x_length):
            for j in range(4):
                if i == 36:
                    self.rewards[i][1] = -100
                elif 25 <= i <= 34:
                    self.rewards[i][2] = -100
                else:
                    self.rewards[i][j] = -1

    def grid2num(self, x, y):
        n = y*self.x_length + x
        return n

    def move(self, x, y, act):
        goal = 0
        if x == self.x_length - 1 and y == self.y_length-1:
            goal = 1
        if act == 0:
            y = y - 1
        elif act == 1:
            x = x + 1
        elif act == 2:
            y = y + 1
        else:
            x = x - 1

        x = max(0, x)
        x = min(self.x_length-1, x)
        y = max(0, y)
        y = min(self.y_length-1, y)

        if goal == 1:
            return x, y, -1
        if 0 < x < self.x_length-1 and y == 3:
            return 0, 3, -100
        return x, y, -1

    def greedy_eps(self, state):
        ran_int = random.randint(0, 3)
        if random.random() < self.eps:
            act = ran_int
        else:
            q_max = self.q[state][0]
            act = 0
            for i in range(1, 4):
                if self.q[state][i] >= q_max:
                    q_max = self.q[state][i]
                    act = i
        return act

    def greedy_max(self, state):
        q_max = self.q[state][0]
        act = 0
        for i in range(1, 4):
            if self.q[state][i] >= q_max:
                q_max = self.q[state][i]
                act = i
        return act

    def sarsa_policy(self):
        for _ in range(self.epochs):
            for i in range(self.num_episode):
                reward_sum = 0
                x = 0
                y = 3
                state = self.grid2num(x, y)
                act = self.greedy_eps(state)
                while True:
                    x_next, y_next, reward = self.move(x, y, act)
                    state = self.grid2num(x, y)
                    reward_sum += reward
                    state_next = self.grid2num(x_next, y_next)
                    act_next = self.greedy_eps(state_next)
                    self.q[state][act] += self.alpha*(reward + self.gamma*self.q[state_next][act_next] - self.q[state][act])
                    if self.grid2num(x, y) == 47:
                        break
                    x = x_next
                    y = y_next
                    act = act_next
                print("Finish Sarsa episode %d" % (i+1))
                self.sum_reward[i] += reward_sum
        self.sum_reward /= self.epochs
        avg_reward = []
        # 十个reward一组，作图更清晰
        for i in range(9):
            avg_reward.append(np.mean(self.sum_reward[:i + 1]))
        for i in range(10, len(self.sum_reward) + 1):
            avg_reward.append(np.mean(self.sum_reward[i - 10:i]))
        return avg_reward

    def qlearning_policy(self):
        for _ in range(self.epochs):
            for i in range(self.num_episode):
                reward_sum = 0
                x = 0
                y = 3
                while True:
                    state = self.grid2num(x, y)
                    act = self.greedy_eps(state)
                    x_next, y_next, reward = self.move(x, y, act)
                    reward_sum += reward
                    state_next = self.grid2num(x_next, y_next)
                    act_next = self.greedy_max(state_next)
                    self.q[state][act] += self.alpha*(reward + self.gamma*self.q[state_next][act_next] - self.q[state][act])
                    if self.grid2num(x, y) == 47:
                        break
                    x = x_next
                    y = y_next
                print("Finish Qlearning episode %d"%(i+1))
                self.sum_reward[i] += reward_sum
        self.sum_reward /= self.epochs
        avg_reward = []
        # 十个reward一组，作图更清晰
        for i in range(9):
            avg_reward.append(np.mean(self.sum_reward[:i + 1]))
        for i in range(10, len(self.sum_reward) + 1):
            avg_reward.append(np.mean(self.sum_reward[i - 10:i]))
        return avg_reward

    def optimal_way(self):
        # print(self.q)
        for i in range(len(self.q)):
            if i == 47:
                print("G ")
                continue
            act = np.argmax(self.q[i])
            if act == 0:
                print("↑ ", end="")
            elif act == 1:
                print("→ ", end="")
            elif act == 2:
                print("↓ ", end="")
            else:
                print("← ", end="")
            if i % 12 == 11:
                print("")



if __name__ == "__main__":
    c1 = Cliff(alpha=0.5, gamma=1, eps=0.1, num_episode=500)
    c2 = Cliff(alpha=0.5, gamma=1, eps=0.1, num_episode=500)
    q_reward = c1.qlearning_policy()
    sarsa_reward = c2.sarsa_policy()
    plt.plot(range(len(sarsa_reward)), sarsa_reward, label="sarsa")
    plt.plot(range(len(q_reward)), q_reward, label="qlearning")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Sarsa Vs. Q-Learning")
    plt.legend()
    plt.show()
    print("Qlearning optimal way:")
    c1.optimal_way()
    print("Sarsa optimal way:")
    c2.optimal_way()