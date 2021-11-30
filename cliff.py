# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :pythonProject
# @File     :cliff
# @Date     :2021/10/20 14:44
# @Author   :Xinqi Chen
# @Software :PyCharm
-------------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np
import random


class Cliff:
    def __init__(self, alpha, epsilon, gamma):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.x_length = 12
        self.y_length = 4
        self.actionRewards = np.zeros((self.x_length, self.y_length, 4))
        # 初始化reward图，其中↑ → ← ↓分别为0,1,2,3；左下角为原点
        self.actionRewards[:, :, :] = -1.0
        self.actionRewards[1:11, 1, 2] = -100.0
        self.actionRewards[0, 0, 1] = -100.0
        # 移动到哪个位置
        self.actionDestination = []
        for i in range(0, 12):
            self.actionDestination.append([])
            for j in range(0, 4):
                destination = dict()
                destination[0] = [i, min(j + 1, 3)]
                destination[1] = [min(i + 1, 11), j]
                if 0 < i < 11 and j == 1:
                    destination[2] = [0, 0]
                else:
                    destination[2] = [i, max(j - 1, 0)]
                destination[3] = [max(i - 1, 0), j]
                self.actionDestination[-1].append(destination)
        self.actionDestination[0][0][1] = [0, 0]

    # 返回坐标与reward
    def move(self, x, y, act):
        goal = 0
        if x == self.x_length - 1 and y == 0:
            goal = 1
        # 向上
        if act == 0:
            y += 1
        # 向右
        if act == 1:
            x += 1
        # 向下
        if act == 2:
            y -= 1
        # 向左
        if act == 3:
            x -= 1

        x = max(0, x)
        x = min(self.x_length - 1, x)
        y = max(0, y)
        y = min(self.y_length - 1, y)

        if goal == 1:
            return x, y, -1
        if x > 0 and x < self.x_length - 1 and y == 0:
            return 0, 0, -100
        return x, y, -1

    # Sarsa的更新方式
    def epsilon_policy(self, x, y, q):
        eps = self.epsilon
        t = random.randint(0, 3)
        if random.random() < eps:
            a = t
        else:
            q_max = q[x][y][0]
            a_max = 0
            for i in range(4):
                if q[x][y][i] >= q_max:
                    q_max = q[x][y][i]
                    a_max = i
            a = a_max
        return a

    # QLearning的更新方式
    def max_q(self, x, y, q):
        q_max = q[x][y][0]
        a_max = 0
        for i in range(4):
            if q[x][y][i] >= q_max:
                q_max = q[x][y][i]
                a_max = i
        a = a_max
        return a

    def sarsa_on_policy(self, q):
        epochs = 20
        rewards = np.zeros([500])
        for j in range(epochs):
            for i in range(500):
                reward_sum = 0
                x = 0
                y = 0
                act = self.epsilon_policy(x, y, q)
                while True:
                    [x_next, y_next] = self.actionDestination[x][y][act]
                    reward = self.actionRewards[x][y][act]
                    reward_sum += reward
                    act_next = self.epsilon_policy(x_next, y_next, q)
                    q[x][y][act] += self.alpha * (reward + self.gamma * q[x_next][y_next][act_next] - q[x][y][act])
                    if x == self.x_length - 1 and y == 0:
                        break
                    x = x_next
                    y = y_next
                    act = act_next
                rewards[i] += reward_sum
        rewards /= epochs
        avg_rewards = []
        # 十个reward一组，作图更清晰
        for i in range(9):
            avg_rewards.append(np.mean(rewards[:i + 1]))
        for i in range(10, len(rewards) + 1):
            avg_rewards.append(np.mean(rewards[i - 10:i]))
        return q, avg_rewards

    def q_learning_policy(self, q):
        epochs = 20
        rewards = np.zeros([500])
        for j in range(epochs):
            for i in range(500):
                reward_sum = 0
                x = 0
                y = 0
                while True:
                    act = self.epsilon_policy(x, y, q)
                    x_next, y_next, reward = self.move(x, y, act)
                    act_next = self.max_q(x_next, y_next, q)
                    reward_sum += reward
                    q[x][y][act] += self.alpha * (reward + self.gamma * q[x_next][y_next][act_next] - q[x][y][act])
                    if x == self.x_length - 1 and y == 0:
                        break
                    x = x_next
                    y = y_next
                rewards[i] += reward_sum
        rewards /= epochs
        avg_rewards = []
        # 十个reward一组，作图更清晰
        for i in range(9):
            avg_rewards.append(np.mean(rewards[:i + 1]))
        for i in range(10, len(rewards) + 1):
            avg_rewards.append(np.mean(rewards[i - 10:i]))
        return q, avg_rewards

    def OptimalPath(self, q):
        x = 0
        y = 0
        path = np.zeros([self.x_length, self.y_length]) - 1
        end = 0
        exist = np.zeros([self.x_length, self.y_length])
        while (x != self.x_length-1 or y != 0) and end == 0:
            act = self.max_q(x,y,q)
            path[x][y] = act
            if exist[x][y] == 1:
                end = 1
            exist[x][y] = 1
            x,y,r = self.move(x,y,act)
        for j in range(self.y_length-1,-1,-1):
            for i in range(self.x_length):
                if i == self.x_length-1 and j == 0:
                    print("G ",end = "")
                    continue
                a = path[i,j]
                if a == -1:
                    print("0 ",end = "")
                elif a == 0:
                    print("↑ ",end = "")
                elif a == 1:
                    print("→ ",end = "")
                elif a == 2:
                    print("↓ ",end = "")
                elif a == 3:
                    print("← ",end = "")
            print("")

if __name__ == "__main__":
    c = Cliff(alpha=0.5, epsilon=0.1, gamma=1)
    q1 = np.zeros([12, 4, 4])
    q2 = np.zeros([12, 4, 4])
    q_sarsa, sarsa_rewards = c.sarsa_on_policy(q1)
    q_qlearning, q_learning_rewards = c.q_learning_policy(q2)

    plt.plot(range(len(sarsa_rewards)), sarsa_rewards, label="sarsa")
    plt.plot(range(len(sarsa_rewards)), q_learning_rewards, label="q learning")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Sarsa Vs Q-Learning")
    plt.legend()
    plt.show()
    print("Sarsa optimal way:")
    c.OptimalPath(q1)
    print("")
    print("Q-Learning optimal way:")
    c.OptimalPath(q2)

