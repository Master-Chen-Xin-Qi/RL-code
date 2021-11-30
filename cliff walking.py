# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :pythonProject
# @File     :cliff walking
# @Date     :2021/10/20 14:18
# @Author   :Xinqi Chen
# @Software :PyCharm
-------------------------------------------------
"""
from collections import defaultdict, deque
import random
import numpy as np
import gym
import sys
import matplotlib.pyplot as plt

env = gym.make('CliffWalking-v0')
class cliff:
    def __init__(self, alpha, gamma, episodes):
        self.state = [i for i in range(48)]
        self.terminal_state = [j for j in range(37,47)]
        self.direction = ["up", "down", "left", "right"]
        self.gold_state = 47
        self.alpha = alpha
        self.gamma = gamma
        self.episodes = episodes
        self.eps = 0.1
        self.Q = np.array((4,12))

    # 利用下一个action进行更新，sarsa更新方式
    def update_sarsa(self, alpha, gamma, Q, state, action, reward, next_state=None, next_action=None):
        current = Q[state][action]
        if next_state is not None:
            Qsa_next = Q[next_state][next_action]
        else: Qsa_next = 0

        target = reward + gamma * Qsa_next
        new_value = current + (alpha * (target - current))

        return new_value

    # 采用最大值进行更新，Q-learning更新方式
    def update_Q_learning(self, alpha, gamma, Q, state, action, reward, next_state=None):
        current = Q[state][action]

        Qsa_next = np.max(Q[next_state]) if next_state is not None else 0
        target = reward + gamma * Qsa_next

        new_value = current + alpha * (target - current)

        return new_value


    def epsilon_greedy(self, Q, state, nA, eps):
        if random.random() > eps:
            return np.argmax(Q[state])  # select greedy action with probability epsilon
        else:
            return random.choice(np.arange(env.action_space.n))


    def sarsa(self, plot_every=10):
        epochs = 1
        alpha = self.alpha
        gamma = self.gamma
        num_episodes = self.episodes
        nA = env.action_space.n
        Q = defaultdict(lambda: np.zeros(nA))

        tmp_scores = deque(maxlen=plot_every)
        avg_scores = deque(maxlen=num_episodes)
        scores = []
        for _ in range(epochs):
            for i in range(num_episodes):
                score = 0
                state = env.reset()

                action = self.epsilon_greedy(Q, state, nA, self.eps)

                while True:
                    next_state, reward, done, info = env.step(action)
                    score += reward

                    if not done:
                        next_action = self.epsilon_greedy(Q, next_state, nA, self.eps)
                        Q[state][action] = self.update_sarsa(alpha, gamma, Q, state, action, reward, next_state, next_action)
                        state = next_state
                        action = next_action
                    if done:
                        Q[state][action] = self.update_sarsa(alpha, gamma, Q, state, action, reward)
                        tmp_scores.append(score)
                        break
                if (i + 1) % plot_every == 0:
                    print("\rEpisode {}/{}".format(i + 1, num_episodes), end="")
                    sys.stdout.flush()
                    avg_scores.append(np.mean(tmp_scores))

        return Q, avg_scores

    def q_learning(self, plot_every=10):
        epochs = 1
        num_episodes = self.episodes
        alpha = self.alpha
        gamma = self.gamma
        nA = env.action_space.n
        Q = defaultdict(lambda: np.zeros(nA))

        tmp_scores = deque(maxlen=plot_every)
        avg_scores = deque(maxlen=num_episodes)
        for _ in range(epochs):
            for i in range(1, num_episodes + 1):
                score = 0
                state = env.reset()
                while True:
                    action = self.epsilon_greedy(Q, state, nA, self.eps)
                    next_state, reward, done, info = env.step(action)
                    score += reward
                    Q[state][action] = self.update_Q_learning(alpha, gamma, Q, state, action, reward, \
                                                         next_state)
                    state = next_state

                    if done:
                        tmp_scores.append(score)
                        break
                if i % plot_every == 0:
                    avg_scores.append(np.mean(tmp_scores))
                    print("\rEpisode {}/{}".format(i, num_episodes), end="")
                    sys.stdout.flush()

        return Q, avg_scores

if __name__ == "__main__":
    c = cliff(alpha=0.01, gamma=1.0, episodes=500)
    Q_sarsa, scores_sarsa = c.sarsa()
    Q_Qlearning, scores_Qlearning = c.q_learning()
    x = np.linspace(0, 500, len(scores_Qlearning), endpoint=False)

    plt.plot(x, np.asarray(scores_sarsa), label="Sarsa")
    plt.plot(x, np.asarray(scores_Qlearning), label="Q-Learning")
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward')
    plt.title("Sarsa Vs Q-Learning")
    plt.legend()
    plt.show()