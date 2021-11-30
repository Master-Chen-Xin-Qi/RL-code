# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :pythonProject
# @File     :homework1
# @Date     :2021/9/29 14:07
# @Author   :Xinqi Chen
# @Software :PyCharm
-------------------------------------------------
"""


class GridWorld:
    def __init__(self, theta, edge_size=4):
        self.theta = theta
        self.states = [i for i in range(edge_size ** 2)]
        self.terminal_states = [0, edge_size ** 2 - 1]
        self.edge_size = edge_size
        self.values = dict()
        self.pi = dict()
        self.gamma = 1
        self.actions = ["north", "south", "west", "east"]
        # 初始化值函数
        for s in self.states:
            self.values[s] = 0
        # 初始化策略
        for s in self.states:
            self.pi[s] = []
            if s in self.terminal_states:
                continue
            self.pi[s].append("north")
            self.pi[s].append("south")
            self.pi[s].append("west")
            self.pi[s].append("east")
        self.optimal_pi = self.pi.copy()

    # 状态转为坐标
    def state_to_grid(self, state):
        row = state // 4
        col = state % 4
        return row, col

    # 坐标转为状态
    def grid_to_state(self, row, col):
        state = row * 4 + col
        return state

    # 做出相应action
    def move(self, state, action):
        if state in self.terminal_states:
            return state, 0
        row, col = self.state_to_grid(state)
        if action == "north":
            row = row - 1
            if row < 0:
                row = 0
        if action == "south":
            row = row + 1
            if row >= self.edge_size:
                row = self.edge_size - 1
        if action == "west":
            col = col - 1
            if col < 0:
                col = 0
        if action == "east":
            col = col + 1
            if col >= self.edge_size:
                col = self.edge_size - 1
        state = self.grid_to_state(row, col)
        return state, -1

    # 显示GridWorld
    def show(self, values):
        print("————————————————————————")
        print("|%.1f|%.1f|%.1f|%.1f|" % (values[0], values[1], values[2], values[3]))
        print("————————————————————————")
        print("|%.1f|%.1f|%.1f|%.1f|" % (values[4], values[5], values[6], values[7]))
        print("————————————————————————")
        print("|%.1f|%.1f|%.1f|%.1f|" % (values[8], values[9], values[10], values[11]))
        print("————————————————————————")
        print("|%.1f|%.1f|%.1f|%.1f|" % (values[12], values[13], values[14], values[15]))
        print("————————————————————————")

    # 策略更新迭代值函数
    def policy_iteration(self):
        print("----------iteration 0------------")
        self.show(self.values)
        for i in range(1000):
            delta = 0
            last_values = self.values.copy()
            for s in self.states:
                if s in self.terminal_states:
                    continue
                cur_v = last_values[s]
                new_value = 0
                for act in self.actions:
                    new_s, reward = self.move(s, act)
                    del_value = round(0.25 * (reward + self.gamma * last_values[new_s]), 5)
                    new_value += del_value
                self.values[s] = new_value
                dis = [delta, abs(cur_v - self.values[s])]
                delta = max(dis)
            print("----------iteration %d------------" % (i + 1))
            self.show(self.values)
            if delta < self.theta:
                break

    # 改进策略
    def policy_improvement(self):
        policy_stable = True
        for s in self.states:
            if s in self.terminal_states:
                continue
            old_action = self.optimal_pi[s]
            max_value = -10e5
            for act in self.actions:
                new_s, reward = self.move(s, act)
                new_value = round(0.25 * (reward + self.gamma * self.values[new_s]), 5)
                # 找出四种移动方式中value最大的作为该点的策略
                if new_value > max_value:
                    max_value = new_value
                    self.optimal_pi[s] = [act]
                if new_value == max_value and act not in self.optimal_pi[s]:
                    self.optimal_pi[s].append(act)
            if old_action != self.optimal_pi[s]:
                policy_stable = False
            print(s, ": %s" % self.optimal_pi[s])
        return policy_stable


if __name__ == "__main__":
    o = GridWorld(theta=0.0001, edge_size=4)
    for i in range(100):
        o.policy_iteration()
        stable_flag = o.policy_improvement()
        print(stable_flag)
        if stable_flag:
            break
