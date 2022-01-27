# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :final project
# @File     :test
# @Date     :2022/1/8 11:53
# @Author   :Xinqi Chen
# @Software :PyCharm
-------------------------------------------------
"""
import time
import torch
import argparse
import gym
from utils import DQNAgent, DDQNAgent, Normalize
from atari_wrappers import make_atari, wrap_deepmind
from config import max_buff, USE_CUDA, learning_rate, MAX_STEP
from models import DQN, PPO
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser('------Test algorithm')
    parser.add_argument('-t', '--test', type=str, default='PPO', choices=['DQN', 'DDQN', 'PPO'])
    arg = parser.parse_args()
    if arg.test == 'DQN':
        env = make_atari('PongNoFrameskip-v4')
        env = wrap_deepmind(env, scale=False, frame_stack=True)
        frame = env.reset()
        action_space = env.action_space
        action_dim = env.action_space.n
        state_dim = env.observation_space.shape[0]
        state_channel = env.observation_space.shape[2]
        agent = DQNAgent(in_channels=state_channel, action_space=action_space, USE_CUDA=USE_CUDA, lr=learning_rate,
                         memory_size=max_buff)
        agent.DQN = DQN(in_channels=4, act_output=action_dim)
        agent.DQN.load_state_dict(torch.load('./DQN_best.pt'))
        device = torch.device('cuda:0')
        agent.DQN.to(device)
        total_reward = 0
        while True:
            env.render()
            time.sleep(0.01)
            state_tensor = agent.observe(frame)
            action = agent.act(state_tensor, 0)
            next_frame, reward, done, _ = env.step(action)
            frame = next_frame
            total_reward += reward
            if total_reward == 21:
                print('Game Over! Win!')
                break
    elif arg.test == 'PPO':
        env = gym.make('Hopper-v2')
        env.reset()
        N_S = env.observation_space.shape[0]
        N_A = env.action_space.shape[0]
        ppo = PPO(N_S, N_A)
        # load saved model
        ppo.actor_net.load_state_dict(torch.load('./PPO_train/actor_best.pt'))
        ppo.critic_net.load_state_dict(torch.load('./PPO_train/critic_best.pt'))
        normalize = Normalize(N_S)
        steps = 0
        while steps < 2048:
            s = normalize(env.reset())
            reward = 0
            for _ in range(MAX_STEP):
                env.render()
                steps += 1
                # choose action
                a = ppo.actor_net.choose_action(torch.from_numpy(np.array(s).astype(np.float32)).unsqueeze(0))[0]
                s_, r, done, info = env.step(a)
                s_ = normalize(s_)
                mask = (1 - done) * 1
                s = s_
                if done:
                    break

