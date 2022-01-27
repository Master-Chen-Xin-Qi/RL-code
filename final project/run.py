# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :final project
# @File     :run.py
# @Date     :2022/1/7 14:36
# @Author   :Xinqi Chen
# @Software :PyCharm
-------------------------------------------------
"""

import argparse
import matplotlib.pyplot as plt
import torch
import gym
from collections import deque
from models import DQN, PPO
import math
import numpy as np
from atari_wrappers import make_atari, wrap_deepmind
from config import epsilon_max, epsilon_min, eps_decay, frames, USE_CUDA, learning_rate, max_buff, update_tar_interval, \
    batch_size, print_interval, learning_start, win_reward, win_break, DDQN_epsilon_min, models_name, env_name,\
    Iter, MAX_STEP
from utils import DQNAgent, DDQNAgent, Normalize

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='------Reinforcement Learning Final Project------')
    parser.add_argument('-e', type=str, default='PongNoFrameskip-v4', choices=env_name)
    parser.add_argument('-a', type=str, default='DQN', choices=models_name)
    args = parser.parse_args()
    if args.e == 'PongNoFrameskip-v4':
        env = make_atari('PongNoFrameskip-v4')
        env = wrap_deepmind(env, scale=False, frame_stack=True)
        # check the env
        # env.reset()
        # for i in range(1000):
        #     env.render()
        #     env.step(env.action_space.sample())
        action_space = env.action_space
        action_dim = env.action_space.n
        state_dim = env.observation_space.shape[0]
        state_channel = env.observation_space.shape[2]
        if args.algorithm == 'DQN':
            agent = DQNAgent(in_channels=state_channel, action_space=action_space, USE_CUDA=USE_CUDA, lr=learning_rate,
                         memory_size=max_buff)
        else:
            agent = DDQNAgent(in_channels=state_channel, action_space=action_space, USE_CUDA=USE_CUDA, lr=learning_rate,
                         memory_size=max_buff)

        frame = env.reset()
        episode_reward = 0
        max_reward = -100
        all_rewards = []
        losses = []
        episode_num = 0
        is_win = False

        # e-greedy decay
        if args.algorithm == 'DQN':
            epsilon_by_frame = lambda frame_idx: epsilon_min + (epsilon_max - epsilon_min) * math.exp(
                -1. * frame_idx / eps_decay)
        else:
            epsilon_by_frame = lambda frame_idx: DDQN_epsilon_min + (epsilon_max - DDQN_epsilon_min) * math.exp(
                -1. * frame_idx / eps_decay)
        # plt.plot([epsilon_by_frame(i) for i in range(10000)])
        if args.algorithm == 'DDQN':
            frames = frames//2
        for i in range(frames):
            epsilon = epsilon_by_frame(i)
            state_tensor = agent.observe(frame)
            action = agent.act(state_tensor, epsilon)

            next_frame, reward, done, _ = env.step(action)

            episode_reward += reward
            agent.memory_buffer.push(frame, action, reward, next_frame, done)
            frame = next_frame

            loss = 0
            if agent.memory_buffer.size() >= learning_start:
                loss = agent.learn_from_experience(batch_size)
                losses.append(loss)

            if i % print_interval == 0:
                print("Frames num: %5d, Reward: %4f, Loss: %4f, Epsilon: %4f, Episode: %4d" % (
                    i, np.mean(all_rewards[-10:]), loss, epsilon, episode_num))
                if np.mean(all_rewards[-10:]) > max_reward:
                    max_reward = np.mean(all_rewards[-10:])
                    print('Save model!')
                    save_name = args.algorithm + '_train_best.pt'
                    save_model = './' + args.algorithm + '_train/' + save_name
                    torch.save(agent.DQN.state_dict(), save_model)

            if i % update_tar_interval == 0:
                agent.DQN_target.load_state_dict(agent.DQN.state_dict())

            if done:
                frame = env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0
                episode_num += 1
                avg_reward = float(np.mean(all_rewards[-100:]))

    elif args.e == 'Hopper-v2':
        env = gym.make(args.e)
        N_S = env.observation_space.shape[0]
        N_A = env.action_space.shape[0]

        # check the env
        # env.reset()
        # for i in range(100000):
        #     env.render()
        #     env.step(env.action_space.sample())

        # initialize random seeds
        env.seed(100)
        torch.manual_seed(100)
        np.random.seed(100)
        ppo = PPO(N_S, N_A)
        nomalize = Normalize(N_S)
        episodes = 0
        eva_episodes = 0
        score_total = []
        max_reward = -100
        for i in range(Iter):
            memory = deque()
            rewards = []
            steps = 0
            while steps < 2048:
                episodes += 1
                s = nomalize(env.reset())
                reward = 0
                for _ in range(MAX_STEP):
                    steps += 1
                    # choose action
                    a = ppo.actor_net.choose_action(torch.from_numpy(np.array(s).astype(np.float32)).unsqueeze(0))[0]
                    s_, r, done, info = env.step(a)
                    s_ = nomalize(s_)

                    mask = (1 - done) * 1
                    memory.append([s, a, r, mask])
                    reward += r
                    s = s_
                    if done:
                        break
                with open('log_' + args.e + '.txt', 'a') as outfile:
                    outfile.write('\t' + str(episodes) + '\t' + str(reward) + '\n')
                rewards.append(reward)
            reward_avg = np.mean(rewards)
            print('{} episode score is {:.2f}'.format(episodes, reward_avg))
            if reward_avg > max_reward:
                max_reward = reward_avg
                torch.save(ppo.actor_net.state_dict(), './PPO_train/actor_best.pt')
                torch.save(ppo.critic_net.state_dict(), './PPO_train/critic_best.pt')
                print('Save model!')
            score_total.append(reward_avg)
            # update parameters
            ppo.train(memory)
        plt.plot(score_total)
        plt.title('Reward for PPO')
        plt.show()


