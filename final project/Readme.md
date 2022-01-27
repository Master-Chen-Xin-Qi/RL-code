# Final Project

## 运行训练

- 在文件夹目录下执行下面语句即可开始训练，可选择的环境有PongNoFrameskip-v4、Humanoid-v2

  ```bash
  python3 run.py -e PongNoFrameskip-v4 -a DQN
  ```

- PongNoFrameskip-v4环境采用的算法是DQN和DDQN，Humanoid-v2环境采用的算法是PP0
- 训练后可以分别在 `DQN_train`、`DDQN_train` 和 `PPO_train` 下找到训练过程中保存的模型参数
- 已经训练好，可供测试的模型参数为 `DQN_best.pt` 和 `PPO_best.pt`

## 测试

- 在文件夹目录下执行下面语句即可开始测试训练好的模型

  ```bash
  python3 test.py -e Humanoid-v2
  ```

- 可以得到游戏实时的表现结果