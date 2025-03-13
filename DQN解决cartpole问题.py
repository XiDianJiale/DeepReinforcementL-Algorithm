#form gym import "cat"
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import random
import math

from collections import deque



class DQN(nn.Module):  #后期预测的奖励等于DQN（state）
    def __init__(self,input_dim,output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    def forward(self,x):
        x = torch.relu(self.fc1(x))  #只传入一个参数x即可，不需要传入维度信息
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  #最后一层不用RELu
        return x

    #经验回放
#class ex
class ReplayBuffer:  #经验回放的作用一是提升对样本和经验的利用效率。二十帮助模型避免依赖数据相关性
    def __init__(self,capacity):
        self.buffer = deque(maxlen=capacity)   #我在今天编程时也感受到这里作用是定义类的固有特征和属性的内容

    def push(self,state,action,reward,next_state,done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self,batch_size):
        batch = random.sample(self.buffer,batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

# 超参数设置
learning_rate = 1e-3
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
batch_size = 64
replay_buffer_size = 10000
target_update_frequency = 100

#初始化环境和模型固定思路
env = gym.make('CartPole-v1')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
policy_net = DQN(input_dim,output_dim)
target_net = DQN(input_dim,output_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(replay_buffer_size)

#这里前面的准备工作都已经做好，在正式开始训练前我们coding的内容：定义DQN用于用于预测奖励Q，定义经验回放机制用于训练的过程中调用
#下面是正式训练的coding，我们需要做的是知道模型action和更新参数（注意不是及时更新,以及我们epsilon探索技巧也要融入

#3def train:
num_episodes = 500
epsilon = epsilon_start
for episode in range(num_episodes):
    state,_ = env.reset()
    done = False
    total_reward = 0

  #  while done:
    while not done:
        if random.random()<epsilon:
            action = random.randrange(output_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = policy_net(state_tensor)
            action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action) #执行动作获得下一状态各个参数
            total_reward += reward
            # 将经验存入回放缓冲区
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            # 从回放缓冲区采样并训练
            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states_tensor = torch.FloatTensor(states)
                actions_tensor = torch.LongTensor(actions).unsqueeze(1)
                rewards_tensor = torch.FloatTensor(rewards)
                next_states_tensor = torch.FloatTensor(next_states)
                dones_tensor = torch.FloatTensor(dones)

                # 计算 Q 值
                q_values = policy_net(states_tensor).gather(1, actions_tensor).squeeze()
                next_q_values = target_net(next_states_tensor).max(1)[0]

                target_q_values = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)

                # 计算损失并更新模型
                loss = nn.MSELoss()(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # 更新目标网络
                # 更新目标网络
            if episode % target_update_frequency == 0:
                target_net.load_state_dict(policy_net.state_dict())
    # 更新 epsilon
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    # 打印训练信息
    print(f'Episode {episode + 1}, Reward: {total_reward}, Epsilon: {epsilon}')

print('Training finished!')



