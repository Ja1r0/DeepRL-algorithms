import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import gym
import random
from collections import namedtuple
import matplotlib.pyplot as plt
from network import *
from utils import *


# 问题1：参数的初始化有什么作用
### parameters ###
env_id = 'Pendulum-v0'
Max_episode =200
Maxsteps_per_epi=200
Train_freq=1
Update_target_frac=1
Batch_size=32
Tau=0.01
Gamma=0.9
Capacity=10000
Observe=10000
A_lr=1e-3
C_lr=2e-3
Render=False
###

class Ddpg:
    def __init__(self,
                 train_freq=Train_freq,
                 update_target_frac=Update_target_frac,
                 batch_size=Batch_size,
                 tau=Tau,
                 gamma=Gamma,
                 capacity=Capacity,
                 act_dim=1,
                 obs_dim=3,
                 var=3.0
                 ):
        self.train_freq = train_freq
        self.update_target_frac = update_target_frac
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.memory = Replay_buffer(capacity)
        self.observe=Observe
        self.actor = Actor_simple(obs_dim, act_dim).cuda()
        self.actor_target = Actor_simple(obs_dim, act_dim).cuda()
        self.critic = Critic_simple(obs_dim, act_dim).cuda()
        self.critic_target = Critic_simple(obs_dim, act_dim).cuda()
        self.a_lr=A_lr
        self.c_lr=C_lr
        self.var=var
        for param in self.actor_target.parameters():
            param.requires_grad=False
        for param in self.critic_target.parameters():
            param.requires_grad=False
    def action(self, obs, noise,act_low,act_high):
        action = self.actor.forward(obs).data.cpu().numpy()  # {ndarray}
        action_noise = np.random.normal(action.squeeze(0),noise)
        action=np.clip(action_noise,act_low,act_high)
        return action

    def update(self, time_step,transition):  # time_step should start from 0
        self.memory.add(transition)
        if time_step % self.train_freq == 0 and time_step > self.observe:  # 两个网络同时更新
            samples = self.memory.sample(self.batch_size)
            trans = Transition(*zip(*samples))
            obs_batch = np.stack(trans.obs)
            act_batch = np.stack(trans.act)
            reward_batch = np.stack(trans.reward)
            done_batch = np.stack(trans.done)
            obs_next_batch = np.stack(trans.obs_next)
            self.train_actor(obs_batch)  # 这里对两个网络的更新是用的同一批样本(?)
            self.train_critic(obs_batch, act_batch, reward_batch, done_batch, obs_next_batch)
            self.var *= 0.9995
            if time_step % self.update_target_frac == 0:  # 两个目标网络同时更新
                soft_update(self.actor,self.actor_target,self.tau)
                soft_update(self.critic,self.critic_target,self.tau)

    def train_actor(self, obs_batch):
        obs_batch = Variable(Tensor(obs_batch))
        l = self.critic(obs_batch, self.actor(obs_batch))
        loss = -torch.mean(l)
        self.actor.zero_grad()
        loss.backward()
        optimizer = optim.Adam(self.actor.parameters(),lr=self.a_lr) # initial lr=1e-4
        optimizer.step()

    def train_critic(self, obs_batch, act_batch, reward_batch, done_batch, obs_next_batch):
        obs_batch = Variable(Tensor(obs_batch))
        act_batch = Variable(Tensor(act_batch))
        reward_batch = Variable(Tensor(reward_batch))
        done_batch = Variable(Tensor(done_batch))
        obs_next_batch = Variable(Tensor(obs_next_batch))
        next_q = self.critic_target(obs_next_batch, self.actor_target(obs_next_batch))
        target_batch = reward_batch + self.gamma * next_q * done_batch
        actual_batch = self.critic(obs_batch, act_batch)
        loss = nn.MSELoss()
        output=loss(actual_batch,target_batch)
        self.critic.zero_grad()
        output.backward()
        optimizer = optim.Adam(self.critic.parameters(),lr=self.c_lr, weight_decay=1e-2) # initial lr=1e-3
        optimizer.step()


use_gpu = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
Tensor = FloatTensor
Transition = namedtuple('Transition', ['obs', 'act', 'reward',  'obs_next','done'])

def play():
    env = gym.make(env_id)
    env=env.unwrapped
    env.seed(1)
    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    agent = Ddpg(act_dim=act_dim,obs_dim=obs_dim)
    explore_noise=OUNoise(act_dim)
    render=Render
    epi_num = 0
    time_step=0
    var=3.0
    reward_list = []
    epi_reward = []
    mean_reward = []
    plt.ion()
    obs_np = env.reset()
    obs = Variable(torch.unsqueeze(Tensor(obs_np),0))
    while True:
        if render:
            env.render()
        action = agent.action(obs,var,env.action_space.low,env.action_space.high) # size:(1,)
        obs_next, reward, done, info = env.step(action)
        transition = Transition(obs_np, action, reward/10,  obs_next,done)
        reward_list.append(reward)
        agent.update(time_step, transition)
        time_step += 1

        if done or time_step%Maxsteps_per_epi==0:
            epi_total_reward = sum(reward_list)
            epi_reward.append(epi_total_reward)
            if epi_num > 70:
                render = True
            reward_list=[]
            mean100 = np.mean(epi_reward[-101:-1])
            mean_reward.append(mean100)
            plot_graph(mean_reward)
            obs_np = env.reset()
            obs=Variable(torch.unsqueeze(Tensor(obs_np),0))
            explore_noise.reset()
            epi_num+=1
            print('episode %d timestep %d : total reward=%.2f' % (epi_num, time_step, epi_total_reward))
        else:
            obs_np=obs_next
            obs = Variable(torch.unsqueeze(Tensor(obs_next), 0))
        if epi_num > Max_episode:
            break

if __name__ == '__main__':
    play()







