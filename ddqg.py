import torch
import torch.nn as nn
import numpy as np
import gym
import random
from collections import namedtuple

# 问题1：参数的初始化有什么作用
class Actor_net(nn.Module):
	def __init__(self,obs_dim,act_dim):
		super(Actor_net,self).__init__()
		self.fc1=nn.Linear(obs_dim,400)
		self.relu1=nn.ReLU(inplace=True)
		self.fc2=nn.Linear(400,300)
		self.relu2=nn.ReLU(inplace=True)
		self.fc3=nn.Linear(300,act_dim)
		self.tanh=nn.Tanh()
		# initial the parameters of layers
		nn.init.uniform(self.fc1.weight,-1.0/np.sqrt(obs_shape),1.0/np.sqrt(obs_shape))
		nn.init.uniform(self.fc1.bias,-1.0/np.sqrt(obs_shape),1.0/np.sqrt(obs_shape))
		nn.init.uniform(self.fc2.weight,-1.0/np.sqrt(400),1.0/np.sqrt(400))
		nn.init.uniform(self.fc2.bias,-1.0/np.sqrt(400),1.0/np.sqrt(400))
		nn.init.uniform(self.fc3.weight,-3e-3,3e-3)
		nn.init.uniform(self.fc3.bias,-3e-3,3e-3)		
		self.layers=nn.Sequential(
		self.fc1,
		self.relu1,
		self.fc2,
		self.relu2,
		self.fc3,
		self.tanh
		)
	def forward(self,x):
		out=self.layers(x)
		return out
	def train(self,samples):
		
		
		
		
		
		
class Critic_net(nn.Module):
	def __init__(self,obs_dim,act_dim):
		super(Critic_net,self).__init__()
		self.fc1=nn.Linear(obs_dim,400)
		self.relu1=nn.ReLU(inplace=True)
		self.fc2=nn.Linear(400+act_dim,300)
		self.relu2=nn.ReLU(inplace=True)
		self.fc3=nn.Linear(300,1)
		#self.tanh=nn.Tanh()
		# initial the parameters of layers
		nn.init.uniform(self.fc1.weight,-1.0/np.sqrt(obs_shape),1.0/np.sqrt(obs_dim))
		nn.init.uniform(self.fc1.bias,-1.0/np.sqrt(obs_shape),1.0/np.sqrt(obs_dim))
		nn.init.uniform(self.fc2.weight,-1.0/np.sqrt(400+act_dim),1.0/np.sqrt(400+act_dim))
		nn.init.uniform(self.fc2.bias,-1.0/np.sqrt(400+act_dim),1.0/np.sqrt(400+act_dim))
		nn.init.uniform(self.fc3.weight,-3e-3,3e-3)
		nn.init.uniform(self.fc3.bias,-3e-3,3e-3)		
		
	def forward(self,s,a):
		x=self.fc1(s)
		x=self.relu1(x)
		x=torch.stack(x,a,dim=1)
		x=self.fc2(x)
		x=self.relu2(x)
		x=self.fc3(x)
		return x 
		
class Replay_buffer:
	def __init__(self,capacity):
		self.capacity=capacity
		
		self.memory=[]
	def add(self,transition):
		self.memory.append(transition)
		self.memory=self.memory[-self.capacity:]
	def sample(self,batch_size):
		samples=random.sample(self.memory,batch_size)
		return samples
	

class Ddpg:
	def __init__(self,
	env,
	eps_start,
	eps_end,
	eps_frac,
	train_freq,
	update_target_frac,
	batch_size,
	tau,
	):
		self.env=env
		self.eps_start=eps_start
		self.eps_end=eps_end
		self.eps_frac=eps_frac
		self.train_freq=train_freq
		self.update_target_frac=update_target_frac
		self.batch_size=batch_size
		self.tau=tau
		
		self.act_dim=self.env.action_space.shape[0] 
		self.obs_dim=self.env.observation_space.shape[0] 
		self.actor=Actor_net(obs_dim,act_dim)
		self.actor_target=self.actor
		self.critic=Critic_net(obs_dim,act_dim)
		self.critic_target=self.critic
		self.epsilon=self.eps_start
	def action(self,obs):
		prob=random.uniform(0,1)
		if prob < self.epsilon:
			action=self.env.action_space.sample() # {ndarray}
		else:
			action=self.actor(obs).data.numpy() # {ndarray}
		return action
	def update(self,time_step,episode_num,trainsition): # time_step should start from 0
		self.epsilon=self.eps_start-(self.eps_start-self.eps_end)/self.eps_frac*time_step
		self.replay_buffer.update(transition)
		if time_step % self.train_freq == 0: # 两个网络同时更新
			samples=self.relay_buffer.sample(self.batch_size)
			stacks=Transition(*zip(*samples))
			obs_batch=np.array(stacks.obs)
			act_batch=np.array(stacks.act)
			reward_batch=np.array(stacks.act)
			done_batch=np.array(stacks.done)
			obs_next_batch=np.array(stacks.obs_next)
			self.train_actor(obs_batch,act_batch) # 这里对两个网络的更新是用的同一批样本(?)
			self.train_critic(reward_batch,obs_next_batch)
		if time_step % self.update_target_frac == 0: # 两个目标网络同时更新
			self.actor_target.load_state_dict(self.actor.state_dict())
			self.critic_target.load_state_dict(self.critic.state_dict())
		
	def memory(self):
	def train_actor(self,obs_batch,act_batch):
		
		
		
		
		
	def train_critic(self,reward_batch,obs_next_batch):
	
use_gpu=torch.cuda.is_available()
FloatTensor=torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
Tensor=FloatTensor
### parameters ###
env_id='Pendulum-v0'
Transition=namedtuple('Transition',['obs','act','reward','done','obs_next'])
###
def play():
	env=gym.make(env_id)
	
	