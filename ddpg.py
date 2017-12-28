import torch
import torch.nn as nn
import numpy as np
import gym
import random
from collections import namedtuple
import matplotlib.pyplot as plt

def obs_process(frame):
    '''
    Parameters
    ----------
    frame: {ndarray} of shape (_,_,3)
    Returns
    -------
    frame: {Tensor} of shape torch.Size([1,84,84])
    '''
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    frame=frame.astype('float64')
    frame=frame/255.
    frame=torch.from_numpy(frame)
    frame=frame.unsqueeze(0).type(Tensor)
    return frame
	
def plot_graph(mean_reward):
	plt.figure()
	plt.xlabel('episode')
	plt.ylabel('mean reward')
	plt.legend()
	plt.plot(mean_reward)
	plt.show()





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
	gamma,
	):
		self.env=env
		self.eps_start=eps_start
		self.eps_end=eps_end
		self.eps_frac=eps_frac
		self.train_freq=train_freq
		self.update_target_frac=update_target_frac
		self.batch_size=batch_size
		self.tau=tau
		self.gamma=gamma
		
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
			obs_batch=stacks.obs
			act_batch=stacks.act
			reward_batch=stacks.act
			done_batch=stacks.done
			obs_next_batch=stacks.obs_next
			self.train_actor(obs_batch,act_batch) # 这里对两个网络的更新是用的同一批样本(?)
			self.train_critic(obs_batch,act_batch,reward_batch,obs_next_batch)
		if time_step % self.update_target_frac == 0: # 两个目标网络同时更新
			self.actor_target.load_state_dict(self.actor.state_dict())
			self.critic_target.load_state_dict(self.critic.state_dict())
		
	def memory(self):
		pass
	def train_actor(self,obs_batch,act_batch):
		obs_batch=Variable(torch.cat(Tensor(obs_batch)))
		act_batch=Variable(torch.cat(Tensor(act_batch)))
		l=-self.critic(obs_batch,self.actor(obs_batch))
		loss=torch.mean(l)
		self.actor.zero_grad()
		loss.backward()
		optim=nn.Adam()
		optim.step()		
	def train_critic(self,obs_batch,act_batch,reward_batch,obs_next_batch):
		reward_batch=Variable(torch.cat(Tensor(reward_batch)))
		obs_next_batch=Variable(torch.cat(Tensor(obs_next_batch)))
		obs_batch=Variable(torch.cat(Tensor(obs_batch)))
		act_batch=Variable(torch.cat(Tensor(act_batch)))
		target_batch=reward_batch+self.gamma*self.critic_target(obs_next_batch,self.actor_target(obs_next_batch))
		actual_batch=self.critic(obs_batch,act_batch)
		loss=nn.MSELoss(target_batch,actual_batch)
		self.critic.zero_grad()
		loss.backward()
		optim=nn.Adam()
		optim.step()
use_gpu=torch.cuda.is_available()
FloatTensor=torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
Tensor=FloatTensor
### parameters ###
env_id='Pendulum-v0'
Transition=namedtuple('Transition',['obs','act','reward','done','obs_next'])
###
def play():
	env=gym.make(env_id)
	action0=env.action_space.sample()
	obs0,_,_,_=env.step(action0)
	obs0=obs_process(obs0)
	obs=torch.cat((obs0,obs0,obs0,obs0),0) # {Tensor}
	time_step=0
	reward_list=[]
	epi_reward=[]
	mean_reward=[]
	agent=Ddpg()
	plt.ion()
	for epi_num in range(max_episode):
		action=agent.actor(obs)
		obs_next,reward,done,info=env.step(action)
		
		transition=Transition(obs,action,reward,done,obs_next)
		obs=torch.cat((obs[1:,:,:],obs_next),0)
		reward_list.append(reward)
		agent.update(time_step,epi_num,transition)
		step+=1
		if done:
			epi_total_reward=sum(reward_list)
			epi_reward.append(epi_total_reward)
			mean100=np.mean(epi_reward[-101:-1])
			mean_reward.append(mean100)
			plot_graph(mean_reward)
			obs=env.reset()
			print(print('episode %d timestep %d : threshold=%.5f total reward=%.2f'
			%(epi_num,time_step,epi_total_reward))
			break
	
if __name__ == '__main__':
    play()
	
	
	
	
	
	
	