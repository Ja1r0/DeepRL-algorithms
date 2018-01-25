import torch
from torch.autograd import Variable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from collections import namedtuple
import gym

def soft_update(new,old,tau):
    for param_new,param_old in zip(new.parameters(),old.parameters()):
        param_old.data.copy_(tau*param_new.data+(1.0-tau)*param_old.data)

def plot_graph(mean_reward):
    plt.figure(1)
    plt.clf()
    plt.plot(mean_reward)
    plt.xlabel('episode')
    plt.ylabel('mean reward')
    plt.pause(0.001)
class epsilon_greedy:
    def __init__(self,eps_start,eps_end,eps_frac):
        self.eps_start=eps_start
        self.eps_end=eps_end
        self.eps_frac=eps_frac
        self.eps=self.eps_start
    def choose_action(self,actor,obs,env):
        prob=np.random.random_sample()
        if prob<self.eps:
            action=env.action_space.sample()
        else:
            action=actor(obs)
        return action
    def updata(self,time_step):
        self.eps=self.eps_start-(self.eps_start-self.eps_end)/self.eps_frac*time_step

Transition = namedtuple('Transition', ['obs', 'act', 'reward',  'obs_next','done'])
class Replay_buffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
    def add(self, transition):
        obs=transition.obs # {ndarray} (1,...)
        action=transition.act
        reward=np.array([transition.reward])
        obs_next=transition.obs_next
        done=np.array([float(transition.done==False)])
        sample=Transition(obs,action,reward,obs_next,done)
        self.buffer.append(sample)
        self.buffer = self.buffer[-self.capacity:]
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return samples


class OUNoise:
    """should be reset before a new episode begin"""
    def __init__(self,action_dimension,mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


if __name__ == '__main__':
    '''
    Transition = namedtuple('Transition', ['obs', 'act', 'reward', 'done', 'obs_next'])
    obs=np.random.randn(3)
    obs_next=np.random.randn(3)
    act=np.random.randn(1)
    reward=random.random()
    done=False
    info={}
    sample=Transition(obs,act,reward,done,obs_next)
    memory=Replay_buffer(10)
    for i in range(10):
        memory.add(sample)
    print(memory.memory)
    obs = np.random.randn(3)
    obs_next = np.random.randn(3)
    act = np.random.randn(1)
    reward = random.random()
    done = False
    t = Transition(obs, act, reward, done, obs_next)
    memory.add(t)
    print(memory.memory)

    plt.ion()
    mean_reward=[]
    for i in range(10):
        reward=random.randint(-3,3)
        mean_reward.append(reward)
        plot_graph(mean_reward)
    env=gym.make('Pendulum-v0')
    action_dim=env.action_space.sample().size
    explor_noise=OUNoise(action_dimension=action_dim)
    explor_noise.reset()
    out=explor_noise.noise()
    print(out)
    '''
    ou = OUNoise(3)
    states = []
    for i in range(1000):
        states.append(ou.noise())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()