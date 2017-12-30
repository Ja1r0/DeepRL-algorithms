import torch
import torch.nn as nn
from torch.autograd import Variable
import gym
import numpy as np

class Actor_net(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor_net, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 400)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(400, 300)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(300, act_dim)
        self.tanh = nn.Tanh()
        # initial the parameters of layers
        nn.init.uniform(self.fc1.weight, -1.0 / np.sqrt(obs_dim), 1.0 / np.sqrt(obs_dim))
        nn.init.uniform(self.fc1.bias, -1.0 / np.sqrt(obs_dim), 1.0 / np.sqrt(obs_dim))
        nn.init.uniform(self.fc2.weight, -1.0 / np.sqrt(400), 1.0 / np.sqrt(400))
        nn.init.uniform(self.fc2.bias, -1.0 / np.sqrt(400), 1.0 / np.sqrt(400))
        nn.init.uniform(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform(self.fc3.bias, -3e-3, 3e-3)
        self.layers = nn.Sequential(
            self.fc1,
            self.relu1,
            self.fc2,
            self.relu2,
            self.fc3,
            self.tanh
        )
    def forward(self, s):
        out = self.layers(s)
        return out


class Critic_net(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Critic_net, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 400)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(400 + act_dim, 300)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(300, 1)
        # self.tanh=nn.Tanh()
        # initial the parameters of layers
        nn.init.uniform(self.fc1.weight, -1.0 / np.sqrt(obs_dim), 1.0 / np.sqrt(obs_dim))
        nn.init.uniform(self.fc1.bias, -1.0 / np.sqrt(obs_dim), 1.0 / np.sqrt(obs_dim))
        nn.init.uniform(self.fc2.weight, -1.0 / np.sqrt(400 + act_dim), 1.0 / np.sqrt(400 + act_dim))
        nn.init.uniform(self.fc2.bias, -1.0 / np.sqrt(400 + act_dim), 1.0 / np.sqrt(400 + act_dim))
        nn.init.uniform(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, s, a):
        x = self.fc1(s)
        x = self.relu1(x)
        x = torch.cat((x, a), dim=1)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x



if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
    Tensor = FloatTensor
    '''
    actor=Actor_net(3,1)
    actor.cuda()
    params=actor.state_dict()
    for k,v in params.items():
        print(k)
    print(params['fc1.weight'].max())
    print(params['fc1.weight'].min())
    
    critic=Critic_net(3,1)
    critic.cuda()
    params=critic.state_dict()
    for k,v in params.items():
        print(k)
    print(params['fc3.weight'].max())
    print(params['fc3.weight'].min())
    print(params['fc3.bias'])
    '''
    env=gym.make('Pendulum-v0')
    obs = env.reset()
    act_dim=env.action_space.sample().size
    obs_dim=obs.reshape(obs.size).size
    obs=Variable(torch.unsqueeze(Tensor(obs),0))
    actor=Actor_net(obs_dim=obs_dim,act_dim=act_dim).cuda()
    critic=Critic_net(obs_dim=obs_dim,act_dim=act_dim).cuda()
    action=actor.forward(obs)
    act1x1=action.data.cpu().numpy()
    act1=action.data.squeeze(0).cpu().numpy()
    #print(env.step(act1))
    print(env.step(act1x1))
    print(env.step(env.action_space.sample()))
    #print(env.action_space.sample())
    #print(act1)
    #print(act1x1)
    #print(action.data.cpu().numpy())
    value=critic.forward(obs,action)
    #print(value)


