import torch
import torch.nn as nn
import numpy as np
import random

class Q_meta_control(nn.Module):
	def __init__(self,):
		super(Q_meta_control,self).__init__()
		self.conv1=nn.Conv2d(4,32,kernel_size=8,stride=4)
		self.relu1=nn.ReLU(inplace=True)
		self.conv2=nn.Conv2d(32,64,kernel_size=4,stride=2)
		self.relu2=nn.ReLU(inplace=True)
		self.conv3=nn.Conv2d(64,64,kernel_size=3,stride=1)
		self.relu3=nn.ReLU(inplace=True)
		self.fc1=nn.Linear(,512)
		self.relu4=nn.ReLU(inplace=True)
		self.fc2=nn.Linear(512,)
	def forward(self,x):
		x=self.conv1(x)
		x=self.relu1(x)
		x=self.conv2(x)
		x=self.relu2(x)
		x=self.conv3(x)
		x=self.relu3(x)
		x=x.view(x.size(0),-1)
		x=self.fc1(x)
		x=self.relu4(x)
		x=self.fc2(x)
		return x 
		
def Eps_greedy(x,B,eps,Q)
	prob=np.random.random_sample()
	if prob < eps:
		return random.sample(B,1)
	else:
		values=Q(x)
		v,idx=torch.max(values,1)
		return idx
