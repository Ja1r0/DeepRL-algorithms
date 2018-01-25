import random

class Decision_process:
	def __init__(self):
		self.a_space=['left','right']
		self.s_space=['s1','s2','s3','s4','s5','s6']
		self.visits={'s1':0,'s2':0,'s3':0,'s4':0,'s5':0,'s6':0}
		self.state='s2'
		self.terminal=False
	def step(self,action):
		if action=='left':
			idx=self.s_space.index(self.state)-1
		elif action=='right':
			if self.state=='s6':
				a=random.choice([-1,0])
				idx=self.s_space.index(self.state)+a
			else:
				a=random.choice([-1,1])
				idx=self.s_space.index(self.state)+a
		next_state=self.s_space[idx]
		self.state=next_state
		self.visits[self.state]+=1
		if self.state=='s1':
			self.terminal=True
			if self.visits['s6']!=0:
				reward=1
			else:
				reward=1.0/100
		else:
			self.terminal=False	
		return reward,next_state,self.terminal		
	def reset(self):
		for k,n in self.visits.items():
			n=0
		self.state='s2'

class Q_goal:
	def __init__(self):
		self.s1={'s2':0.,'s3':0.,'s4':0.,'s5':0.,'s6':0.}
		self.s2={'s1':0.,'s3':0.,'s4':0.,'s5':0.,'s6':0.}
		self.s3={'s1':0.,'s2':0.,'s4':0.,'s5':0.,'s6':0.}
		self.s4={'s1':0.,'s2':0.,'s3':0.,'s5':0.,'s6':0.}
		self.s5={'s1':0.,'s2':0.,'s3':0.,'s4':0.,'s6':0.}
		self.s6={'s1':0.,'s2':0.,'s3':0.,'s4':0.,'s5':0.}
		self.space=[self.s1,self.s2,self.s3,self.s4,self.s5,self.s6]
	def reset(self):
		for s in self.space:
			for k,v in s.items():
				v=0.0
class Q_action:
	def __init__(self):
		self.s1={'s2':{'left':0.0,'right':0.0},'s3':{'left':0.0,'right':0.0},
		's4':{'left':0.0,'right':0.0},'s5':{'left':0.0,'right':0.0},'s6':{'left':0.0,'right':0.0}}
		self.s2={'s1':{'left':0.0,'right':0.0},'s3':{'left':0.0,'right':0.0},
		's4':{'left':0.0,'right':0.0},'s5':{'left':0.0,'right':0.0},'s6':{'left':0.0,'right':0.0}}
		self.s3={'s1':{'left':0.0,'right':0.0},'s2':{'left':0.0,'right':0.0},
		's4':{'left':0.0,'right':0.0},'s5':{'left':0.0,'right':0.0},'s6':{'left':0.0,'right':0.0}}
		self.s4={'s1':{'left':0.0,'right':0.0},'s2':{'left':0.0,'right':0.0},
		's3':{'left':0.0,'right':0.0},'s5':{'left':0.0,'right':0.0},'s6':{'left':0.0,'right':0.0}}
		self.s5={'s1':{'left':0.0,'right':0.0},'s2':{'left':0.0,'right':0.0},
		's3':{'left':0.0,'right':0.0},'s4':{'left':0.0,'right':0.0},'s6':{'left':0.0,'right':0.0}}
		self.s6={'s1':{'left':0.0,'right':0.0},'s2':{'left':0.0,'right':0.0},
		's3':{'left':0.0,'right':0.0},'s4':{'left':0.0,'right':0.0},'s5':{'left':0.0,'right':0.0}}
		self.space=[self.s1,self.s2,self.s3,self.s4,self.s5,self.s6]
	def reset(self):
		for s in self.space:
			for state,actions in s.items():
				for a,v in actions.items():
					v=0.0
def Eps_greedy(x,B,eps,Q)
	prob=np.random.random_sample()
	if prob < eps:
		return random.choice(B)
	else:
		if isinstance(x,str):
			values=Q.x
		elif isinstance(x,tuple):
			s=x[0]
			g=x[1]
			values=Q.g[s]
		k,v=sorted(values.iteritems(),key=lambda values:values[1],reverse=True)
		return k

Eps1=1.0
Eps2=1.0
Num_episodes=200

	
def Learning():
	### initialize ###
	D1=[] # 1 denote for controller
	D2=[] # 2 denote for meta-controller
	G=['s1','s2','s3','s4','s5','s6'] # goal space
	A=['left','right'] # action space
	Q1=Q_tabular(type='state_action')
	Q2=Q_tabular(type='goal')
	eps1=Eps1
	eps2=Eps2
	###
	for i in range(Num_episodes):
		game=Decision_process()
		s0=game.state
		g=Eps_greedy(s,G,eps2,Q2)
		while not game.terminal:
			F=0.0
			s=s0
			while not (game.terminal or g==s):
				a=Eps_greedy((s,g),A,eps1,Q1)
				
		
		
		
	
	