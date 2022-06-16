import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from QLearning.qnetwork import DQN
from QLearning import ReplayMemory
from pacman import Environment, RandomAgent, Agent
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

device = 'cpu'
n_actions = 4
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
height, width = (28, 29)

def QTensor(array):
	return torch.tensor(np.array(array), dtype=torch.float)

def optimize_model(memory, optimizer, policy_net, target_net):
	if len(memory) < BATCH_SIZE:
		return
	transitions = memory.sample(BATCH_SIZE)
	# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
	# detailed explanation). This converts batch-array of Transitions
	# to Transition of batch-arrays.
	batch = Transition(*zip(*transitions))
	state_batch = torch.cat(batch.state)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)
	next_batch = torch.cat(batch.next_state)

	# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
	# columns of actions taken. These are the actions which would've been taken
	# for each batch state according to policy_net
	state_action_values = policy_net(state_batch).gather(1, action_batch)

	# Compute V(s_{t+1}) for all next states.
	# Expected values of actions for next_batch are computed based
	# on the "older" target_net; selecting their best reward with max(1)[0].
	# This is merged based on the mask, such that we'll have either the expected
	# state value or 0 in case the state was final.
	next_state_values = target_net(next_batch).max(1)[0].detach()
	# Compute the expected Q values
	expected_state_action_values = (next_state_values * GAMMA) + reward_batch

	# Compute Huber loss
	criterion = nn.SmoothL1Loss()
	loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

	# Optimize the model
	optimizer.zero_grad()
	loss.backward()
	for param in policy_net.parameters():
		param.grad.data.clamp_(-1, 1)
	optimizer.step()



steps_done = 0
def select_action(policy_net, state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)



class DeepAgent(Agent):
	def __init__(self, speed=0.1):
		super().__init__(speed)

		self.policy_net = DQN(height, width, n_actions).to(device)
		self.target_net = DQN(height, width, n_actions).to(device)
		self.target_net.load_state_dict(self.policy_net.state_dict())
		self.target_net.eval()

		self.optimizer = optim.RMSprop(self.policy_net.parameters())
		self.memory = ReplayMemory(10000)

	def __call__(self, state):
		action = select_action(state) #!
		self.actions[action]()

	def feedback(self, transition: Transition):
		state, action, reward, state_new = transition



ghosts = [RandomAgent(0.05) for _ in range(3)]
pacman = DeepAgent()
env = Environment(pacman, ghosts)







