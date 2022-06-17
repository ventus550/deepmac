import torch
import numpy as np
from QLearning import DQN, ReplayMemory, Qptimizer
from QLearning.utilities import select_epsilon, get_device
from pacman import Environment, RandomAgent, Agent, GameController

device = get_device()
height, width = Environment().shape

def QTensor(array, dtype=torch.float):
	return torch.tensor(np.array(array), dtype=dtype)

class DeepAgent(Agent):
	def __init__(self, speed=0.1):
		super().__init__(speed)

		self.policy_net = DQN(height, width, 4).to(device)
		self.target_net = DQN(height, width, 4).to(device)
		self.target_net.copy_from(self.policy_net)
		self.target_net.eval()

		self.memory = ReplayMemory(10000)
		self.optimizer = torch.optim.RMSprop(self.policy_net.parameters())
		self.optimizer = Qptimizer(
			self.memory,
			self.optimizer,
			self.policy_net,
			self.target_net
		)

	def __call__(self, state):
		action = select_epsilon(self.policy_net, QTensor([state])) #!
		self.actions[action]()

	def feedback(self, state, action, reward, state_new):
		self.memory.push(QTensor([state]), QTensor([action]), QTensor([reward]), QTensor([state_new]))
		self.optimizer()



target_net_update = 2
ghosts = [RandomAgent(0.05) for _ in range(3)]
pacman = DeepAgent(1)

for episode in range(10):
	env = GameController(pacman, ghosts)
	while not env.terminal():
		next(env).update()
	if episode % target_net_update == 0:
		pacman.target_net.copy_from(pacman.policy_net)








