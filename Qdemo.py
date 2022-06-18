import torch
import numpy as np
from QLearning import DQN, ReplayMemory, Qptimizer
from QLearning.utilities import select_epsilon, get_device, quickplot
from pacman import Environment, RandomAgent, Agent, GameController

device = get_device()
height, width = Environment().shape

def QTensor(t, dtype=torch.float):
	return t.unsqueeze(0)

class DeepAgent(Agent):
	def __init__(self, speed=0.1):
		super().__init__(speed)

		self.policy_net = DQN(height, width).to(device)
		self.target_net = DQN(height, width).to(device)
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
		action = select_epsilon(self.policy_net, state.unsqueeze(0), 0.05) #!
		self.actions[action]()

	def feedback(self, state, action, reward, new_state):
		reward = 101 if reward == 1 else reward
		reward -= 1
		reward *= 1000
		self.memory.push(state, action, reward, new_state)
		self.optimizer(gamma = 0.999)



target_net_update = 5
ghosts = [RandomAgent(0.05) for _ in range(3)]
pacman = DeepAgent(1)

score_history = []
for episode in range(100):
	print(f"episode {episode}")
	env = GameController(pacman, ghosts)
	while not env.terminal():
		next(env).update()
	if episode % target_net_update == 0:
		pacman.target_net.copy_from(pacman.policy_net)
	pacman.optimizer.plot_loss_variance()
	score_history.append(env.score)
	quickplot(score_history, "Score", "./score")








