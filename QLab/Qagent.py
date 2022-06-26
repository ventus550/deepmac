import context
import torch
from QLearning import QNetwork, Qptimizer, ReplayMemory, utilities
from tqdm import tqdm
import random
from environment import Environment
import contextlib

@contextlib.contextmanager
def ignored(*exceptions):
    try:
        yield
    except exceptions:
        pass

eps_min = 0.05
eps_decay_steps = 500000

class QAgent:
	def __init__(self, qnet : QNetwork, learning_rate = 0.01, gamma = 0.97, memory = 20000, criterion = torch.nn.MSELoss()):
		self.gamma = gamma
		self.memory = ReplayMemory(memory)

		self.policy_net = qnet
		self.target_net = qnet.clone()
		self.target_net.copy_from(self.policy_net)
		self.target_net.eval()

		# self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=learning_rate)
		self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
		self.optimizer = Qptimizer(
			self.memory,
			self.optimizer,
			self.policy_net,
			self.target_net,
			criterion = criterion
		)

	def __call__(self, state, epsilon = 0):
		sample = random.random()
		if sample > epsilon:
			with torch.no_grad():
				return self.policy_net(state.unsqueeze(0)).max(1)[1][0]
		return torch.tensor(random.randrange(9), dtype=torch.long)

	def save(self, path = "./net"):
		self.policy_net.save(path)

	def load(self, path = "./net"):
		self.policy_net.load(path)
		self.target_net.copy_from(self.policy_net) # probably the right way to do this ...

	def train(self, episodes = 1, update_frq = 10, epsilon = 0.9, live = False, plot = False):
		rewards = []
		schedule = []
		steps_done = 0
		env = Environment()
		with ignored(KeyboardInterrupt):
			for i_episode in tqdm(range(episodes)):
				reward_acc = 0
				env.reset()
				done = False
				while not done:
					if live: env.render()
					eps = epsilon - (epsilon - eps_min) * steps_done / eps_decay_steps
					action = self(env.state, max(eps_min, eps))
					state, reward, next_state, done = env.step(action)
					reward_acc += reward
					steps_done += 1

					# Store the transition in memory
					self.memory.push(state, action, reward, next_state)

					# Perform one step of the optimization (on the policy network)
					self.optimizer(gamma = self.gamma)

				schedule.append(eps)
				rewards.append(reward_acc)
				if plot:
					utilities.quickplot(schedule, "Epsilon", path = "./epsilon")
					utilities.quickplot(rewards, "Score", path = "./score")
					self.optimizer.plot_loss()
					self.optimizer.plot_loss_variance()

				# Update the target network, copying all weights and biases in DQN
				if i_episode % update_frq == 0:
					self.target_net.copy_from(self.policy_net)
		if live: env.close()
