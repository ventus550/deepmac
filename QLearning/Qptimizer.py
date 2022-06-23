import torch
from torch.nn import SmoothL1Loss
from QLearning.utilities import Transition, quickplot

class Qptimizer:
	"""
	Generic double Qnetwork optimizer class.

	Args:
		memory		-- replay memory from which training data will be sampled

		optimizer	-- optimization algorithm

		policy_net	-- dynamically updated Qnetwork

		target_net	-- updates every few episodes in order to stablizie learning

		criterion	-- applied to residuals to compute loss

		batchsz		-- size of each training batch
	"""
	def __init__(self, memory, optimizer, policy_net, target_net, criterion = SmoothL1Loss(), batchsz = 128):
		self.memory = memory
		self.optimizer = optimizer
		self.policy_net = policy_net
		self.target_net = target_net
		self.criterion = criterion
		self.batchsz = batchsz
		self.loss_variance_history = [0]
		self.loss_history = [0]
		self.variance = 0

	def plot_loss_variance(self, path = './variance'):
		quickplot(self.loss_variance_history, "Loss Variance", path = path)
	
	def plot_loss(self, path = './loss'):
		quickplot(self.loss_history, "Loss", path = path)
	
	def __call__(self, gamma = 0.5):
		"""Perform single optimization step with gamma discount."""
		if len(self.memory) < self.batchsz:
			return
		transitions = self.memory.sample(self.batchsz)

		# Convert batch of transitions into transition of batches
		batch = Transition(*zip(*transitions))
		state_batch = torch.cat(batch.state)
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)
		action_batch = action_batch.unsqueeze(1).type(torch.int64)

		# Compute V(s_{t+1}) for all next states.
		# Expected values of actions for next_batch are computed based
		# on the "older" target_net; selecting their best reward with max(1)[0].
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
		non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
		Q_target = torch.zeros(len(state_batch))
		Q_target[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
		Q_expected = (Q_target * gamma) + reward_batch

		# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
		# columns of actions taken. These are the actions which would've been taken
		# for each batch state according to policy_net
		Q_policy = self.policy_net(state_batch).gather(1, action_batch)

		# Compute loss diagnostics using rolling variance
		loss = self.criterion(Q_policy, Q_expected.unsqueeze(1))
		self.loss_history.append(loss.item())
		self.variance += (loss.item()**2 - self.variance) / len(self.loss_variance_history)
		self.loss_variance_history.append(self.variance)


		# Optimize the model
		self.optimizer.zero_grad()
		loss.backward()
		for param in self.policy_net.parameters():
			# clip the gradient to [-1, 1] range
			param.grad.data.clamp_(-1, 1)
		self.optimizer.step()