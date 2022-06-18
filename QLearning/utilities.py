import torch
from random import random, randint
from collections import namedtuple
import matplotlib.pyplot as plt
from seaborn import set_theme
set_theme(style = "darkgrid", palette="dark")

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


def get_device():
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_action(policy_net, state):
	"Select action in accordance with the policy."
	with torch.no_grad():
		# t.max(1) will return largest column value of each row.
		# second column on max result is index of where max element was
		# found, so we pick action with the larger expected reward.
		return policy_net(state).max(1)[1].view(1, 1)


def select_epsilon(policy_net, state, epsilon = 0.5):
	"Follow the policy with (1 - ε) chance or perform a random action instead."
	roll = random()
	if roll > epsilon:
		return select_action(policy_net, state)
	return randint(0, 3)


def quickplot(values, values_title = "quickplot", path = "./quickplot"):
	plt.plot(values)
	plt.ylabel(values_title)
	plt.savefig(path)
	plt.clf()
