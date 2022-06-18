import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from QLearning.utilities import get_device


class QNetwork(nn.Module):
    """
    Generic QNetwork implementing some essential features.
    Use this template to create various Q-learning architectures through subclassing.
    """
    def __init__(self):
        super().__init__()
    
    def save(self, path):
        "Store model in a file."
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        "Load model from a file."
        self.load_state_dict(torch.load(path))

    def copy_from(self, qnet):
        """
        Make this network an exact copy
        of another network with the same architecture.
        """
        self.load_state_dict(qnet.state_dict())
    
    def clone(self):
        "Make a clone of the model"
        return deepcopy(self)


class DQN(QNetwork):
    "Simple and probably terrible triple layered qnetwork."
    def __init__(self, h, w):
        super().__init__()
        actions = 4
        self.device = get_device()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = (conv2d_size_out(conv2d_size_out(w)))
        convh = (conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, actions)
        self.float()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.head(torch.flatten(x, 1))

    