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