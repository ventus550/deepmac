from random import choices
from collections import deque
from QLearning.utilities import Transition

class ReplayMemory:
    """
    Memory snippets recorder.
    Allows storing and sampling memorized transitions.
    When its maximum capacity is reached the oldest transitions
    are discarded in favor of the new ones. 
    """
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, reward, new_state):
        "Save a transition."
        if new_state is not None:
            new_state = new_state.unsqueeze(0)
        self.memory.append(
            Transition(
                state.unsqueeze(0),
                action.unsqueeze(0),
                reward.unsqueeze(0),
                new_state
            )
        )

    def sample(self, batch_size):
        "Sample a batch of trasitions."
        return choices(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)