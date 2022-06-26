# Deepmac
Deep Q-Learning agent playing Pac-Man for the Neural Networks course.
The final model is demonstrably proven to improve over time but the training requires thousands of episodes thus being computationally far too expensive.
That being said agent easily outperforms random agents.
As of this moment the most successful approach was to run double Q-Learning with convolutional neural networks on memory-sampled transition batches while scheduling gradual decrease of epsilon (in ɛ-greedy strategy).
