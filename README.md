# Deepmac
Deep Q-learning pacman agent


## TODO

### Missing Features
- Agent movements should be smooth and stable (direction can't be changed every frame)

### Performance Improvements
- Identify and optimize bottlenecks
- Shorten game duration (with hard cap, soft tolerance cap or with more agressive ghost agents)
- Run networks on cuda?
- Multithreading?

### Parameter Testing
- Network architectures
- Reward system (passivity should be discouraged)
- Input formats (perhaps pacman should not be 0 after all D:)
- Gamma (reward discounts)

- Training batch and replay memory size
- Epsilon (or episolon scheduler)
- Optimizers
- Loss criteria (MSE, Hubert, Entropy etc.)
- Target network update frequency