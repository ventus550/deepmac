# Deepmac
Deep Q-learning pacman agent


## TODO

### Missing Features
- Some sort of test lab to make testing new agents easier and simpler
- Agent movements should be smooth and stable (direction can't be changed every frame)

### Performance Improvements
- Identify and optimize bottlenecks
- Shorten game duration (with hard cap, soft tolerance cap or with more agressive ghost agents)
- Run networks on cuda?
- Multithreading?

### Parameter Testing
- Reward system (passivity should be discouraged)
- Network architectures
- Optimizers
- Epsilon (or episolon scheduler)
- Gamma (reward discounts)
- Target network update frequency
- Loss criteria (MSE, Hubert, Entropy etc.)
- Training batch and replay memory size
- Input formats (perhaps pacman should not be 0 after all D:)