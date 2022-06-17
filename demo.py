from pacman import GameController, RandomAgent


pacman = RandomAgent(0.05)
ghosts = [RandomAgent(0.025) for _ in range(3)]
GameController(pacman, ghosts, file = "pacman/default.map").run()