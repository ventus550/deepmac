from pacman import GameController, RandomAgent


pacman = RandomAgent()
ghosts = [RandomAgent(0.05) for _ in range(3)]
GameController(pacman, ghosts, board = "pacman/default.map")