from pacman import GameController, RandomAgent, ChaserAgent


pacman = RandomAgent(0.05)
ghost_spd = .025
ghosts = [RandomAgent(ghost_spd) for _ in range(2)] + [ChaserAgent(ghost_spd)]
GameController(pacman, ghosts, file = "pacman/default.map").run()