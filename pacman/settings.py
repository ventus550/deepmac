import pyautogui

# pixel size of game block
blksz = int(round(pyautogui.size().height / 28 * .8))

# gameplay frame duration in seconds
frame_time = 0.01

# display distance matrix in real time
distances = False

# cache distance computations into a file
cache = True

# colors
background = (46, 52, 64)
pacman = (255, 255, 120)
paths = (76, 86, 106)
ghosts = [
	(90,  194, 255),
	(255, 125,   0),
	(255, 0,   155)
]