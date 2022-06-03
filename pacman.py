from numpy import array, zeros, ndarray, zeros_like
from itertools import product
from random import choice


board = (
	"............##............",
	".####.#####.##.####.#####.",
	".####.#####.##.####.#####.",
	".####.#####.##.####.#####.",
	"..........................",
	".####.##.########.##.####.",
	"......##....##....##......",
	"#####.##### ## #####.#####",
	"#####.##    12    ##.#####",
	"#####.## ###34### ##.#####",
	"#####.## ######## ##.#####",
	"#####.## ######## ##.#####",
	"#####.## ######## ##.#####",
	"#####.## ######## ##.#####",
	"#####.##          ##.#####",
	"#####.## ######## ##.#####",
	"............##............",
	".####.#####.##.####.#####.",
	".####.#####.##.####.#####.",
	"...##.......@........##...",
	"##.##.##.########.##.##.##",
	"##.##.##.########.##.##.##",
	"......##....##....##......",
	".##########.##.##########.",
	".##########.##.##########.",
	".........................."
)


def round(array):
	return array.astype(int)


class Agent:
	L = (-1, 0)
	R = ( 1, 0)
	U = ( 0,-1)
	D = ( 0, 1)

	def __init__(self, threat = 0):
		self.threat = threat
		self.speed = 0.5
		self.pacman = False
		self.position = None
		self.walls = None
		


	def coords(self):
		x, y = round(self.position)
		return y, x


	def move(self, direction):
		width, height = len(self.walls), len(self.walls[0])
		x, y = round(self.position + direction)
		x, y = x % width, y % height
		if self.walls[y, x] == 0:
			self.position += direction


	def left(self):
		self.move(Agent.L)


	def right(self):
		self.move(Agent.R)


	def up(self):
		self.move(Agent.U)


	def down(self):
		self.move(Agent.D)


	def __call__(self, state):
		raise NotImplementedError


class RandomAgent(Agent):
	def __call__(self, state):
		choice((self.left, self.right, self.up, self.down))()


class Pacman:
	Empty = ' '
	Wall = '#'
	Coin = '.'
	Pacman = '@'
	Ghost = int


	def __init__(self, pacman : Agent, ghosts : list, board=board):
		width, height = len(board[0]), len(board)
		self.agents = [pacman] + ghosts
		self.channels = zeros(shape = (4, height, width), dtype = int)
		self.walls, self.coins, self.ghosts, self.distance = self.channels
		
		for x, y in product(range(width), range(height)):
			cell = board[y][x]
			if cell == Pacman.Wall:
				self.walls[y, x] = 1

			elif cell == Pacman.Coin:
				self.coins[y, x] = 1

			elif cell == Pacman.Pacman:
				pacman = self.agents[0]
				pacman.pacman = True
				pacman.position, pacman.walls = array((x, y)), self.walls

			elif cell.isnumeric():
				assert 0 < int(cell) < len(self.agents), "Invalid ghost signature!"
				self.ghosts[y][x] = Pacman.Ghost(cell)
				ghost = self.agents[int(cell)]
				ghost.position, ghost.walls = array((x, y)), self.walls


	def __next__(self):
		ghosts = zeros_like(self.ghosts)
		for i, agent in enumerate(self.agents):
			agent(self.channels)
			pos = agent.coords()
			ghosts[pos] = i

			if i == 0:
				self.coins[pos] = 0
				if self.ghosts[pos]:
					exit("Game Over")
		self.ghosts = ghosts


	def __getitem__(self, key):
		x, y = key
		if self.walls[y][x] == 1:
			return Pacman.Wall
		elif self.coins[y][x] == 1:
			return Pacman.Coin
		elif self.ghosts[y][x]:
			return self.ghosts[y][x]
		elif self.agents[0].coords() == (y, x):
			return Pacman.Pacman
		return Pacman.Empty
	

	def __repr__(self):
		string = []
		width, height = self.walls.shape
		for y in range(height):
			string.append("".join(str(self[x, y]) for x in range(width)))
		return "\n".join(string)



p = Pacman(RandomAgent(), [RandomAgent() for _ in range(4)])
print(p)
print()
next(p)

print(p)
print()
next(p)

print(p)
print()
next(p)

print(p)
print()
next(p)