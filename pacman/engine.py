from numpy import array, float64, zeros, zeros_like, inf, ones
from numpy.random import choice
from math import  floor, ceil
from itertools import product
from pacman import settings
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import sys
import pickle



class Engine:
	"""
	Stores and manages the game environment and all its agents.
	The game world is loaded from the file as a plain text during the objects initialization.
	Game objects are then discretized and split into separate layers ( walls, coins, ghosts, distances(NYI) )
	and can be accessed individually or as a tensor ( channels ).

	Attributes:
		score		-- the amount of collected coins ( win condition )

		agents		-- list of all the agents ( pacman is always placed at index zero )

		shape		-- game world's width and height measured in game blocks

		pacman		-- pacman agent reference ( equivalent to agents[0] )
	"""

	Empty = ' '
	Wall = '#'
	Coin = '.'
	Pacman = '@'
	Ghost = int
	UnreachableDist = -1


	def __init__(self, pacman, ghosts : list, file="pacman/default.map"):
		"""
		Load the game world from a map file and spawn the agents.
		Note that the single digits in the world map file correspond to the ghost agents passed in the argument.
		"""
		world = open(file).read().split('\n')
		width, height = len(world[0]), len(world)
		self.shape = array((width, height))
		self.score = self.victory_score = self.time = 0
		self.agents = [pacman] + ghosts
		self.pacman = pacman
		self.channels = zeros(shape = (4, height, width), dtype = int)
		self.walls, self.coins, self.ghosts, self.distances = self.channels

		for x, y in product(range(width), range(height)):
			cell = world[y][x]
			if cell == Engine.Wall:
				self.walls[y, x] = 1

			elif cell == Engine.Coin:
				self.coins[y, x] = 1
				self.victory_score += 1

			elif cell == Engine.Pacman:
				pacman = self.agents[0]
				pacman.position, pacman.walls = array((x, y), dtype=float64), self.walls

			elif cell.isnumeric():
				assert 0 < int(cell) < len(self.agents), "Invalid ghost signature!"
				gid = Engine.Ghost(cell)
				self.ghosts[y][x] = gid
				ghost = self.agents[gid]
				ghost.color = settings.ghosts[(gid - 1) % len(settings.ghosts)]
				ghost.position, ghost.walls = array((x, y), dtype=float64), self.walls

		# Stores distance between every pair of fields. Usage: `self.dist[(from_x, from_y, to_x, to_y)]`
		self.dist: Dict[Tuple[int, int, int, int], int] = {}
		pickle_path = file.rsplit("/", 1)[-1] + ".p"
		try:
			self.dist = pickle.load(open(pickle_path, "rb"))
		except FileNotFoundError:
			self._create_dist_dict(width, height, world)
			try:
				pickle.dump(self.dist, open(pickle_path, "wb"))
			except pickle.PickleError as err:
				print(err, file=sys.stderr)

		self.no_wall_coords = []
		for x, y in product(range(width), range(height)):
			if world[y][x] != Engine.Wall:
				self.no_wall_coords.append((x, y))


	def __next__(self):
		"Transition to the next game state."
		ghosts = zeros_like(self.ghosts)
		for i, agent in enumerate(self.agents):
			agent(self.channels)
			pos = agent.coords()
			ghosts[pos] = i

		pos = self.pacman.coords()
		self.score += self.coins[pos]
		self.coins[pos] = 0
		self.pacman.alive &= not self.ghosts[pos]

		self.ghosts = ghosts
		self.time += 1

		self.distances = self.dist_matrix(self.distances.shape, self.pacman.coords())
		return self


	def __getitem__(self, coords):
		"Access the the object at the given (x, y) map coordinates."
		x, y = coords
		if self.walls[y][x] == 1:
			return Engine.Wall
		elif self.coins[y][x] == 1:
			return Engine.Coin
		elif self.ghosts[y][x]:
			return self.ghosts[y][x]
		elif self.pacman.coords() == (y, x):
			return Engine.Pacman
		return Engine.Empty


	def __repr__(self):
		string = []
		width, height = self.walls.shape
		for y in range(height):
			string.append("".join(str(self[x, y]) for x in range(width)))
		return "\n".join(string)


	def _create_dist_dict(self, width, height, world: List[str]):
		for from_x, from_y in product(range(width), range(height)):
			for to_x, to_y in product(range(width), range(height)):
				if from_x == to_x and from_y == to_y:
					self.dist[(from_x, from_y, to_x, to_y)] = 0
				# Walls are unreachable.
				elif world[from_y][from_x] == Engine.Wall or world[to_y][to_x] == Engine.Wall:
					self.dist[(from_x, from_y, to_x, to_y)] = inf
				# Difference by one vertically/horizontally.
				elif (abs(from_x - to_x) == 1 and abs(from_y - to_y) == 0) or (abs(from_x - to_x) == 0 and abs(from_y - to_y) == 1):
					self.dist[(from_x, from_y, to_x, to_y)] = 1
				else:
					self.dist[(from_x, from_y, to_x, to_y)] = inf
		# Special case - corresponding fields near the opposite edges that allow warping.
		for x in range(width):
			if world[0][x] != Engine.Wall and world[height - 1][x] != Engine.Wall:
				self.dist[(x, 0, x, height - 1)] = 1
				self.dist[(x, height - 1, x, 0)] = 1
		for y in range(height):
			if world[y][0] != Engine.Wall and world[y][width - 1] != Engine.Wall:
				self.dist[(0, y, width - 1, y)] = 1
				self.dist[(width - 1, y, 0, y)] = 1
		# Compute distance between each pair of board fields using Floyd-Warshall algorithm.
		coordinates = []
		for x, y in product(range(width), range(height)):
			# Do not consider walls in the algorithm.
			if world[y][x] != Engine.Wall:
				coordinates.append((x, y))
		# Floyd-Warshall
		for mid_x, mid_y in tqdm(coordinates):
			for from_x, from_y in coordinates:
				for to_x, to_y in coordinates:
					dist_through_mid = self.dist[(from_x, from_y, mid_x, mid_y)] + self.dist[(mid_x, mid_y, to_x, to_y)]
					if self.dist[(from_x, from_y, to_x, to_y)] > dist_through_mid:
						self.dist[(from_x, from_y, to_x, to_y)] = dist_through_mid
		# Switch inf values to NN-safe ones.
		for k, v in self.dist.items():
			if v == inf:
				self.dist[k] = Engine.UnreachableDist


	def terminal(self):
		"Test if the game has run its course and return the result."
		if self.score == self.victory_score:
			return 1
		elif not self.pacman.alive:
			return -1
		return 0


	def dist_matrix(self, shape: Tuple[int, int], field: Tuple[int, int]):
		"""Return matrix of distances relative to the given field.

		Args:
			shape (Tuple[int, int]): Shape of the board (height, width).
			field (Tuple[int, int]): Coordinates of the field (y, x).
		"""
		distances = ones(shape, dtype=int) * Engine.UnreachableDist
		from_y, from_x = field
		for x, y in self.no_wall_coords:
			distances[y][x] = self.dist[(from_x, from_y, x, y)]
		return distances



class Agent:
	"""
	Entity capable of interacting with the game engine.
	Use this template to create various game agents through subclassing.
	Note that __call__( ) method is virtual and has to be implemented ( see RandomAgent example ).

	Attributes:
		speed		-- the speed with which the agent traverses the environment

		position	-- real agent position ( for discrete representation use Agent.coords( ) instead )
	"""

	L = array((-1, 0))
	R = array(( 1, 0))
	U = array(( 0,-1))
	D = array(( 0, 1))

	def __init__(self, speed = 0.1):
		self.speed = speed
		self.alive = True
		self.position = None
		self.walls = None
		self.facing = choice((self.left, self.right))


	def coords(self):
		"Returns discretized map coordinates."
		x, y = self.position.round().astype(int)
		return y, x


	def move(self, direction):
		"Attempt to move in the target direction and return the result."
		width, height = len(self.walls), len(self.walls[0])
		x, y = self.position + direction * self.speed
		rnd = (ceil, floor)

		if all(self.walls[p(y) % height, q(x) % width] == 0 for p, q in product(rnd, rnd)):
			self.position += direction * self.speed
			self.position %= (width, height)
			return True
		self.position = self.position.round()
		return False


	def left(self):
		self.facing = self.left
		return self.move(Agent.L)


	def right(self):
		self.facing = self.right
		return self.move(Agent.R)


	def up(self):
		self.facing = self.up
		return self.move(Agent.U)


	def down(self):
		self.facing = self.down
		return self.move(Agent.D)


	def __call__(self, state):
		"""
		Perform an action in response to the current game state.
		This method is called once every turn.
		"""
		raise NotImplementedError



class RandomAgent(Agent):
	"Unintelligent and uninteresting, yet pretty useful for testing purposes :)"

	def __init__(self, speed=0.1):
		super().__init__(speed)
		self.directions = (self.left, self.right, self.up, self.down)


	def __call__(self, state):
		if not self.facing():
			self.facing = choice(self.directions)
