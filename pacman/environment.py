from pacman import settings
from numpy.random import choice
from numpy import array, zeros, zeros_like, inf, ones
from math import  floor, ceil
from itertools import product
from tqdm.auto import tqdm
from typing import Dict, Tuple
import sys, pickle



class Environment:
	"""
	Stores and manages the game environment and all its agents.
	The game world is loaded from the file as a plain text during the objects initialization.
	Game objects are then discretized and split into separate layers ( walls, coins, ghosts, distances )
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
	Unreachable = -1
	path = "/".join(__loader__.path.split('/')[:-1])

	def __init__(self, pacman, ghosts : list, file=f"{path}/default.map"):
		"""
		Load the game world from a map file and spawn the agents.
		Note that the single digits in the world map file correspond to the ghost agents passed in the argument.
		"""
		world 			= open(file).read().split('\n')
		width, height 	= len(world[0]), len(world)
		self.shape 		= array((width, height))
		self.score 		= self.victory_score = self.time = 0
		self.agents 	= [pacman] + ghosts
		self.pacman 	= pacman
		self.tiles 		= []
		self.channels 	= zeros(shape = (4, height, width), dtype = int)
		self.walls, self.coins, self.ghosts, self.distances = self.channels

		for x, y in self.coords():
			cell = world[y][x]
			if cell != Environment.Wall:
				self.tiles.append((x, y))

			if cell == Environment.Wall:
				self.walls[y, x] = 1

			elif cell == Environment.Coin:
				self.coins[y, x] = 1
				self.victory_score += 1

			elif cell == Environment.Pacman:
				pacman = self.agents[0]
				pacman.position, pacman.walls = array((x, y), dtype=float), self.walls

			elif cell.isnumeric():
				assert 0 < int(cell) < len(self.agents), "Invalid ghost signature!"
				gid = Environment.Ghost(cell)
				self.ghosts[y][x] = gid
				ghost = self.agents[gid]
				ghost.color = settings.ghosts[(gid - 1) % len(settings.ghosts)]
				ghost.position, ghost.walls = array((x, y), dtype=float), self.walls

		# Stores distance between every pair of fields. Usage: "self.dist[(from_x, from_y, to_x, to_y)]"
		self.dist: Dict[Tuple[int, int, int, int], int] = {}
		if settings.cache:
			self._cache(file)
		else:
			self._create_dist_dict(width, height)


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

		self.distances = self.dist_matrix(self.pacman.coords())
		return self


	def __getitem__(self, coords):
		"Access the the object at the given (x, y) map coordinates."
		x, y = coords
		if self.walls[y, x] == 1:
			return Environment.Wall
		elif self.pacman.coords() == (y, x):
			return Environment.Pacman
		elif self.ghosts[y, x]:
			return self.ghosts[y, x]
		elif self.coins[y, x] == 1:
			return Environment.Coin
		return Environment.Empty


	def __repr__(self):
		string = []
		width, height = self.shape
		for y in range(height):
			string.append("".join(str(self[x, y]) for x in range(width)))
		return "\n".join(string)


	def _create_dist_dict(self, width, height):
		for source, destination in product(self.coords(), self.coords()):
			(sx, sy), (dx, dy) = source, destination
			self.dist[(sx, sy, dx, dy)] = inf
			if source == destination:
				self.dist[(sx, sy, dx, dy)] = 0
			# Walls are unreachable
			elif self[source] == Environment.Wall or self[destination] == Environment.Wall:
				self.dist[(sx, sy, dx, dy)] = inf
			# Difference by one vertically/horizontally
			elif abs(sx - dx) % (width - 2) + abs(sy - dy) % (height - 2) == 1:
				self.dist[(sx, sy, dx, dy)] = 1

		# Floyd-Warshall
		self._fw(self.tiles)
		
		# Switch inf values to NN-safe ones
		for k, v in self.dist.items():
			self.dist[k] = Environment.Unreachable if v == inf else v
	

	def _fw(self, vertices):
		"Compute distance between each pair of board fields using Floyd-Warshall algorithm."
		for mid_x, mid_y in tqdm(vertices):
			for (from_x, from_y), (to_x, to_y) in product(vertices, vertices):
				dist_through_mid = self.dist[(from_x, from_y, mid_x, mid_y)] + self.dist[(mid_x, mid_y, to_x, to_y)]
				if self.dist[(from_x, from_y, to_x, to_y)] > dist_through_mid:
					self.dist[(from_x, from_y, to_x, to_y)] = dist_through_mid


	def _cache(self, file):
		width, height = self.shape
		pickle_path = file.rsplit("/", 1)[-1] + ".p"
		try:
			self.dist = pickle.load(open(pickle_path, "rb"))
		except FileNotFoundError:
			self._create_dist_dict(width, height)
			try:
				pickle.dump(self.dist, open(pickle_path, "wb"))
			except pickle.PickleError as err:
				print(err, file=sys.stderr)


	def dist_matrix(self, field: Tuple[int, int]):
		"""Return matrix of distances relative to the given field.

		Args:
			field (Tuple[int, int]): Coordinates of the field (y, x).
		"""
		distances = ones(self.distances.shape, dtype=int) * Environment.Unreachable
		from_y, from_x = field
		for x, y in self.tiles:
			distances[y][x] = self.dist[(from_x, from_y, x, y)]
		return distances


	def coords(self):
		width, height = self.shape
		return product(range(width), range(height))


	def terminal(self):
		"Test if the game has run its course and return the result."
		if self.score == self.victory_score:
			return 1
		if not self.pacman.alive:
			return -1
		return 0



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