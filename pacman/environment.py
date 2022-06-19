from pacman import settings
from numpy.random import randint
from numpy import array, inf
from math import  floor, ceil
from itertools import product
from tqdm.auto import tqdm
from typing import Dict, Tuple
import torch
from torch import tensor
import pygame

import sys, pickle



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
		self.position = None
		self.walls = None
		self.recent_action = randint(0, 2)
		self.actions = (self.left, self.right, self.up, self.down)


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
		self.recent_action = 0
		return self.move(Agent.L)


	def right(self):
		self.recent_action = 1
		return self.move(Agent.R)


	def up(self):
		self.recent_action = 2
		return self.move(Agent.U)


	def down(self):
		self.recent_action = 3
		return self.move(Agent.D)

	
	def perform_action(self, action : int):
		self.actions[action]()


	def feedback(self, state, action, reward, next_state):
		"""
		Transition feedback sent from the environment after performing action through __call__().
		"""


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

	def __call__(self, state):
		if not self.actions[self.recent_action]():
			self.recent_action = randint(0, 4)



class ControllableAgent(Agent):
	""""Agent that can be controlled with keyboard."""

	def __init__(self, speed=.1):
		super().__init__(speed)

	def __call__(self, state):
		keys = pygame.key.get_pressed()
		if keys[pygame.K_UP]:
			self.up()
		if keys[pygame.K_DOWN]:
			self.down()
		if keys[pygame.K_LEFT]:
			self.left()
		if keys[pygame.K_RIGHT]:
			self.right()



class ChaserAgent(Agent):
	"""Designated for ghost. Constantly chasing Pac-Man."""

	def __init__(self, speed=.1):
		super().__init__(speed)

	def __call__(self, state):
		my_coords = self.coords()
		my_y, my_x = my_coords
		my_coords = (my_x, my_y)
  
		distances = state[3]
		dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
  
		def get_value_ignore_neg_inf(i):
			v = distances[(my_y + dirs[i][1]) % distances.shape[0]][(my_x + dirs[i][0]) % distances.shape[1]]
			return v if v >= 0 else inf
  
		dir_idx = min(range(len(dirs)), key=lambda i: get_value_ignore_neg_inf(i))
		self.actions[dir_idx]()



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
	default_ghosts = [RandomAgent(0.05) for _ in range(3)]
	hostile_ghosts = [RandomAgent(0.1) for _ in range(2)] + [ChaserAgent(0.1)]

	def __init__(self, pacman = RandomAgent(), ghosts = default_ghosts, file=f"{path}/default.map"):
		"""
		Load the game world from a map file and spawn the agents.
		Note that the single digits in the world map file correspond to the ghost agents passed in the argument.
		"""
		world 				= open(file).read().split('\n')
		width, height 		= len(world[0]), len(world)
		self.shape 			= array((width, height))
		self.score 			= self.victory_score = self.time = 0
		self.agents 		= [pacman] + ghosts
		self.pacman 		= pacman
		self.pacman_alive 	= True
		self.tiles 			= []
		self.channels 		= torch.zeros(4, height, width, dtype = int)
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
		old_state = self.channels.float() # copying cast
		ghosts = torch.zeros_like(self.ghosts)
		for i, agent in enumerate(self.agents):
			agent(old_state)
			pos = agent.coords()
			ghosts[pos] = i

		pos = self.pacman.coords()
		self.pacman_alive &= not self.ghosts[pos]
		self.score += self.coins[pos]

		# compute reward
		reward = self.coins[pos]
		if not self.pacman_alive:
			reward -= self.victory_score
		elif self.score == self.victory_score:
			reward += self.victory_score

		self.coins[pos] = 0
		self.ghosts = ghosts
		self.time += 1
		self.distances[...] = self.dist_matrix(self.pacman.coords())

		# send transition feedback to all agents
		for agent in self.agents:
			sentiment = 2*(agent == self.pacman) - 1 
			agent.feedback(
				old_state,
				tensor(agent.recent_action, dtype=torch.float),
				(sentiment * reward).float(),
				self.channels.float()
			)

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
		distances = torch.ones(self.distances.shape, dtype=int) * Environment.Unreachable
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
		if not self.pacman_alive:
			return -1
		return 0