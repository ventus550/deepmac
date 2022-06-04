import pygame
from pygame.locals import *
from pygame.gfxdraw import pie
from pacman.engine import Engine
from pacman import settings
from numpy import array
from itertools import product
from time import sleep



class GameController(Engine):
	"""
	Runs the app version of the game.
	Provides a reliable way of supervising the agents' actions. 
	"""
	def __init__(self, pacman, ghosts : list, board = "pacman/default.map"):
		super().__init__(pacman, ghosts, board)
		pygame.init()
		screensize = self.shape * settings.blksz
		self.screen = pygame.display.set_mode(screensize, 0, 32)
		self.background = pygame.surface.Surface(screensize).convert()
		self.background.fill(settings.paths)
		self.run()


	def update(self):
		self.checkEvents()
		self.render()


	def checkEvents(self):
		for event in pygame.event.get():
			if event.type == QUIT:
				exit()


	def draw_wall(self, pos, fill = False, size = settings.blksz):
		x, y = pos * settings.blksz
		pygame.draw.rect(self.screen, settings.background, pygame.Rect(x, y, size, size), width = 3*(1 - fill))


	def draw_pacman(self, pos):
		radius = settings.blksz // 2
		x, y = pos * settings.blksz
		pacman = self.agents[0]
		ang = abs(self.time % 45 - 22) * 2 + 1
		rot = {pacman.right : 0, pacman.down : 90, pacman.left : 180, pacman.up : 270}[pacman.facing]
		pie(self.screen, int(x) + radius, int(y) + radius, int(radius * 0.8), ang + rot, 360 - ang + rot, settings.pacman)

	
	def draw_coin(self, pos):
		radius = settings.blksz / 16
		pygame.draw.circle(self.screen, settings.pacman, pos * settings.blksz + settings.blksz/2, radius * 0.9, width=2)

	
	def draw_ghost(self, pos, color):
		blksz = settings.blksz
		x, y = pos * blksz; s = 0.8
		triangle = (
			(x + (1-s)*blksz, y + s*blksz),
			(x + s*blksz, y + s*blksz),
			(x + blksz/2, y + (1-s)*blksz)
		)
		pygame.draw.polygon(self.screen, color, triangle, 5)
	

	def text(self, surface, size, x, y, text, color):
		font = pygame.font.SysFont("monospace", size)
		text = font.render(text, 1, color)
		surface.blit(text, (x, y))


	def render(self):
		width, height = self.shape
		self.screen.blit(self.background, (0,0))
		for x, y in product(range(width), range(height)):
			pos = array((x, y)); cell = self[pos]
			self.draw_wall(pos, fill = cell == Engine.Wall)
			if cell == Engine.Coin:
				self.draw_coin(pos)
		self.draw_pacman(self.agents[0].position)
		for ghost in self.agents[1:]:
			self.draw_ghost(ghost.position, ghost.color)
		self.text(self.screen, 20, 10, 10, f'Game time {self.time} Score {self.score}', settings.pacman)
		pygame.display.update()

	
	def run(self):
		while not next(self).terminal():
			sleep(settings.frame_time)
			self.update()