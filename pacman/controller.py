import pygame
from pygame.locals import *
from pygame.gfxdraw import pie
from pacman.environment import Environment
from pacman import settings
from numpy import array
from itertools import product
from time import sleep


class GameController(Environment):
	"""
	Runs the app version of the game.
	Provides a reliable way of supervising the agents' actions. 
	"""
	def __init__(self, pacman, ghosts : list, file = f"{Environment.path}/default.map"):
		super().__init__(pacman, ghosts, file)
		pygame.init()
		screensize = self.shape * settings.blksz
		self.screen = pygame.display.set_mode(screensize, 0, 32)
		self.background = pygame.surface.Surface(screensize).convert()
		self.background.fill(settings.paths)
		self.font = pygame.font.SysFont("monospace", settings.blksz // 4)


	def update(self):
		next(self)
		self.checkEvents()
		self.render()
		return self


	def checkEvents(self):
		for event in pygame.event.get():
			if event.type == QUIT:
				pygame.quit()
	
	def close(self):
		pygame.quit()


	def draw_wall(self, pos, fill = False, size = settings.blksz):
		x, y = pos * settings.blksz
		pygame.draw.rect(self.screen, settings.background, pygame.Rect(x, y, size, size), width = 3*(1 - fill))


	def draw_pacman(self, pos):
		radius = settings.blksz // 2
		x, y = pos * settings.blksz
		pacman = self.agents[0]
		ang = abs(self.time * 4 % 45 - 22) * 2 + 1
		rot = {1 : 0, 3 : 90, 0 : 180, 2 : 270}[pacman.recent_action]
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
	

	def text(self, x, y, text, color):
		text = self.font.render(text, True, color)
		self.screen.blit(text, (x, y))


	def render(self):
		width, height = self.shape
		self.screen.blit(self.background, (0,0))
		d, blksz = self.distances, settings.blksz
		for x, y in product(range(width), range(height)):
			pos = array((x, y)); cell = self[pos]
			self.draw_wall(pos, fill = cell == Environment.Wall)
			if settings.distances and d[y, x] != Environment.Unreachable:
				self.text(x*blksz, y*blksz, str(d[y, x]), settings.pacman)
			if self.coins[y, x] == 1:
				self.draw_coin(pos)
		self.draw_pacman(self.agents[0].position)
		for ghost in self.agents[1:]:
			self.draw_ghost(ghost.position, ghost.color)
		self.text(20, 10, f'Game time {self.time} Score {self.score}', settings.pacman)
		pygame.display.update()

	
	def run(self):
		try:
			while not next(self).terminal():
				sleep(settings.frame_time)
				self.update()
		except pygame.error: ...