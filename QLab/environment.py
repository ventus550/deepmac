import gym
import numpy as np
import torch

color = np.array([210, 164, 74]).mean()
def preprocess_observation(obs):
    # Crop and resize the image
    img = obs[1:176:2, ::2]
    # Convert the image to greyscale
    img = img.mean(axis=2)
    # Improve image contrast
    img[img==color] = 0
    # Next we normalize the image from -1 to +1
    img = (img - 128) / 128 - 1
    return img.reshape(88,80,1).transpose((2,0,1))

class Environment:
	def __init__(self):
		self.env = gym.make('MsPacman-v0')
		self.state = None
		self.reset()

	def get_screen(self):
		obs = self.env.render(mode='rgb_array')
		return torch.tensor(preprocess_observation(obs), dtype=torch.float32)

	def step(self, action):
		_, reward, done, _ = self.env.step(action)
		reward = torch.tensor(reward)
		self.state, prev_state = None, self.state
		if not done:
			self.state = self.get_screen()
		return prev_state, reward, self.state, done

	def reset(self):
		self.env.reset()
		# last_screen = self.get_screen()
		# current_screen = self.get_screen()
		# self.state = current_screen - last_screen
		self.state = self.get_screen()
		
	def render(self):
		self.env.render()

	def close(self):
		self.env.close()