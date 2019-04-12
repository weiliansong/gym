import time
import gym
from gym import logger, wrappers
from skimage.io import imsave

img_root = '/media/Big_Stuff/Data/acrobot/images/'

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

env = gym.make('Acrobot-v1')
outdir = '/tmp/random-agent-results'
env = wrappers.Monitor(env, directory=outdir, video_callable=False, force=True)
env.seed(0)
agent = RandomAgent(env.action_space)
env.reset()

idx = 0
with open('full.csv', 'w') as f:
  f.write('idx,x,y\n')

  while True:
    env.reset()
    while True:
      action = agent.act(None, None, None)
      _, _, done, _ = env.step(action)
      img, obj_loc = env.render(mode='rgb_array')
      imsave(img_root + '%d.png' % idx, img, check_contrast=False)
      f.write('%d,%f,%f\n' % (idx, obj_loc[0], obj_loc[1]))

      idx += 1

      # Stop when we saved enough
      if idx == 1000:
        env.close()
        exit(0)

      if done:
        break
