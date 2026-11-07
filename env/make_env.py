from env.multiagent.environment import MultiAgentEnv
from env.multiagent.core import World
import random

world = World()
env = MultiAgentEnv(world)

for i in range(1000):
    pos = [[random.uniform(103.9787,104.1631),random.uniform(30.5965,30.7309)]
           for _ in range(len(env.world.agents))]
    env.step(pos)