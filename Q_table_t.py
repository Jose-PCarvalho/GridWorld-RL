import itertools
import random
import sys
import time
from collections import defaultdict
from environment import *
import pickle
from algorithms import *


with open('new_obs.pkl', 'rb') as file:
    Q = pickle.load(file)
env = GridWorld(config={"render": "human"})
num_episodes = 10000
for i_episode in range(num_episodes):
    # Print out which episode we're on, useful for debugging.
    if (i_episode + 1) % 1 == 0:
        print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
        sys.stdout.flush()

    # Reset the environment and pick the first action
    state ,info = env.reset()

    # One step in the environment
    # total_reward = 0.0
    for t in itertools.count():
        # Take a step
        action = np.argmax(Q[state])
        next_state, reward, done, info = env.step(action)
        print(next_state)
        time.sleep(1)
        if done:
            break
        state = next_state
