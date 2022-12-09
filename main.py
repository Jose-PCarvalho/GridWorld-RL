import itertools
import random
import sys
import time
from collections import defaultdict
from environment import *
import pickle
from algorithms import *


env = GridWorld()
#Q = q_learning(env, 10000)
Q= sarsa(env,10000)
with open('new_obs.pkl', 'wb') as file:
    pickle.dump(Q, file, pickle.HIGHEST_PROTOCOL)

env = GridWorld(config={"render": "human"})
num_episodes = 10000
for i_episode in range(num_episodes):
    # Print out which episode we're on, useful for debugging.
    if (i_episode + 1) % 1 == 0:
        print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
        sys.stdout.flush()

    # Reset the environment and pick the first action
    state = env.reset()

    # One step in the environment
    # total_reward = 0.0
    for t in itertools.count():
        # Take a step
        action = np.argmax(Q[state])
        next_state, reward, done, _ = env.step(action)
        if done:
            break
        state = next_state
