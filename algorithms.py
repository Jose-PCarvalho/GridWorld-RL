from environment import *
from collections import defaultdict
import itertools
import sys
import datetime, time
import os
import pickle
from pathlib import Path


def create_Q():
    return np.zeros(5)


def q_learning(env, num_episodes, discount_factor=1.0):
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(create_Q)
    N0 = 30
    NSA = defaultdict(lambda: np.zeros(env.action_space.n))
    NS = defaultdict(lambda: np.zeros(1))
    alpha = lambda state, action: 1 / NSA[state][action]
    epsilon = lambda state: N0 / (N0 + NS[state])
    actions = [0, 1, 2, 3, 4]

    def epsilonGreedy(state, info):
        eps = max(0.01, epsilon(state))

        if np.random.random() < eps:
            # exploration
            a = []
            if info[0] != 0:
                a.append(0)
            if info[1] != 0:
                a.append(1)
            if info[2] != 0:
                a.append(2)
            if info[3] != 0:
                a.append(3)
            if not a:
                a = [4]
            #print(a)
            action = np.random.choice(a)

        else:
            # exploitation
            action = np.argmax(Q[state])

        return action

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state, info = env.reset()

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():

            # Take a step

            action = epsilonGreedy(state, info)
            NS[state] += 1
            NSA[state][action] += 1
            next_state, reward, done, info = env.step(action)
            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha(state, action) * td_delta

            if done:
                break
            state = next_state

    return Q


def sarsa(env, num_episodes, discount_factor=1.0):
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(create_Q)
    N0 = 5
    NSA = defaultdict(lambda: np.zeros(env.action_space.n))
    NS = defaultdict(lambda: np.zeros(1))
    alpha = lambda state, action: 1 / NSA[state][action]
    epsilon = lambda state: N0 / (N0 + NS[state])
    actions = [0, 1, 2, 3]

    def epsilonGreedy(state, info):
        eps = max(0.01, epsilon(state))

        if np.random.random() < eps:
            # exploration
            a = []
            if info[0] != 0:
                a.append(0)
            if info[1] != 0:
                a.append(2)
            if info[2] != 0:
                a.append(3)
            if info[3] != 0:
                a.append(1)
            if not a:
                a = actions
            action = np.random.choice(a)

        else:
            # exploitation
            action = np.argmax(Q[state])

        return action

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state, info = env.reset()

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():

            # Take a step
            action = epsilonGreedy(state, info)
            NS[state] += 1
            NSA[state][action] += 1
            next_state, reward, done, info = env.step(action)
            # TD Update
            next_action = epsilonGreedy(state, info)
            td_target = reward + discount_factor * Q[next_state][next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha(state, action) * td_error

            if done:
                break
            state = next_state

    return Q
