import time

import numpy as np
from enum import Enum
import pygame
import gym
from gym import spaces
from gym.spaces import MultiDiscrete, Discrete
from collections import defaultdict
from collections import deque


class State:
    def __init__(self):
        self.state = np.array((4, 1), dtype=int)

    def update(self, obs):
        self.state = obs


class Queue:
    def __init__(self):
        self.elements = deque()

    def empty(self) -> bool:
        return not self.elements

    def put(self, x):
        self.elements.append(x)

    def get(self):
        return self.elements.popleft()


class Actions(Enum):
    UP = 0
    Down = 2
    Left = 3
    Right = 1


class GridMap:
    def __init__(self, a=None):
        if a is not None:
            self.height = a.shape[0]
            self.width = a.shape[1]
            self.map = {(0, 0): []}
            self.visited_list = [(0, 0)]
            for i in range(self.width):
                for j in range(self.height):
                    self.new_tile((i, j))
        else:
            self.height = 1
            self.width = 1
            self.map = {(0, 0): []}
            self.visited_list = [(0, 0)]

    @staticmethod
    def adjacentTiles(tile):

        return [(tile[0] + 1, tile[1]), (tile[0] - 1, tile[1]), (tile[0], tile[1] + 1), (tile[0], tile[1] - 1)]

    def getTiles(self):
        return list(self.map.keys())

    def new_tile(self, tile):
        if tile not in self.map:
            if tile[0] >= self.height:
                self.height = tile[0] + 1
            elif tile[1] >= self.width:
                self.width = tile[1] + 1

            self.map[tile] = []
            adjacent = self.adjacentTiles(tile)
            for adj in adjacent:
                if adj in self.getTiles():
                    self.map[tile].append(adj)
                    self.map[adj].append(tile)

    def print_graph(self):
        tiles = self.getTiles()
        for t in tiles:
            print(t, ": ", self.map[t])
        print("end\n")

    def visit_tile(self, tile):
        if tile not in self.visited_list:
            self.visited_list.append((tile[0], tile[1]))

    def graph_to_array(self):
        a = np.zeros((self.height, self.width), dtype=int)

        for i in range(self.height):
            for j in range(self.width):
                tile = (i, j)
                if tile not in self.getTiles():
                    a[i, j] = -1
                elif tile in self.visited_list:
                    a[i, j] = 1

        return a

    def coverage_array(self, full_map):
        size = full_map.height
        a = np.zeros((size, size), dtype=int)

        for i in range(size):
            for j in range(size):
                tile = (i, j)
                if tile in self.visited_list:
                    a[i, j] = 1

        return a

    def breadth_first_search(self, start, finish):
        # print out what we find
        frontier = Queue()
        frontier.put(start)
        reached = {start: True}
        reachable = False

        while not frontier.empty():
            current = frontier.get()
            if current == finish:  # early exit
                reachable = True
                break

            for next in self.map[current]:
                if (next not in reached) and (next not in self.visited_list):
                    frontier.put(next)
                    reached[next] = True
        return reachable

    def laser_scanner(self, tile, full_map):
        r = 3
        tiles = {"up": [],
                 "down": [],
                 "right": [],
                 "left": []}
        for i in range(r):
            tiles["up"].append((tile[0] - i, tile[1]))
            tiles["down"].append((tile[0] + i, tile[1]))
            tiles["right"].append((tile[0], tile[1] + i))
            tiles["left"].append((tile[0], tile[1] - i))
        directions = tiles.keys()
        to_remove = []

        for dir in directions:
            to_remove.clear()
            for t in tiles[dir]:
                if t in full_map.getTiles():
                    self.new_tile(t)
                    # self.print_graph()
                else:
                    to_remove.append(t)

                if t in self.visited_list:
                    to_remove.append(t)
            for rem in to_remove:
                tiles[dir].remove(rem)
        return tiles


class GridWorld:
    def __init__(self, config=None):
        config = config or {}
        # Dimensions of the grid.
        self.width = config.get("width", 6)
        self.height = config.get("height", 6)
        self.size = self.width  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.window = None
        self.clock = None
        self.observation_space = MultiDiscrete([5, 5, 5, 5])
        self.action_space = Discrete(4)
        # Reset env.
        self._state = State()
        self.reset()
        self.render_mode = "None"

        # For rendering.
        if config.get("render"):
            self.render_mode = "human"

    def reset(self):
        """Returns initial observation of next(!) episode."""
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
        # Row-major coords.
        # self.agent_pos = np.array([np.random.randint(0,self.width),np.random.randint(0,self.height)])   # upper left corner
        self.agent_pos = np.array([0, 0])  # upper left corner
        # Accumulated rewards in this episode.
        self.agent_R = 0.0

        # Reset agent1's visited fields.
        self.map = np.zeros((self.height, self.width), dtype=int)
        self.full_graph = GridMap(self.map)
        self.map[self.agent_pos[0], self.agent_pos[1]] = 1
        self.graph = GridMap()
        self.remaining = self.height * self.width - 1;
        # How many timesteps have we done in this episode.
        self.timesteps = 0
        # Return the initial observation in the new episode.
        self.graph.laser_scanner((self.agent_pos[0], self.agent_pos[1]), self.full_graph)
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        # limit_up, limit_down, limit_right, limit_left = self.find_limit()
        # self._state.update(np.array([limit_up, limit_down, limit_right, limit_left]))
        # return limit_up, limit_down, limit_right, limit_left

        obs = tuple(map(tuple, self.graph.coverage_array(self.full_graph))) + tuple(map(int, self.agent_pos))
        return obs

    def _get_info(self):
        # limit_up, limit_down, limit_right, limit_left, new_seen = self.find_limit()
        # return limit_up, limit_down, limit_right, limit_left
        return None, None, None, None

    def step(self, action):
        # increase our time steps counter by 1.
        self.timesteps += 1
        events = self._move(self.agent_pos, action)
        self.graph.laser_scanner((self.agent_pos[0], self.agent_pos[1]), self.full_graph)
        # Get observations (based on new agent positions).
        obs = self._get_obs()
        r = 0
        if "agent_new_field" in events:
            r = 1 / (self.width * self.height)
        else:
            r = -5 / (self.width * self.height)
        terminated = self.remaining == 0
        if terminated:
            r += 5
        if self.timesteps > self.height * self.width * 100:
            terminated = True
        self.agent_R += r
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return obs, r, terminated, info  # <- info dict (not needed here).

    def _move(self, coords, action):
        # Change the row: 0=up (-1), 2=down (+1)
        coords[0] += -1 if action == 0 else 1 if action == 2 else 0
        # Change the column: 1=right (+1), 3=left (-1)
        coords[1] += 1 if action == 1 else -1 if action == 3 else 0
        # No agent blocking -> check walls.
        blocked = False
        if coords[0] < 0:
            coords[0] = 0
            blocked = True
        elif coords[0] >= self.height:
            coords[0] = self.height - 1
            blocked = True
        if coords[1] < 0:
            coords[1] = 0
            blocked = True
        elif coords[1] >= self.width:
            coords[1] = self.width - 1
            blocked = True
        if blocked:
            return {"Blocked"}
        # If agent1 -> "new" if new tile covered.
        if not tuple(coords) in self.graph.visited_list:
            self.graph.visit_tile((coords[0], coords[1]))
            self.remaining -= 1;
            return {"agent_new_field"}
        # No new tile for agent1.
        return set()

    def render(self, mode=None):

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        size = max(self.graph.width, self.graph.height)
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))
        pix_square_size = (
                self.window_size / size
        )  # The size of a single grid square in pixels

        # First we draw the target
        a = self.graph.graph_to_array()
        for i in range(self.graph.height):
            for j in range(self.graph.width):
                if self.graph.graph_to_array()[i, j] == 1:
                    pygame.draw.rect(
                        canvas,
                        (255, 0, 0),
                        pygame.Rect(
                            pix_square_size * np.array([j, i]),
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif self.graph.graph_to_array()[i, j] == 0:
                    pygame.draw.rect(
                        canvas,
                        (255, 255, 255),
                        pygame.Rect(
                            pix_square_size * np.array([j, i]),
                            (pix_square_size, pix_square_size),
                        ),
                    )

        # Now we draw the agent

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (np.flip(np.array(self.agent_pos)) + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.graph.height + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            for x in range(self.graph.width + 1):
                pygame.draw.line(
                    canvas,
                    0,
                    (pix_square_size * x, 0),
                    (pix_square_size * x, self.window_size),
                    width=3,
                )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
