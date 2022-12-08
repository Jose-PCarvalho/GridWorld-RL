import numpy as np
from enum import Enum
import pygame
import gym
from gym import spaces
from gym.spaces import MultiDiscrete, Discrete
from collections import defaultdict


class State:
    def __init__(self):
        self.state = np.array((4, 1), dtype=int)

    def update(self, obs):
        self.state = obs


class Actions(Enum):
    UP = 0
    Down = 2
    Left = 3
    Right = 1


class GridTile:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x_y(x, y)
        self.visited = False

    def visit(self):
        self.visited = True


class GridMap:
    def __init__(self, a=None):
        if a is not None:
            self.height = a.shape[0]
            self.width = a.shape[1]
            self.map = {(0, 0): []}
            for i in range(self.width):
                for j in range(self.height):
                    self.new_tile((i, j))
        else:
            self.height = 1
            self.width = 1
            self.map = {(0, 0): []}

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

    def graph_to_array(self):
        a = np.zeros((self.height, self.width), dtype=int)

        for i in range(self.height):
            for j in range(self.width):
                tile = (i, j)
                if tile not in self.getTiles():
                    a[i, j] = -1

        return a
    # def is_reachable(self,start,finish):


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
        self.agent_visited_fields = set([tuple(self.agent_pos)])
        self.agent_seen_fields = set([tuple(self.agent_pos)])
        self.map = np.zeros((self.height, self.width), dtype=int)
        self.graph_map = GridMap(self.map)
        self.map[self.agent_pos[0], self.agent_pos[1]] = 1
        self.coverage_map = np.zeros((self.height, self.width), dtype=int)
        self.coverage_map[self.agent_pos[0], self.agent_pos[1]] = 1
        self.slam_map = np.zeros((self.height, self.width), dtype=int)
        self.slam_map[self.agent_pos[0], self.agent_pos[1]] = 1
        self.graph = GridMap()
        self.remaining = self.height * self.width - 1;
        # How many timesteps have we done in this episode.
        self.timesteps = 0
        # Return the initial observation in the new episode.
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        # limit_up, limit_down, limit_right, limit_left = self.find_limit()
        # self._state.update(np.array([limit_up, limit_down, limit_right, limit_left]))
        # return limit_up, limit_down, limit_right, limit_left
        # obs=tuple(map(tuple, self.coverage_map))

        obs = tuple(map(tuple, self.coverage_map)) + tuple(map(int, self.agent_pos))
        return obs

    def _get_info(self):
        limit_up, limit_down, limit_right, limit_left, new_seen = self.find_limit()
        return limit_up, limit_down, limit_right, limit_left

    def find_limit(self):
        up = True
        down = True
        left = True
        right = True
        limit_up = 0
        limit_down = 0
        limit_right = 0
        limit_left = 0
        new_seen = 0
        r = 3
        for i in range(1, r):
            if down:

                if self.agent_pos[0] == self.height - 1:
                    limit_down = 0
                    down = False
                elif not (self.agent_pos[0] + i, self.agent_pos[1]) in self.agent_seen_fields:
                    self.agent_seen_fields.add((self.agent_pos[0] + i, self.agent_pos[1]))
                    self.slam_map[self.agent_pos[0] + i, self.agent_pos[1]] = 1
                    self.graph.new_tile((self.agent_pos[0] + i, self.agent_pos[1]))
                    new_seen += 1
                elif (self.agent_pos[0] + i == self.height - 1) and (
                        self.coverage_map[self.agent_pos[0] + i, self.agent_pos[1]] == 0):
                    limit_down = i
                    down = False
                elif self.coverage_map[self.agent_pos[0] + i, self.agent_pos[1]] != 0:
                    limit_down = i - 1
                    down = False

            if up:

                if self.agent_pos[0] == 0:
                    limit_up = 0
                    up = False
                elif not (self.agent_pos[0] - i, self.agent_pos[1]) in self.agent_seen_fields:
                    self.agent_seen_fields.add((self.agent_pos[0] - i, self.agent_pos[1]))
                    self.slam_map[self.agent_pos[0] - i, self.agent_pos[1]] = 1
                    self.graph.new_tile((self.agent_pos[0] - i, self.agent_pos[1]))
                    new_seen += 1
                elif (self.agent_pos[0] - i == 0) and (
                        self.coverage_map[self.agent_pos[0] - i, self.agent_pos[1]] == 0):
                    limit_up = i
                    up = False
                elif self.coverage_map[self.agent_pos[0] - i, self.agent_pos[1]] != 0:
                    limit_up = i - 1
                    up = False

            for i in range(1, r):
                if right:

                    if self.agent_pos[1] == self.width - 1:
                        limit_right = 0
                        right = False
                    elif not (self.agent_pos[0], self.agent_pos[1] + i) in self.agent_seen_fields:
                        self.agent_seen_fields.add((self.agent_pos[0], self.agent_pos[1] + i))
                        self.slam_map[self.agent_pos[0], self.agent_pos[1] + i] = 1
                        self.graph.new_tile((self.agent_pos[0], self.agent_pos[1] + i))
                        new_seen += 1
                    elif (self.agent_pos[1] + i == self.width - 1) and (
                            self.coverage_map[self.agent_pos[0], self.agent_pos[1] + i] == 0):
                        limit_right = i
                        right = False
                    elif self.coverage_map[self.agent_pos[0], self.agent_pos[1] + i] != 0:
                        limit_right = i - 1
                        right = False

                if left:

                    if self.agent_pos[1] == 0:
                        limit_left = 0
                        left = False
                    elif not (self.agent_pos[0], self.agent_pos[1] - i) in self.agent_seen_fields:
                        self.agent_seen_fields.add((self.agent_pos[0], self.agent_pos[1] - i))
                        self.slam_map[self.agent_pos[0], self.agent_pos[1] - i] = 1
                        self.graph.new_tile((self.agent_pos[0], self.agent_pos[1] - i))
                        new_seen += 1
                    elif self.agent_pos[1] - i == 0 and (
                            self.coverage_map[self.agent_pos[0], self.agent_pos[1] - i] == 0):
                        limit_left = i
                        left = False
                    elif self.coverage_map[self.agent_pos[0], self.agent_pos[1] - i] != 0:
                        limit_left = i - 1
                        left = False

        return limit_up, limit_down, limit_right, limit_left, new_seen

    def step(self, action):
        # increase our time steps counter by 1.
        self.timesteps += 1
        events = self._move(self.agent_pos, action)
        # Get observations (based on new agent positions).
        obs = self._get_obs()
        self.closest()

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
        self.graph.graph_to_array()
        return obs, r, terminated, info  # <- info dict (not needed here).

    def closest(self):
        non_explored = np.argwhere(self.coverage_map == 0)
        if non_explored.size == 0:
            return 0
        distances = non_explored - self.agent_pos
        distances = np.linalg.norm(distances, axis=1)
        min_i = np.argmin(distances)
        return non_explored[min_i]

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
        if not tuple(coords) in self.agent_visited_fields:
            self.agent_visited_fields.add(tuple(coords))
            self.coverage_map[self.agent_pos[0], self.agent_pos[1]] = 1
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

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        for i in range(self.size):
            for j in range(self.size):
                if self.slam_map[i, j] == 1:
                    pygame.draw.rect(
                        canvas,
                        (255, 0, 0),
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
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
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
