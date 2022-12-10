import time

import numpy as np
from enum import Enum
import pygame
import gym
from gym import spaces
from gym.spaces import MultiDiscrete, Discrete
from collections import defaultdict
from collections import deque
import random
import heapq


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


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return not self.elements

    def put(self, item, priority: float):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


class Actions(Enum):
    UP = 0
    Down = 2
    Left = 3
    Right = 1


class Agent:
    def __init__(self, x, y):
        self.position = (x, y)
        self.reward = 0
        self.scanner = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
        self.x = x
        self.y = y
        self.last_action = 0

    def update_pos(self, tile):
        self.position = tile
        self.x = tile[0]
        self.y = tile[1]


class GridMap:
    def __init__(self, a=None, start=(0, 0)):
        if a is not None:
            self.height = a.shape[0]
            self.width = a.shape[1]
            self.map = {start: []}
            self.visited_list = [start]
            for i in range(self.width):
                for j in range(self.height):
                    self.new_tile((i, j))
        else:
            self.height = 1
            self.width = 1
            self.map = {start: []}
            self.visited_list = [start]

    @staticmethod
    def adjacentTiles(tile):

        return [(tile[0] + 1, tile[1]), (tile[0] - 1, tile[1]), (tile[0], tile[1] + 1), (tile[0], tile[1] - 1)]

    def getTiles(self):
        return list(self.map.keys())

    def new_tile(self, tile):
        if tile not in self.getTiles():
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
            self.visited_list.append(tile)

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

    def is_reachable(self, start):
        frontier = Queue()
        frontier.put(start)
        reached = {start: None}
        non_visited = set(self.map) - set(self.visited_list)
        reachable = True
        while not frontier.empty():
            current = frontier.get()
            for next in self.map[current]:
                if (next not in reached) and (next not in self.visited_list):
                    frontier.put(next)
                    reached[next] = current
        reached.pop(start)
        for t in non_visited:
            if t not in reached:
                reachable = False
                break
        return reachable

    def dijkstra_search(self, start, finish):
        frontier = PriorityQueue()
        frontier.put(start, 0)
        came_from = {start: None}
        cost_so_far = {start: 0}

        while not frontier.empty():
            current = frontier.get()

            if current == finish:
                break

            for next in self.map[current]:
                new_cost = cost_so_far[current] + 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost
                    frontier.put(next, priority)
                    came_from[next] = current

        return self.reconstruct_path(came_from, start, finish)

    @staticmethod
    def reconstruct_path(came_from, start, finish):
        current = finish
        path = []
        if finish not in came_from.keys():  # no path was found
            return []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.reverse()  # optional
        return path

    def closest(self, start):
        min = 100
        close = None
        non_visited = set(self.getTiles()) - set(self.visited_list)
        for t in non_visited:
            distance = np.linalg.norm(np.array(start) - np.array(t))
            if distance < min:
                min = distance
                close = t
        return close

    def laser_scanner(self, tile, full_map):
        r = 3
        tiles = {"up": [],
                 "down": [],
                 "right": [],
                 "left": [],
                 "up-right": [],
                 "up-left": [],
                 "down-right": [],
                 "down-left": []
                 }
        for i in range(1, r):
            tiles["up"].append((tile[0] - i, tile[1]))
            tiles["down"].append((tile[0] + i, tile[1]))
            tiles["right"].append((tile[0], tile[1] + i))
            tiles["left"].append((tile[0], tile[1] - i))
        tiles["up-right"].append((tile[0] - 1, tile[1] + 1))
        tiles["up-left"].append((tile[0] - 1, tile[1] - 1))
        tiles["down-right"].append((tile[0] + 1, tile[1] + 1))
        tiles["down-left"].append((tile[0] + 1, tile[1] - 1))
        directions = tiles.keys()
        to_remove = []

        for dir in directions:
            to_remove.clear()
            remove_further = False
            for t in tiles[dir]:
                if t in full_map.getTiles():
                    self.new_tile(t)
                else:
                    to_remove.append(t)

                if t in self.visited_list or (remove_further and t in full_map.getTiles()):
                    to_remove.append(t)
                    remove_further = True
            for rem in to_remove:
                tiles[dir].remove(rem)
        ranges = []
        for dir in directions:
            ranges.append(len(tiles[dir]))
        return ranges


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
        self.action_space = Discrete(5)
        self._state = State()
        self.render_mode = "None"
        # For rendering.
        if config.get("render"):
            self.render_mode = "human"
        # self.reset()

    def reset(self):
        """Returns initial observation of next(!) episode."""
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
        self.agent = Agent(random.randint(0, self.height - 1), random.randint(0, self.width - 1))
        self.map = np.zeros((self.height, self.width), dtype=int)
        self.full_graph = GridMap(self.map)
        self.map[self.agent.x, self.agent.y] = 1
        self.graph = GridMap(start=self.agent.position)
        self.remaining = self.height * self.width - 1;
        self.timesteps = 0
        self.agent.scanner.append(self.graph.laser_scanner(self.agent.position, self.full_graph))
        self.agent.scanner.pop(0)
        self.bad_move = False
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        # obs = tuple(map(tuple, self.graph.coverage_array(self.full_graph))) + tuple(map(int, self.agent.position))
        obs = tuple(map(tuple, self.agent.scanner)) + (self.agent.last_action,)
        return obs

    def _get_info(self):
        # limit_up, limit_down, limit_right, limit_left, new_seen = self.find_limit()
        # return limit_up, limit_down, limit_right, limit_left
        info = [0, 0, 0, 0]
        for i in range(4):
            info[i] = self.agent.scanner[-1][i]
        return info

    def step(self, action):
        # increase our time steps counter by 1.
        self.timesteps += 1
        events = self._move(action)
        r = 0
        if not self.graph.is_reachable(self.agent.position) and not self.bad_move:
            r -= 100 / (self.width * self.height)
            self.bad_move = True
            # print("BAD MOVE")
        if "agent_new_field" in events:
            r += 1 / (self.width * self.height)
        else:
            r -= 5 / (self.width * self.height)
        if action == self.agent.last_action:
            r += 0.5 / (self.width * self.height)
        if action != 4:
            if self.agent.scanner[-1][action] == 0 and self.agent.scanner[-2][action] != 0:
                r += 1 / (self.width * self.height)
        terminated = self.remaining == 0
        if terminated:
            r += 5
        if self.timesteps > self.height * self.width * 10:
            terminated = True
        self.agent.reward += r
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        self.agent.last_action = action
        obs = self._get_obs()
        return obs, r, terminated, info  # <- info dict (not needed here).

    def _move(self, action):
        heuristic = False
        events = []
        if action == 4:
            path = self.graph.dijkstra_search(self.agent.position, self.graph.closest(self.agent.position))
            self.agent.update_pos(path[0])
            heuristic = True
        elif action != 4:
            self.agent.x += -1 if action == 0 else 1 if action == 1 else 0
            # Change the column: 1=right (+1), 3=left (-1)
            self.agent.y += 1 if action == 2 else -1 if action == 3 else 0

        blocked = False
        if self.agent.x < 0:
            self.agent.x = 0
            blocked = True
        elif self.agent.x >= self.height:
            self.agent.x = self.height - 1
            blocked = True
        if self.agent.y < 0:
            self.agent.y = 0
            blocked = True
        elif self.agent.y >= self.width:
            self.agent.y = self.width - 1
            blocked = True
        if heuristic:
            events.append("Heuristic")
        if blocked:
            events.append("Blocked")
        self.agent.position = (self.agent.x, self.agent.y)
        if self.agent.position not in self.graph.visited_list:
            self.graph.visit_tile((self.agent.x, self.agent.y))
            self.remaining -= 1
            events.append("agent_new_field")
        self.agent.scanner.append(self.graph.laser_scanner(self.agent.position, self.full_graph))
        self.agent.scanner.pop(0)

        return events

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
            (np.flip(np.array(self.agent.position)) + 0.5) * pix_square_size,
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
