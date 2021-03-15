from .cookbook import Cookbook
from misc import util

import numpy as np
import logging
import copy

DOWN = 0
UP = 1
LEFT = 2
RIGHT = 3
USE = 4
# STOP: 5

ROOM_W = 6
ROOM_H = 6
# TODO FdH: fix
P_KEY_PER_ROOM = 0.5
#P_KEY_PER_ROOM = 0.0

BOARD_SIZE_BOUND = 2

class LightWorld(object):
    def __init__(self, config):
        self.n_actions = 5
        self.n_features = 12
        self.cookbook = Cookbook(config.recipes)
        # NOTE FdH: use its own random state to ensure generating the same worlds
        self.random = np.random.RandomState(config.seed)

    def sample_scenario_with_goal(self, goal):
        self.goal = self.cookbook.index.get(goal)
        def walk():
            x, y = 0, 0
            for c in self.goal:
                if c == "L":
                    x -= 1
                elif c == "R":
                    x += 1
                elif c == "U":
                    y -= 1
                elif c == "D":
                    y += 1
                yield x, y

        # figure out board size
        l, r, u, d = 0, 0, 0, 0
        for x, y in walk():
            l = min(l, x)
            r = max(r, x)
            u = min(u, y)
            d = max(d, y)
        l -= self.random.randint(BOARD_SIZE_BOUND)
        r += self.random.randint(BOARD_SIZE_BOUND)
        u -= self.random.randint(BOARD_SIZE_BOUND)
        d += self.random.randint(BOARD_SIZE_BOUND)

        rooms_x = r - l + 1
        rooms_y = d - u + 1

        init_x = -l
        init_y = -u

        board_w = ROOM_W * rooms_x + 1
        board_h = ROOM_H * rooms_y + 1
        walls = np.zeros((board_w, board_h), dtype='int')
        walls[0::ROOM_W, :] = 1
        walls[:, 0::ROOM_H] = 1
        
        # List of (int:x,int:y) positions of doors
        doors = []
        # Map {(int:key_x,int:key_y): (int:door_x, int:door_y)} for position of
        # keys and correspodning doors
        keys = {}

        # create doors

        # necessary
        px, py = 0, 0
        for x, y in walk():
            dx = x - px
            dy = y - py
            cx = int(ROOM_W * (init_x + px) + ROOM_W / 2)
            cy = int(ROOM_H * (init_y + py) + ROOM_H / 2)
            wx = int(cx + ROOM_W / 2 * dx)
            wy = int(cy + ROOM_H / 2 * dy)
            kx = int(cx + self.random.randint(ROOM_W / 2 + 1) - 1)
            ky = int(cy + self.random.randint(ROOM_H / 2 + 1) - 1)
            walls[wx][wy] = 0
            doors.append((wx, wy))
            if self.random.rand() < P_KEY_PER_ROOM:
                keys[(kx, ky)] = (wx, wy)
            px, py = x, y

        # unnecessary
        for i_room in range(min(rooms_x, rooms_y)):
            if rooms_x == 1 or rooms_y == 1:
                continue
            px = self.random.randint(rooms_x-1)
            py = self.random.randint(rooms_y-1)
            dx, dy = (1, 0) if self.random.randint(2) else (0, 1)
            cx = int(ROOM_W * px + ROOM_W / 2)
            cy = int(ROOM_H * py + ROOM_H / 2)
            wx = int(cx + ROOM_W / 2 * dx)
            wy = int(cy + ROOM_H / 2 * dy)
            if (wx, wy) in doors:
                continue
            kx = int(cx + self.random.randint(ROOM_W / 2 + 1) - 1)
            ky = int(cy + self.random.randint(ROOM_H / 2 + 1) - 1)
            walls[wx][wy] = 0
            doors.append((wx, wy))
            if self.random.rand() < P_KEY_PER_ROOM:
                keys[(kx, ky)] = (wx, wy)

        # precompute features
        door_features = {d: np.zeros((board_w, board_h, 4)) for d in doors}
        key_features = {k: np.zeros((board_w, board_h, 4)) for k in keys}
        for x in range(board_w):
            for y in range(board_h):
                # room x and room y
                rx = int(x / ROOM_W)
                ry = int(y / ROOM_H)
                for dx, dy in doors:
                    if rx not in (int((dx + 1.0) / ROOM_W), int((dx - 1.0) / ROOM_W)):
                        continue
                    if ry not in (int((dy + 1.0) / ROOM_H), int((dy - 1.0) / ROOM_H)):
                        continue
                    if (x, y) != (dx, dy) and (x % ROOM_W == 0 or y % ROOM_H == 0):
                        continue
                    strength = 10.0 - np.sqrt(np.square((x - dx, y - dy)).sum())
                    strength = max(strength, 0.0)
                    strength /= 10.0
                    if dx <= x:
                        # door is to the left
                        door_features[dx, dy][x, y, 0] += strength
                    if dx >= x:
                        # door is to the right
                        door_features[dx, dy][x, y, 1] += strength
                    if dy <= y:
                        # door is up 
                        door_features[dx, dy][x, y, 2] += strength
                    if dy >= y:
                        # door is down
                        door_features[dx, dy][x, y, 3] += strength
                for kx, ky in keys:
                    if int(kx / ROOM_W) != rx or int(ky / ROOM_H) != ry:
                        continue
                    if x % ROOM_W == 0 or y % ROOM_H == 0:
                        continue
                    strength = 10 - np.sqrt(np.square((x - kx, y - ky)).sum())
                    strength = max(strength, 0.0)
                    strength /= 10.0
                    if kx <= x:
                        # key is to the left
                        key_features[kx, ky][x, y, 0] += strength
                    if kx >= x:
                        # key is to the right
                        key_features[kx, ky][x, y, 1] += strength
                    if ky <= y:
                        # key is up
                        key_features[kx, ky][x, y, 2] += strength
                    if ky >= y:
                        # key is down
                        key_features[kx, ky][x, y, 3] += strength

        #np.set_printoptions(precision=1)
        #print keys
        #print doors
        #print walls
        #print key_features[keys.keys()[0]][..., 0]
        #print key_features[keys.keys()[0]][..., 1]
        #print door_features[doors[0]][..., 0]
        #print door_features[doors[1]][..., 1]
        #exit()
        init_room = (init_x, init_y)
        gx, gy = list(walk())[-1]
        goal_room = (init_x + gx, init_y + gy)
        return LightScenario(walls, doors, keys, door_features, key_features, 
                init_room, goal_room, self.goal, self)

class LightScenario(object):
    def __init__(self, walls, doors, keys, door_features, key_features, 
            init_room, goal_room, hints, world):
        self.walls = walls
        self.hints = hints
        self.doors = doors
        self.keys = keys
        self.door_features = door_features
        self.key_features = key_features
        self.init_room = init_room
        self.goal_room = goal_room
        self.world = world

    def init(self):
        ix, iy = self.init_room
        ix = int(ROOM_W * ix + ROOM_W / 2)
        iy = int(ROOM_H * iy + ROOM_H / 2)
        s = LightState(self.walls, self.doors, self.keys, (ix, iy), self)
        return s

    def __str__(self):
        return "\n"+self.init().pp()

class LightState(object):
    def __init__(self, walls, doors, keys, pos, scenario):
        self.walls = walls
        self.doors = doors
        self.keys = keys
        self.pos = pos
        self.scenario = scenario
        self._cached_features = None

    def door_open(self, door):
        return door not in self.keys.values() 

    def features(self):
        if self._cached_features is None:
            # 3 sensors in cardinal directions, to read distance to
            # * open doors: 0-4 (LEFT, RIGHT, UP, DOWN)
            # * closed doors: positions 4-8 (LEFT, RIGHT, UP, DOWN)
            # * keys: positions 8-12 (LEFT, RIGHT, UP, DOWN)
            # A door is 'open' once the key to that door is held
            out = np.zeros(12)
            for door in self.doors:
                df = self.scenario.door_features[door][self.pos[0], self.pos[1], :]
                if self.door_open(door):
                    out[0:4] += df
                else:
                    out[4:8] += df
            for key in self.keys:
                kf = self.scenario.key_features[key][self.pos[0]][self.pos[1]][:]
                out[8:12] += kf
            self._cached_features = out
        return self._cached_features

    def position_to_room(self, x, y):
        return (int(x / ROOM_W), int(y / ROOM_H))

    def boundaries_room(self, room_x, room_y):
        "Returns ((x_min, x_max), (y_min, y_max)) bounding box for a given room"
        x_min = room_x * ROOM_W
        x_max = ((room_x+1) * ROOM_W)
        y_min = room_y * ROOM_H
        y_max = ((room_y+1) * ROOM_H)
        return ((x_min, x_max), (y_min, y_max))

    def satisfies(self, goal_name, goal_arg):
        room = self.position_to_room(*self.pos)
        sat = room == self.scenario.goal_room
        return sat

    def step(self, action):
        x, y = self.pos
        n_keys = self.keys
        # move actions
        if action == UP:
            dx, dy = (0, -1)
        elif action == DOWN:
            dx, dy = (0, 1)
        elif action == LEFT:
            dx, dy = (-1, 0)
        elif action == RIGHT:
            dx, dy = (1, 0)
        elif action == USE:
            n_keys = dict(self.keys)
            dx, dy = (0, 0)
            if (x, y) in self.keys:
                del n_keys[(x, y)]

        nx, ny = x + dx, y + dy
        if self.walls[nx, ny]:
            nx, ny = x, y
        if not self.door_open((nx, ny)):
            nx, ny = x, y
        return 0, LightState(self.walls, self.doors, n_keys, (nx, ny), self.scenario)

    def pp(self):
        w, h = self.walls.shape
        out = "Hints: {}\n".format(self.scenario.hints)
        ((goal_x_min, goal_x_max), (goal_y_min, goal_y_max)) = self.boundaries_room(*self.scenario.goal_room)
        for y in range(h):
            for x in range(w):
                if (x, y) == self.pos and (x, y) in self.keys:
                    out += "%m"
                elif (x, y) == self.pos:
                    out += "%%"
                elif self.walls[x, y]:
                    out += "##"
                elif (x, y) in self.keys:
                    out += "mm"
                elif (x, y) in self.doors and (x, y) in self.keys.values():
                    out += "$$"
                elif goal_x_min < x < goal_x_max and goal_y_min < y < goal_y_max:
                    out += ".."
                else:
                    out += "  "
            out += "\n"
        out += str(self.scenario.goal_room)
        out += ", sat: {}".format(self.satisfies(None, None))
        out += ", room: {}".format(self.position_to_room(self.pos[0], self.pos[1]))
        out += ", pos: {}".format(self.pos)
        return out
