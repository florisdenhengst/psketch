from .cookbook import Cookbook
from misc import array

import curses
import logging
import numpy as np
from skimage.measure import block_reduce
import time

WIDTH = 10
HEIGHT = 10

WINDOW_WIDTH = 5
WINDOW_HEIGHT = 5

N_WORKSHOPS = 3

DOWN = 0
UP = 1
LEFT = 2
RIGHT = 3
USE = 4
N_ACTIONS = USE + 1

ACTION_NAMES = {
        DOWN: 'down',
        UP: 'up',
        LEFT: 'left',
        RIGHT: 'right',
        USE: 'use',
        USE+1: 'STOP!',
        USE+2: 'forced STOP!',}

def random_free(grid, random):
    pos = None
    while pos is None:
        (x, y) = (random.randint(WIDTH), random.randint(HEIGHT))
        if grid[x, y, :].any():
            continue
        pos = (x, y)
    return pos

def neighbors(pos, dir=None):
    x, y = pos
    neighbors = []
    if x > 0 and (dir is None or dir == LEFT):
        neighbors.append((x-1, y))
    if y > 0 and (dir is None or dir == DOWN):
        neighbors.append((x, y-1))
    if x < WIDTH - 1 and (dir is None or dir == RIGHT):
        neighbors.append((x+1, y))
    if y < HEIGHT - 1 and (dir is None or dir == UP):
        neighbors.append((x, y+1))
    return neighbors

class CraftWorld(object):
    def __init__(self, config):
        self.cookbook = Cookbook(config.recipes)
        self.n_features = \
                2 * WINDOW_WIDTH * WINDOW_HEIGHT * self.cookbook.n_kinds + \
                self.cookbook.n_kinds + \
                4
        self.n_actions = N_ACTIONS

        self.non_grabbable_indices = self.cookbook.environment
        self.grabbable_indices = [i for i in range(self.cookbook.n_kinds)
                if i not in self.non_grabbable_indices]
        self.workshop_indices = [self.cookbook.index["workshop%d" % i]
                for i in range(N_WORKSHOPS)]
        self.water_index = self.cookbook.index["water"]
        self.stone_index = self.cookbook.index["stone"]

        self.random = np.random.RandomState(config.seed)

    def sample_scenario_with_goal(self, goal):
        assert goal not in self.cookbook.environment
        if goal in self.cookbook.primitives:
            make_island = goal == self.cookbook.index["gold"]
            make_cave = goal == self.cookbook.index["gem"]
            return self.sample_scenario({goal: 1}, make_island=make_island,
                    make_cave=make_cave)
        elif goal in self.cookbook.recipes:
            ingredients = self.cookbook.primitives_for(goal)
            return self.sample_scenario(ingredients)
        else:
            assert False, "don't know how to build a scenario for %s" % goal

    def sample_scenario(self, ingredients, make_island=False, make_cave=False):
        # generate grid
        grid = np.zeros((WIDTH, HEIGHT, self.cookbook.n_kinds), dtype=int)
        i_bd = self.cookbook.index["boundary"]
        grid[0, :, i_bd] = 1
        grid[WIDTH-1:, :, i_bd] = 1
        grid[:, 0, i_bd] = 1
        grid[:, HEIGHT-1:, i_bd] = 1

        # treasure
        if make_island or make_cave:
            (gx, gy) = (1 + np.random.randint(WIDTH-2), 1)
            treasure_index = \
                    self.cookbook.index["gold"] if make_island else self.cookbook.index["gem"]
            wall_index = \
                    self.water_index if make_island else self.stone_index
            grid[gx, gy, treasure_index] = 1
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if not grid[gx+i, gy+j, :].any():
                        grid[gx+i, gy+j, wall_index] = 1

        # ingredients
        for primitive in self.cookbook.primitives:
            if primitive == self.cookbook.index["gold"] or \
                    primitive == self.cookbook.index["gem"]:
                continue
            for i in range(4):
                (x, y) = random_free(grid, self.random)
                grid[x, y, primitive] = 1

        # generate crafting stations
        self.workshops = {}
        for i_ws in range(N_WORKSHOPS):
            ws_x, ws_y = random_free(grid, self.random)
            grid[ws_x, ws_y, self.cookbook.index["workshop%d" % i_ws]] = 1
            self.workshops[i_ws] = (ws_x, ws_y, i_ws, self.cookbook.index["workshop%d" % i_ws])

        # generate init pos
        init_pos = random_free(grid, self.random)

        return CraftScenario(grid, init_pos, self)

    def visualize(self, transitions, task):
        def _visualize(win):
            curses.start_color()
            for i in range(1, 8):
                curses.init_pair(i, i, curses.COLOR_BLACK)
                curses.init_pair(i+10, curses.COLOR_BLACK, i)
            states = [transitions[0].s1] + [t.s2 for t in transitions]
            mstates = [transitions[0].m1] + [t.m2 for t in transitions]
            actions = [t.a for t in transitions]
            for state, mstate, action in zip(states, mstates, actions):
                sleep = .5
                win.clear()
                win.addstr("Goal: {}\n".format(self.cookbook.index.get(task.goal[1])))
                win.addstr("Action: {}\n".format(ACTION_NAMES[action]))
                win.addstr("Subtask i: {}\n".format(mstate.at_subtask))
                if state is None:
                    win.addstr("State is None\n")
                    sleep = 1
                else:
                    state.visualize(win, self.cookbook, action)
                win.refresh()
                time.sleep(sleep)
        curses.wrapper(_visualize)

class CraftScenario(object):
    def __init__(self, grid, init_pos, world):
        self.init_grid = grid
        self.init_pos = init_pos
        self.init_dir = 0
        self.world = world

    def init(self):
        inventory = np.zeros(self.world.cookbook.n_kinds, dtype=int)
        state = CraftState(self, self.init_grid, self.init_pos, self.init_dir, inventory)
        return state

class CraftState(object):
    def __init__(self, scenario, grid, pos, dir, inventory):
        self.scenario = scenario
        self.world = scenario.world
        self.grid = grid
        self.inventory = inventory
        self.pos = pos
        self.dir = dir
        self._cached_features = None

    def visualize(self, win, cookbook, action=None):
        try:
           win.addstr("\n" * (HEIGHT + 2) + "Features: {}\n".format(' '.join([str(i) for i in self.features()])))
        except curses.error:
            pass
        for y in range(HEIGHT):
            for x in range(WIDTH):
                if not (self.grid[x, y, :].any() or (x, y) == self.pos):
                    continue
                thing = self.grid[x, y, :].argmax()
                if (x, y) == self.pos:
                    if self.dir == LEFT:
                        ch1 = "<"
                    elif self.dir == RIGHT:
                        ch1 = ">"
                    elif self.dir == UP:
                        ch1 = "^"
                    elif self.dir == DOWN:
                        ch1 = "v"
                    if action == USE:
                        color = curses.color_pair(1)
                    color = curses.color_pair(0)
                elif thing == cookbook.index["boundary"]:
                    ch1 = curses.ACS_BOARD
                    color = curses.color_pair(10 + thing)
                else:
                    color = curses.color_pair(10 + thing)
                    name = cookbook.index.get(thing)
                    if name == 'iron':
                        ch1 = 'i'
                        color = curses.COLOR_WHITE
                    elif name == 'wood':
                        ch1 = 'w'
                        color = curses.COLOR_YELLOW
                    elif name == 'water':
                        ch1 = 'r'
                        color = curses.COLOR_BLUE
                    elif name == 'grass':
                        ch1 = 'g'
                        color = curses.COLOR_GREEN
                    elif name[:len('workshop')] == 'workshop':
                        ch1 = name[-1]
                        color = curses.COLOR_CYAN
                    else:
                        logging.debug('no item yet {}: {}'.format(thing, name))
                        ch1 = ' '
                        color = curses.color_pair(0)
                win.addch(HEIGHT-y + 2, x, ch1, color)

    def at_workshop(self, workshop_id):
        x, y = self.pos
        here = self.grid[x, y, :]
        workshop_thing_id = self.world.cookbook.index["workshop{}".format(workshop_id)]
        for nx, ny in neighbors(self.pos, self.dir):
            if self.grid[ny, nx, workshop_thing_id] > 0:
                return True
        return False

    def satisfies(self, goal_name, goal_arg):
        return self.inventory[goal_arg] > 0

    def features(self):
        if self._cached_features is None:
            x, y = self.pos
            hw = int(WINDOW_WIDTH / 2)
            hh = int(WINDOW_HEIGHT / 2)
            bhw = int((WINDOW_WIDTH * WINDOW_WIDTH) / 2)
            bhh = int((WINDOW_HEIGHT * WINDOW_HEIGHT) / 2)

            grid_feats = array.pad_slice(self.grid, (x-hw, x+hw+1), 
                    (y-hh, y+hh+1))
            grid_feats_big = array.pad_slice(self.grid, (x-bhw, x+bhw+1),
                    (y-bhh, y+bhh+1))
            grid_feats_big_red = block_reduce(grid_feats_big,
                    (WINDOW_WIDTH, WINDOW_HEIGHT, 1), func=np.max)
            #grid_feats_big_red = np.zeros((WINDOW_WIDTH, WINDOW_HEIGHT, self.world.cookbook.n_kinds))

            self.gf = grid_feats.transpose((2, 0, 1))
            self.gfb = grid_feats_big_red.transpose((2, 0, 1))

            pos_feats = np.asarray(self.pos)
            pos_feats[0] = int(pos_feats[0] / WIDTH)
            pos_feats[1] = int(pos_feats[1] / HEIGHT)

            dir_features = np.zeros(4, dtype=int)
            dir_features[self.dir] = 1

            features = np.concatenate((grid_feats.ravel(),
                    grid_feats_big_red.ravel(), self.inventory, 
                    dir_features))
            assert len(features) == self.world.n_features
            self._cached_features = features

        return self._cached_features

    def step(self, action):
        x, y = self.pos
        n_dir = self.dir
        n_inventory = self.inventory
        n_grid = self.grid

        reward = 0

        # move actions
        if action == DOWN:
            dx, dy = (0, -1)
            n_dir = DOWN
        elif action == UP:
            dx, dy = (0, 1)
            n_dir = UP
        elif action == LEFT:
            dx, dy = (-1, 0)
            n_dir = LEFT
        elif action == RIGHT:
            dx, dy = (1, 0)
            n_dir = RIGHT

        # use actions
        elif action == USE:
            cookbook = self.world.cookbook
            dx, dy = (0, 0)
            success = False
            for nx, ny in neighbors(self.pos, self.dir):
                here = self.grid[nx, ny, :]
                if not self.grid[nx, ny, :].any():
                    continue

                assert here.sum() == 1
                thing = here.argmax()

                if not(thing in self.world.grabbable_indices or \
                        thing in self.world.workshop_indices or \
                        thing == self.world.water_index or \
                        thing == self.world.stone_index):
                    continue
                
                n_inventory = self.inventory.copy()
                n_grid = self.grid.copy()

                if thing in self.world.grabbable_indices:
                    n_inventory[thing] += 1
                    n_grid[nx, ny, thing] = 0
                    success = True

                elif thing in self.world.workshop_indices:
                    # TODO not with strings
                    workshop = cookbook.index.get(thing)
                    for output, inputs in cookbook.recipes.items():
                        if inputs["_at"] != workshop:
                            continue
                        yld = inputs["_yield"] if "_yield" in inputs else 1
                        ing = [i for i in inputs if isinstance(i, int)]
                        if any(n_inventory[i] < inputs[i] for i in ing):
                            continue
                        n_inventory[output] += yld
                        for i in ing:
                            n_inventory[i] -= inputs[i]
                        success = True

                elif thing == self.world.water_index:
                    if n_inventory[cookbook.index["bridge"]] > 0:
                        n_grid[nx, ny, self.world.water_index] = 0
                        n_inventory[cookbook.index["bridge"]] -= 1

                elif thing == self.world.stone_index:
                    if n_inventory[cookbook.index["axe"]] > 0:
                        n_grid[nx, ny, self.world.stone_index] = 0
                break

        # other
        else:
            raise Exception("Unexpected action: %s" % action)

        n_x = x + dx
        n_y = y + dy
        if self.grid[n_x, n_y, :].any():
            n_x, n_y = x, y

        new_state = CraftState(self.scenario, n_grid, (n_x, n_y), n_dir, n_inventory)
        return reward, new_state

    def next_to(self, i_kind):
        x, y = self.pos
        return self.grid[x-1:x+2, y-1:y+2, i_kind].any()
