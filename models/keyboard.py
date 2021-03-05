from misc import util
from . import net
from . import trpo

from collections import namedtuple, defaultdict
import numpy as np
import tensorflow as tf

N_UPDATE = 2000
N_BATCH = 2000

N_HIDDEN = 128
N_EMBED = 64

DISCOUNT = 0.9

ModelState = namedtuple("ModelState", ["action", "arg", "remaining", "task", "step"])

class KeyboardModel(object):
    def __init__(self, config):
        self.world = None
        self.config = config

    def prepare(self, world, trainer):
        assert self.world is None
        self.world = world
        self.trainer = trainer
        self.max_task_steps = max(len(t.steps) for t in trainer.task_index.contents.keys())
        self.n_modules = len(trainer.subtask_index)

    def init(self, states, tasks):
        self.n_tasks = len(tasks)

    def save(self):
        pass

    def load(self):
        pass

    def experience(self, episode):
        pass

    def act(self, states):
        print(states[0].pp())
        print(self.featurize(states[0], None))
        k = input("action: ")
        action = int(k)
        terminate = action >= self.world.n_actions
        return [action] * len(states), [terminate] * len(states)

    def get_state(self):
        return [None] * self.n_tasks

    def train(self, action=None, update_actor=True, update_critic=True):
        pass
    
    def featurize(self, state, mstate):
        if self.config.model.featurize_plan:
            task_features = np.zeros((self.max_task_steps, self.n_modules))
            for i, m in enumerate(self.trainer.task_index.get(mstate.task).steps):
                task_features[i, m] = 1
            return np.concatenate((state.features(), task_features.ravel()))
        else:
            return state.features()
