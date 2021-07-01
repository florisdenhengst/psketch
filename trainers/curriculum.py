from misc import util
from misc.experience import Transition
from worlds.cookbook import Cookbook
from worlds import domain_knowledge

from collections import defaultdict, namedtuple
import itertools
import logging
import numpy as np
import yaml

N_ITERS = 3000000
N_UPDATE = 500
N_BATCH = 100
IMPROVEMENT_THRESHOLD = 0.8

Task = namedtuple("Task", ["goal", "steps"])

class CurriculumTrainer(object):
    def __init__(self, config):
        # load configs
        self.config = config
        self.cookbook = Cookbook(config.recipes)
        self.subtask_index = util.Index()
        self.task_index = util.Index()
        with open(config.trainer.hints) as hints_f:
            self.hints = yaml.load(hints_f)
        
        self.dk_model = domain_knowledge.domain_model(config)
        self.symbolic_action_index = util.Index()
        for sa in self.dk_model.symbolic_actions:
            self.symbolic_action_index.index(sa)

        # initialize randomness
        self.random = np.random.RandomState(config.seed)

        # organize task and subtask indices
        self.tasks_by_subtask = defaultdict(list)
        self.tasks = []
        self.n_train = self.config.trainer.n_train
        for hint_key, hint in self.hints.items():
            goal = util.parse_fexp(hint_key)
            goal = (self.subtask_index.index(goal[0]), self.cookbook.index[goal[1]])
            if config.model.use_args:
                steps = [util.parse_fexp(s) for s in hint]
                steps = [(self.subtask_index.index(a), self.cookbook.index[b])
                        for a, b in steps]
                steps = tuple(steps)
                task = Task(goal, steps)
                for subtask, _ in steps:
                    self.tasks_by_subtask[subtask].append(task)
            else:
                steps = [self.subtask_index.index(a) for a in hint]
                steps = tuple(steps)
                task = Task(goal, steps)
                for subtask in steps:
                    self.tasks_by_subtask[subtask].append(task)
            self.tasks.append(task)
            self.task_index.index(task)
        logging.debug("COOKBOOK {}".format(self.cookbook.index))

    def do_rollout(self, model, world, possible_tasks, task_probs):
        states_before = []
        tasks = []
        goal_names = []
        goal_args = []

        # choose N_BATCH tasks and initialize model N_BATCH times
        for _ in range(N_BATCH):
            task = possible_tasks[self.random.choice(
                len(possible_tasks), p=task_probs)]
            goal, _ = task
            goal_name, goal_arg = goal
            scenario = world.sample_scenario_with_goal(goal_arg)
            states_before.append(scenario.init())
            tasks.append(task)
            goal_names.append(goal_name)
            goal_args.append(goal_arg)
        model.init(states_before, tasks)
        transitions = [[] for _ in range(N_BATCH)]

        # initialize timer
        total_reward = 0.
        timer = self.config.trainer.max_timesteps
        done = [False,] * N_BATCH

        # act!
        while not all(done) and timer > 0:
            # takes N_BATCH steps simultaneously
            mstates_before = model.get_state()
            action, terminate, symbolic_act = model.act(states_before)
            mstates_after = model.get_state()
            states_after = [None for _ in range(N_BATCH)]
            for i in range(N_BATCH):
                if action[i] is None:
                    assert done[i] or terminate[i]
                elif terminate[i]:
                    win = states_before[i].satisfies(goal_names[i], goal_args[i])
                    reward = 1 if win else 0
                    states_after[i] = None
                elif action[i] >= world.n_actions:
                    states_after[i] = states_before[i]
                    reward = 0
                else:
                    reward, states_after[i] = states_before[i].step(action[i])

                if not done[i]:
                    assert action[i] is not None
                    transitions[i].append(Transition(
                            states_before[i], mstates_before[i], symbolic_act[i], action[i], 
                            states_after[i], mstates_after[i], reward))
                    total_reward += reward

                if terminate[i]:
                    done[i] = True

            states_before = states_after
            timer -= 1

        return transitions, total_reward / N_BATCH, tasks

    def train(self, model, world):
        model.prepare(world, self)
        #model.load()
        subtasks = sorted(self.tasks_by_subtask.keys())
        # the maximum length of the sequence of symbolic actions for training (task length)
        if self.config.trainer.use_curriculum:
            max_steps = 1
        else:
            max_steps = 100
        i_iter = 0

        task_probs = []
        n = 0
        while n < self.n_train: # FdH: N_ITERS = 30K
            logging.debug("[train step] %d", n)
            logging.debug("[iteration] %d", i_iter)
            logging.debug("[max steps] %d", max_steps)
            min_reward = np.inf

            # TODO refactor
            for _ in range(1):
                # make sure there's something of this length
                possible_tasks = self.tasks
                possible_tasks = [t for t in possible_tasks if len(t.steps) <= max_steps]
                if len(possible_tasks) == 0:
                    # skip, there are no tasks with this number of actions
                    continue

                # re-initialize task probs if necessary
                if len(task_probs) != len(possible_tasks):
                    # uniform initialization over all possible tasks
                    task_probs = np.ones(len(possible_tasks)) / len(possible_tasks)

                total_reward = 0. # FdH: total avg reward
                total_err = 0.
                total_reward_episodes = 0.
                errs = []
                count = 0.
                task_rewards = defaultdict(lambda: 0)
                task_counts = defaultdict(lambda: 0)
                for j in range(N_UPDATE): # FdH: N_UPDATE=500
                    err = None
                    # get enough samples for one training step
                    while err is None:
                        i_iter += N_BATCH
                        # list of episodes (list of transitions) and avg reward?
                        episodes, reward, tasks = self.do_rollout(model, world, 
                                possible_tasks, task_probs) # produces N_BATCH = 100 episodes
                        for e in episodes:
                            tr = sum(tt.r for tt in e)
                            task_rewards[e[0].m1.task] += tr
                            task_counts[e[0].m1.task] += 1
                            total_reward_episodes += tr
                        total_reward += reward
                        count += 1
                        for e_i, e in enumerate(episodes):
                            # TODO FdH: remove logdebug nonsense
                            if e_i == 0:
                                logdebug = False
                            else:
                                logdebug = True
                            model.experience(e, logdebug)
                        err = model.train()
                        errs.append(err)
                    total_err += err
#                    world.visualize(episodes[0], tasks[0])

                # log
                logging.info("[step] %d", i_iter)
                logging.info("[n train] %d", n)
                scores = []
                for i, task in enumerate(possible_tasks):
                    i_task = self.task_index[task]
                    score = 1. * task_rewards[i_task] / task_counts[i_task]
                    logging.info("[task] %s[%s] %s %s", 
                            self.subtask_index.get(task.goal[0]),
                            self.cookbook.index.get(task.goal[1]),
                            task_probs[i],
                            score)
                    scores.append(score)
                logging.info("")
                logging.info("[n episodes] %s", len(episodes))
                logging.info("[count] %d", count)
                # reward here denotes % of successful runs in [0,1]
                logging.info("[reward] %s", total_reward / count)
                logging.info("[error] %s", total_err / N_UPDATE)
                logging.info("")
                min_reward = min(min_reward, min(scores))
                n += 1

                # recompute task probs
                if self.config.trainer.use_curriculum:
                    task_probs = np.zeros(len(possible_tasks))
                    for i, task in enumerate(possible_tasks):
                        i_task = self.task_index[task]
                        task_probs[i] = 1. * task_rewards[i_task] / task_counts[i_task]
                    task_probs = 1 - task_probs
                    if len(possible_tasks) > 5:
                        task_probs += 0.01
                    else:
                        # TODO FdH: reset 0.01 probability!
                        task_probs += 0.05
                    task_probs /= task_probs.sum()

            logging.info("[min reward] %s", min_reward)
            logging.info("")
            if min_reward > self.config.trainer.improvement_threshold:
                max_steps += 1
        model.save()
