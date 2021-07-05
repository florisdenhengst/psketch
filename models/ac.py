from misc import util
from . import net
from worlds import domain_knowledge

from collections import namedtuple, defaultdict
import logging
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.framework.ops import IndexedSlicesValue

N_UPDATE = 2000
N_BATCH = 2000

N_HIDDEN = 128
N_EMBED = 64

DISCOUNT = 0.9

ActorModule = namedtuple("ActorModule", ["t_probs", "t_chosen_prob", "params",
        "t_decrement_op"])
CriticModule = namedtuple("CriticModule", ["t_value", "params"])
Trainer = namedtuple("Trainer", ["t_loss", "t_grad", "t_train_op"])
InputBundle = namedtuple("InputBundle", ["t_arg", "t_step", "t_feats", 
        "t_action_mask", "t_reward"])

ModelState = namedtuple("ModelState", ["action", "arg", "remaining", "task", "step", "at_subtask"])


def increment_sparse_or_dense(into, increment):
    assert isinstance(into, np.ndarray)
    if isinstance(increment, IndexedSlicesValue):
        for i in range(increment.values.shape[0]):
            into[increment.indices[i], :] += increment.values[i, :]
    else:
        into += increment

class ActorCriticModel(object):
    def __init__(self, config):
        self.experiences = []
        self.world = None
        self.next_actor_seed = config.seed
        self.config = config
        self.dk_model = domain_knowledge.domain_model(config)
        self.shaping_reward = self.config.model.shaping_reward

    def prepare(self, world, trainer):
        """
        Set up model a single model:
        * Set up n actor models for n symbolic actions ('steps')
        * Set up m critic models for m tasks
        """
        assert self.world is None
        self.world = world
        self.trainer = trainer

        # number of tasks
        self.n_tasks = len(trainer.task_index)
        # number of symbolic actions
        self.n_modules = len(trainer.subtask_index)
        # max number of actions per task
        self.max_task_steps = max(len(t.steps) for t in trainer.task_index.contents.keys())
        if self.config.model.featurize_plan:
            self.n_features = world.n_features + self.n_modules * self.max_task_steps
        else:
            self.n_features = world.n_features
        
        # number of actions in the world + 'STOP'
        self.n_actions = world.n_actions + 1
        self.n_net_actions = self.n_actions - 1
        self.STOP = world.n_actions
        self.FORCE_STOP = self.n_actions + 1
        # number of times train() has been completed
        self.t_n_steps = tf.Variable(1., name="n_steps")
        self.t_inc_steps = self.t_n_steps.assign(self.t_n_steps + 1)
        # TODO configurable optimizer
        self.optimizer = tf.compat.v1.train.RMSPropOptimizer(0.001)

        def build_actor(index, t_input, t_action_mask, extra_params=[]):
            with tf.compat.v1.variable_scope("actor_%s" % index):
                t_action_score, v_action = net.mlp(t_input, (N_HIDDEN, self.n_net_actions))

                # TODO this is pretty gross
                # NOTE FdH: this hardcodes a bias against the last action, STOP in the original work
                v_bias = v_action[-1]
                assert "b1" in v_bias.name
                t_decrement_op = v_bias[-1].assign(v_bias[-1] - 0)

                t_action_logprobs = tf.nn.log_softmax(t_action_score)
                t_chosen_prob = tf.math.reduce_sum(t_action_mask * t_action_logprobs, 
                        axis=(1,))

            return ActorModule(t_action_logprobs, t_chosen_prob, 
                    v_action+extra_params, t_decrement_op)

        def build_critic(index, t_input, t_reward, extra_params=[]):
            with tf.compat.v1.variable_scope("critic_%s" % index):
                if self.config.model.baseline in ("task", "common"):
                    t_value = tf.compat.v1.get_variable("b", shape=(),
                            initializer=tf.compat.v1.constant_initializer(0.0))
                    v_value = [t_value]
                elif self.config.model.baseline == "state":
                    t_value, v_value = net.mlp(t_input, (1,))
                    t_value = tf.squeeze(t_value)
                else:
                    raise NotImplementedError(
                            "Baseline %s is not implemented" % self.config.model.baseline)
            return CriticModule(t_value, v_value + extra_params)

        def build_actor_trainer(actor, critic, t_reward):
            t_advantage = t_reward - critic.t_value
            # TODO configurable entropy regularizer
            actor_loss = -tf.math.reduce_sum(actor.t_chosen_prob * t_advantage) + \
                    0.001 * tf.math.reduce_sum(tf.exp(actor.t_probs) * actor.t_probs)
            actor_grad = tf.gradients(actor_loss, actor.params)
            actor_trainer = Trainer(actor_loss, actor_grad, 
                    self.optimizer.minimize(actor_loss, var_list=actor.params))
            return actor_trainer

        def build_critic_trainer(t_reward, critic):
            t_advantage = t_reward - critic.t_value
            critic_loss = tf.math.reduce_sum(tf.math.square(t_advantage))
            critic_grad = tf.compat.v1.gradients(critic_loss, critic.params)
            critic_trainer = Trainer(critic_loss, critic_grad, self.optimizer.minimize(critic_loss, var_list=critic.params))
            return critic_trainer

        # placeholders
        t_arg = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        t_step = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
        t_feats = tf.compat.v1.placeholder(tf.float32, shape=(None, self.n_features))
        t_action_mask = tf.compat.v1.placeholder(tf.float32, shape=(None, self.n_net_actions))
        t_reward = tf.compat.v1.placeholder(tf.float32, shape=(None,))

        if self.config.model.use_args:
            t_embed, v_embed = net.embed(t_arg, len(trainer.cookbook.index),
                    N_EMBED)
            xp = v_embed
            t_input = tf.concat(1, (t_embed, t_feats))
        else:
            t_input = t_feats
            xp = []

        # maps symbolic actions ('modules') -> actors
        actor = None
        # maps (task, action)-tuples -> actor trainers with 1 trainer per tuple
        actor_trainer = None
        # maps (task, action)-tuples -> critics with 1 critic per task
        critic = None
        # maps (task, action)-tuples -> critic trainers with 1 trainer per tuple
        critic_trainer = None

        if self.config.model.featurize_plan:
            actor = build_actor(0, t_input, t_action_mask, extra_params=xp)
        else:
            logging.debug("Building actor")
            actor = build_actor(0, t_input, t_action_mask, extra_params=xp)

        if self.config.model.baseline == "common":
            logging.debug("Building single critic".format(self.n_tasks))
            common_critic = build_critic(0, t_input, t_reward, extra_params=xp)
        else:
            logging.debug("Building single critic".format(self.n_tasks))

        if self.config.model.baseline == "common":
            critic = common_critic
        else:
            critic = build_critic(0, t_input, t_reward, extra_params=xp)

        critic_trainer = build_critic_trainer(t_reward, critic)

        actor_trainer = build_actor_trainer(actor, critic, t_reward)

        self.t_gradient_placeholders = {}
        self.t_update_gradient_op = None
        
        params = [actor.params, critic.params]
        self.saver = tf.compat.v1.train.Saver()

        self.session = tf.compat.v1.Session()
        self.session.run(tf.compat.v1.initialize_all_variables())
        self.session.run(actor.t_decrement_op)

        self.actor = actor
        self.critic = critic
        self.actor_trainer = actor_trainer
        self.critic_trainer = critic_trainer
        self.inputs = InputBundle(t_arg, t_step, t_feats, t_action_mask, t_reward)

    def init(self, states, tasks):
        n_act_batch = len(states)
        self.subtasks = []
        self.args = []
        self.i_task = []
        self.dks = []
        for i in range(n_act_batch):
            if self.config.model.use_args:
                subtasks, args = zip(*tasks[i].steps)
                logging.debug(subtasks)
                logging.debug(args)
            else:
                subtasks = tasks[i].steps
                args = [None] * len(subtasks)
            self.subtasks.append(tuple(subtasks))
            self.args.append(tuple(args))
            self.i_task.append(self.trainer.task_index[tasks[i]])
            steps = tasks[i].steps
            steps = [self.trainer.subtask_index.indicesof(step) for step in steps]
            self.dks.append(self.dk_model.make(tasks[i].goal[1], steps, self.trainer.cookbook))
        # list storing the subtask for each episode
        self.i_subtask = [0 for _ in range(n_act_batch)]
        # vector of metacontroller states for each episode
        self.i_metacontroller_state = np.zeros(n_act_batch)
        self.i_symbolic_action = [None,] * n_act_batch
        self.i_action = [None,] * n_act_batch
        self.i_step = np.zeros((n_act_batch, 1))
        self.i_total_step = np.zeros((n_act_batch, 1))
        self.i_done = np.zeros((n_act_batch, 1))
        self.randoms = []
        for _ in range(n_act_batch):
            self.randoms.append(np.random.RandomState(self.next_actor_seed))
            self.next_actor_seed += 1

    def save(self):
        self.saver.save(self.session, 
                os.path.join(self.config.experiment_dir, "modular_ac.chk"))

    def load(self):
        """
        Loads parameters from file.
        """
        path = os.path.join(self.config.experiment_dir, "modular_ac.chk")
        logging.info("loaded %s", path)
        self.saver.restore(self.session, path)

    def experience(self, episode, logdebug=False):
        running_reward = 0
        shaped_running_reward = 0
        shaping_r = 0
        running_sa = None
        # TODO FdH: remove
        ep_len = len(episode)
        running_sa = episode[-1].sa1
        for i, transition in enumerate(episode[::-1]):
            # TODO FdH: fix rolling sa
            if transition.a == self.STOP:
#                logging.debug('STOP')
                shaping_r = self.shaping_reward
                running_sa = transition.sa1
            else:
                shaping_r = 0
            running_reward = (running_reward * DISCOUNT) + transition.r
            shaped_running_reward = (shaped_running_reward * DISCOUNT) + shaping_r
            n_transition = transition._replace(r=running_reward + shaped_running_reward,
                    sa1=running_sa)
#            if logdebug:
#                logging.debug("i: {}".format(i))
#                logging.debug("running rewards: {} {}".format(running_reward, shaped_running_reward))
#                logging.debug("transition: {} {} {} ".format(transition.a, transition.sa1, transition.r))
#                logging.debug("n_transition: {} {} {} ".format(n_transition.a, n_transition.sa1, n_transition.r))
            if n_transition.a < self.STOP:
#                if logdebug:
#                    logging.debug('append')
                self.experiences.append(n_transition)
            elif n_transition.a == self.FORCE_STOP:
                # STOP due to too many timesteps
#                if logdebug:
#                    logging.debug('"reset"')
                shaped_running_reward = 0
                running_sa = transition.sa1
                pass
            elif n_transition.a != self.STOP:
                raise ValueError('Unknown action {}'.format(n_transition.a))
#        if logdebug:
#            logging.debug('===')

    def featurize(self, state, mstate):
        if self.config.model.featurize_plan:
            task_features = np.zeros((self.max_task_steps, self.n_modules))
            for i, m in enumerate(self.trainer.task_index.get(mstate.task).steps):
                task_features[i, m] = 1
            return np.concatenate((state.features(), task_features.ravel()))
        else:
            return state.features()

    def act(self, states):
        mstates = self.get_state()
        self.i_step += 1
        self.i_total_step += 1
        by_mod = defaultdict(list)
        n_act_batch = len(self.i_subtask)
        action = [None,] * n_act_batch
        terminate = [None,] * n_act_batch
        symbolic_action = [None,] * n_act_batch
#        logging.debug('STEP: {}'.format(self.i_step[0]))
#        logging.debug('TOTAL: {}'.format(self.i_total_step[0]))
#        logging.debug('DK STATE ID: {} {}'.format(self.dks[0].state.id, self.dks[0].state.terminal))
#        if states[0] is not None:
#            logging.debug('GOT_WOOD: {}'.format(self.dks[0].check_inventory(states[0], 'wood')))
#            logging.debug('GOT_PLANK: {}'.format(self.dks[0].check_inventory(states[0], 'plank')))
#            logging.debug('AT WORKSHOP 0: {}'.format(states[0].at_workshop('0')))
#        else:
#            logging.debug('STATE 0 is none')
        
        for i, dk in enumerate(self.dks):
            self.i_done[i] = self.i_done[i][0] or dk.state.terminal
        
        force_stops_i = np.logical_and(np.logical_not(self.i_done), self.i_step >= self.config.model.max_subtask_timesteps)
        continue_i = np.logical_and(np.logical_not(self.i_done), np.logical_not(force_stops_i))
#        logging.debug('DONE: {}'.format(self.i_done[0]))
#        logging.debug('FORCE_STOP : {}'.format(force_stops_i[0]))
#        logging.debug('CONTINUE: {}'.format(continue_i[0]))
        
        for i in np.where(force_stops_i)[0]:
            action[i] = self.FORCE_STOP
            symbolic_action[i] = None
            # advance() automaton to random subsequent state
            terminated = self.dks[i].advance(self.randoms[i])
#            if i == 0:
#                logging.debug('FORCE_STOP ADVANCE TO: {}'.format(self.dks[i].state.id))
            if terminated:
                terminate[i] = 1.
            self.i_step[i] = 0.

        # TODO FdH: vectorize this loop if slow, esp. tensorflow self.session.run() calls
        # if not FORCE_STOP:
        for i in np.where(continue_i)[0]:
            # determine available SAs
            # tick with current state and previous! action
            if states[i] is None:
                raise ValueError('State is None for {}'.format(i))
            symbolic_actions, advanced, terminated = self.dks[i].tick(states[i], self.i_action[i])
#            if i == 0:
#                logging.debug('ADV, TERM: {}, {}'.format(advanced, terminated))
            if terminated:
                terminate[i] = 1.
            if advanced:
                action[i] = self.STOP
                symbolic_action[i] = self.i_symbolic_action[i]
                self.i_step[i] = 0.0
            else:
                # collect value estimates for all available SAs
                feed_dict = {
                    self.inputs.t_feats: [self.featurize(states[i], mstates[i])],
                }
                if self.config.model.use_args:
                    feed_dict[self.inputs.t_arg] = [mstates[i].arg]
#                logging.debug('actor_i: {}'.format(actor_i))
                    # TODO FdH: remove debug line
                symbolic_action[i] = None
                actor = self.actor
                logprobs = self.session.run([actor.t_probs], feed_dict=feed_dict)[0][0]
                probs = np.exp(logprobs)
#                if i == 0:
#                    logging.debug('probs: {} '.format(probs))
#                logging.debug('ACT{}: actor {} probs {}'.format(i, selected_sa, probs))
                action[i] = self.randoms[i].choice(self.n_net_actions, p=probs)
#                logging.debug('ACT{}: actor {} acts {}'.format(i, selected_sa, action[i]))
#        logging.debug('action: {}'.format(action[0]))
        self.i_symbolic_action = symbolic_action
        self.i_action = action
        return action, terminate, symbolic_action

    def get_state(self):
        out = []
        for i in range(len(self.i_subtask)):
            if self.i_subtask[i] >= len(self.subtasks[i]):
                out.append(ModelState(None, None, None, None, [0.], None))
            else:
                out.append(ModelState(
                        self.subtasks[i][self.i_subtask[i]], 
                        self.args[i][self.i_subtask[i]], 
                        len(self.args) - self.i_subtask[i],
                        self.i_task[i],
                        self.i_step[i].copy(),
                        self.i_subtask[i]))
        return out

    def train(self, action=None, update_actor=True, update_critic=True):
        if action is None:
            experiences = self.experiences
        else:
            experiences = [e for e in self.experiences if e.m1.action == action]
        if len(experiences) < N_UPDATE: # 2000
            return None
        # Select first experience tuples from batch (experience = s,a,r,s')
        # Note FdH: should be randomly sampling? (experiences are cleared, so only new exps
        batch = experiences[:N_UPDATE]

        # mapping modules, i.e. (task, symbolic action) tuples, to list of episodes
        by_mod = defaultdict(list)
        for e in batch:
            by_mod[e.m1.task, e.m1.action].append(e)

        grads = {}
        params = {}
        for module in [self.actor, self.critic]:
            for param in module.params:
                if param.name not in grads:
                    grads[param.name] = np.zeros(param.get_shape(), np.float32)
                    params[param.name] = param
        touched = set()

        total_actor_err = 0
        total_critic_err = 0
        # this loop determines the gradients for both critic and actor networks
        for i_task, i_mod1 in sorted(by_mod):
            actor = self.actor # actor for this symbolic action
            critic = self.critic # critic for this task + symbolic action
            actor_trainer = self.actor_trainer
            critic_trainer = self.critic_trainer

            # episodes for this (task, symbolic action) tuple
            all_exps = by_mod[i_task, i_mod1]
            # FdH: is i_batch ever > 0? since len(all_exps) is always < N_BATCH (subset of)
            for i_batch in range(int(np.ceil(1. * len(all_exps) / N_BATCH))):
                exps = all_exps[i_batch * N_BATCH : (i_batch + 1) * N_BATCH]
                s1, m1, sa1, a, s2, m2, r = zip(*exps)
                feats1 = [self.featurize(s, m) for s, m in zip(s1, m1)]
                args1 = [m.arg for m in m1]
                # steps are the symbolic actions to in a task
                steps1 = [m.step for m in m1]
                a_mask = np.zeros((len(exps), self.n_net_actions))
                for i_datum, aa in enumerate(a):
                    a_mask[i_datum, aa] = 1

                feed_dict = {
                    self.inputs.t_feats: feats1,
                    self.inputs.t_action_mask: a_mask,
                    self.inputs.t_reward: r
                }
                if self.config.model.use_args:
                    feed_dict[self.inputs.t_arg] = args1

                actor_grad, actor_err = self.session.run([actor_trainer.t_grad, actor_trainer.t_loss],
                        feed_dict=feed_dict)
                critic_grad, critic_err = self.session.run([critic_trainer.t_grad, critic_trainer.t_loss], 
                        feed_dict=feed_dict)

                total_actor_err += actor_err
                total_critic_err += critic_err

                if update_actor:
                    for param, grad in zip(actor.params, actor_grad):
                        increment_sparse_or_dense(grads[param.name], grad)
                        touched.add(param.name)
                if update_critic:
                    for param, grad in zip(critic.params, critic_grad):
                        increment_sparse_or_dense(grads[param.name], grad)
                        touched.add(param.name)
       
        # normalize and rescale the gradients
        global_norm = 0.0
        for k in params:
            grads[k] /= N_UPDATE
            global_norm += (grads[k] ** 2).sum()
        rescale = min(1., 1. / global_norm)

        # TODO precompute this part of the graph
        updates = []
        feed_dict = {}
        for k in params:
            param = params[k]
            grad = grads[k]
            grad *= rescale
            if k not in self.t_gradient_placeholders:
                self.t_gradient_placeholders[k] = tf.compat.v1.placeholder(tf.float32, grad.shape)
            feed_dict[self.t_gradient_placeholders[k]] = grad
            updates.append((self.t_gradient_placeholders[k], param))
        if self.t_update_gradient_op is None:
            self.t_update_gradient_op = self.optimizer.apply_gradients(updates)
        self.session.run(self.t_update_gradient_op, feed_dict=feed_dict)

        # Clear experiences
        self.experiences = []
        self.session.run(self.t_inc_steps)

        return np.asarray([total_actor_err, total_critic_err]) / N_UPDATE
