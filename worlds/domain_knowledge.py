from worlds import light
from worlds import craft
from collections import namedtuple

import logging

N = namedtuple('N', ['id', 'terminal'])
T = namedtuple('T', ['source', 'target', 'label_i', 'label_o'])

class LightWorldDomainKnowledge():
    def __init__(self, goal, subgoals, cookbook):
        self.prev_action_label = None
        self.prev_observation = None
        # a sequence of symbolic high-level subgoals
        self.subgoals = subgoals
        self.current_goal_i = 0
    
    def advance(self, random):
        self.current_goal_i = min(len(self.subgoals), self.current_goal_i + 1)
    
    def tick(self, state, action):
        observation = state.features()
        subgoal_met = False
        #last_goal = self.current_goal_i == len(self.subgoals)
        if self.prev_action_label != None:
            subgoal_met = self.subgoal_met(observation)
            if subgoal_met:
                self.current_goal_i = min(len(self.subgoals) - 1, self.current_goal_i + 1)
        self.prev_observation = observation
        self.prev_action_label = action
        return subgoal_met #and not last_goal

    def in_door(self, observation):
        return observation[:4].sum() == 4.0

    def subgoal_met(self, observation):
        if not self.in_door(self.prev_observation):
            return False
        elif self.subgoals[self.current_goal_i] == 'left':
            return True
        elif self.subgoals[self.current_goal_i] == 'up':
            return True
        elif self.subgoals[self.current_goal_i] == 'down':
            return True
        elif self.subgoals[self.current_goal_i] == 'right':
            return True
        return False

class LightWorldParallelDomainKnowledge(LightWorldDomainKnowledge):
    def __init__(self, goal, subgoals, cookbook):
        super().__init__(goal, subgoals, cookbook)

class CraftWorldDomainKnowledge():
    def __init__(self, goal, subgoals, cookbook):
        self.prev_action_label = None
        self.prev_observation = None
        # a sequence of symbolic high-level subsubgoals
        self.subgoals = subgoals
        self.current_goal_i = 0
        self.cookbook = cookbook

    def advance(self, random):
        self.current_goal_i = min(len(self.subgoals), self.current_goal_i + 1)

    def tick(self, state, action):
        subgoal_met = self.subgoal_met(state, action)
        #last_goal = self.current_goal_i == len(self.subgoals)
        if subgoal_met:
            self.current_goal_i = min(len(self.subgoals), self.current_goal_i + 1)
        self.prev_state = state
        self.prev_action_label = action
        return subgoal_met #and not last_goal

    def check_inventory(self, state, keyword):
        # TODO FdH:
        #  * Change > 0.0 to == 1.0?
        return state.inventory[self.cookbook.index[keyword]] > 0.0

    def subgoal_met(self, state, action):
        subgoal = self.subgoals[self.current_goal_i]
        if subgoal == 'get_wood':
            return self.check_inventory(state, 'wood')
        if subgoal == 'get_grass':
            return self.check_inventory(state, 'grass')
        if subgoal == 'get_iron':
            return self.check_inventory(state, 'iron')
        if subgoal == 'get_axe':
            return self.check_inventory(state, 'axe')
        if subgoal == 'get_gold':
            return self.check_inventory(state, 'gold')
        if subgoal == 'get_gem':
            return self.check_inventory(state, 'gem')
        if subgoal[:4] == 'make':
            return self.prev_state.at_workshop(subgoal[-1]) and self.prev_action_label == craft.USE
        else:
            raise ValueError("Subgoal not known: {}".format(subgoal))
        return False

class CraftWorldAutomatonDomainKnowledge(CraftWorldDomainKnowledge):
    state_labelling = [
            'iron',
            'wood',
            'plank',
            'stick',
            'bridge',
            'axe',
            'shears',
            'gold',
            'gem',
            ]
    label_ia = []
   
    def __init__(self, goal, subgoals, cookbook):
        self.cookbook = cookbook
        self.state = self.states[0]
        self.prev_obs = None
        super().__init__(goal, subgoals, cookbook)


    # TODO FdH: implement action encoding instead
    def tick(self, observation, prev_action):
        old_state = self.state
        # input atomic propositions 
        label_io = [self.check_inventory(observation, i) for i in self.ap_io]
        symbolic_actions = self.transition(label_io)
        advance = self.state != old_state
        self.prev_obs = observation
        return symbolic_actions, advance, self.state.terminal

    def advance(self, random):
        # get the available successors
        # sample one uniformly with random seed
        target_nodes = [t.target for t in self.transitions[self.state.id] if t.target != self.state.id]
        terminals = [self.states[t].terminal for t in target_nodes]
        if all(terminals):
            target_candidates = target_nodes
        else:
            target_candidates = [t_id for t_id in target_nodes if not self.states[t_id].terminal]
        target = random.choice(target_candidates)
        assert target != self.state.id
        self.state = self.states[target]
        return self.state.terminal

    def transition(self, labelling):
        """
        Transitions transducer from current state -> target state as given by
        labelling and returns output label (label_o)
        """
        # Assumes transducer deterministic (1 transition per node+label_i)
        matched = None
        transitions = self.transitions[self.state.id]
        for t in transitions:
            match = []
            for i, l in enumerate(t.label_i):
                match.append(l is None or l == labelling[i])
            if all(match):
                matched = t
                break
        if matched is None:
            raise ValueError('No transition from state {} with labelling {}'.format(self.state.id, labelling))
        self.state = self.states[matched.target]
        return matched.label_o

class Bed(CraftWorldAutomatonDomainKnowledge):
    ap_io = ['wood', 'grass', 'plank', 'bed']
    states = {
            0: N(0, False),
            1: N(1, False),
            4: N(4, False),
            5: N(5, False),
            6: N(6, True),
            }
    transitions = {
            0: [
                T(0, 0, [False, None, False, False], ['get_wood',]),
                T(0, 1, [True, None, False, False,], ['make0']),
                T(0, 4, [None, False, True, False,], ['get_grass',]),
                T(0, 5, [None, True, True, False,], ['make1',]),
                T(0, 6, [None, None, None, True,], ['make1',]),
                ],
            1: [
                T(1, 1, [None, None, False, False,], ['make0']),
                T(1, 4, [None, False, True, False,], ['get_grass',]),
                T(1, 5, [None, True, True, False,], ['make1',]),
                T(1, 6, [None, None, None, True, None], ['make1',]),
                ],
            4: [
                T(4, 4, [None, False, None, False,], ['get_grass',]),
                T(4, 5, [None, True, None, False,], ['make1',]),
                T(4, 6, [None, None, None, True,], ['make1',]),
                ],
            5: [
                T(5, 5, [None, None, None, False,], ['make1',]),
                T(5, 6, [None, None, None, True,], ['make1']),
                ],
            6: [
                T(6, 6, [None,] * 4, ['make1'])
                ],
            }
class PBed(CraftWorldAutomatonDomainKnowledge):
    ap_io = ['wood', 'grass', 'plank', 'bed']
    states = {
            0: N(0, False),
            1: N(1, False),
            2: N(2, False),
            3: N(3, False),
            4: N(4, False),
            5: N(5, False),
            6: N(6, True),
            }
    transitions = {
            0: [
                T(0, 0, [False,] * 4, ['get_wood', 'get_grass']),
                T(0, 1, [True, False, False, False,], ['get_grass', 'make0']),
                T(0, 2, [False, True, False, False,], ['get_wood',]),
                T(0, 3, [True, True, False, False,], ['make0',]),
                T(0, 4, [None, False, True, False,], ['get_grass',]),
                T(0, 5, [None, True, True, False,], ['make1',]),
                T(0, 6, [None, None, None, True,], ['make1',]),
                ],
            1: [
                T(1, 1, [None, False, False, False,], ['get_grass', 'make0']),
                T(1, 3, [None, True, False, False,], ['make0']),
                T(1, 4, [None, False, True, False,], ['get_grass',]),
                T(1, 5, [None, True, True, False,], ['make1',]),
                T(1, 6, [None, None, None, True, None], ['make1',]),
                ],
            2: [
                T(2, 2, [False, None, False, False,], ['get_wood',]),
                T(2, 3, [True, None, False, False,], ['make0',]),
                T(2, 5, [None, True, True, False,], ['make1',]),
                T(2, 6, [None, None, None, True,], ['make1',]),
                ],
            3: [
                T(3, 3, [None, None, False, False, ], ['make0',]),
                T(3, 5, [None, None, True, False, ], ['make1',]),
                T(3, 6, [None, None, None, True, ], ['make1',]),
                ],
            4: [
                T(4, 4, [None, False, None, False,], ['get_grass',]),
                T(4, 5, [None, True, None, False,], ['make1',]),
                T(4, 6, [None, None, None, True,], ['make1',]),
                ],
            5: [
                T(5, 5, [None, None, None, False,], ['make1',]),
                T(5, 6, [None, None, None, True,], ['make1']),
                ],
            6: [
                T(6, 6, [None,] * 4, ['make1'])
                ],
            }

class Axe(CraftWorldAutomatonDomainKnowledge):
    ap_io = ['wood', 'iron', 'stick', 'axe']
    states = {
            0: N(0, False),
            1: N(1, False),
            4: N(4, False),
            5: N(5, False),
            6: N(6, True),
            }
    transitions = {
            0: [
                T(0, 0, [False, None, False, False,], ['get_wood',]),
                T(0, 1, [True, None, False, False,], ['get_iron',]),
                T(0, 4, [None, False, True, False,], ['get_iron',]),
                T(0, 5, [None, True, True, False,], ['make0',]),
                T(0, 6, [None, None, None, True,], ['make0',]),
                ],
            1: [
                T(1, 1, [None, None, False, False,], ['make1']),
                T(1, 4, [None, False, True, False,], ['get_iron',]),
                T(1, 5, [None, True, True, False,], ['make0',]),
                T(1, 6, [None, None, None, True,], ['make0',]),
                ],
            4: [
                T(4, 4, [None, False, None, False,], ['get_iron',]),
                T(4, 5, [None, True, None, False,], ['make0',]),
                T(4, 6, [None, None, None, True,], ['make0',]),
                ],
            5: [
                T(5, 5, [None, None, None, False,], ['make0',]),
                T(5, 6, [None, None, None, True,], ['make0']),
                ],
            6: [
                T(6, 6, [None,] * 4, ['make0'])
                ],
            }

class PAxe(CraftWorldAutomatonDomainKnowledge):
    ap_io = ['wood', 'iron', 'stick', 'axe']
    states = {
            0: N(0, False),
            1: N(1, False),
            2: N(2, False),
            3: N(3, False),
            4: N(4, False),
            5: N(5, False),
            6: N(6, True),
            }
    transitions = {
            0: [
                T(0, 0, [False,] * 4, ['get_wood', 'get_iron']),
                T(0, 1, [True, False, False, False,], ['get_iron', 'make1']),
                T(0, 2, [False, True, False, False,], ['get_wood',]),
                T(0, 3, [True, True, False, False,], ['make1',]),
                T(0, 4, [None, False, True, False,], ['get_iron',]),
                T(0, 5, [None, True, True, False,], ['make0',]),
                T(0, 6, [None, None, None, True,], ['make0',]),
                ],
            1: [
                T(1, 1, [None, False, False, False,], ['get_iron', 'make1']),
                T(1, 3, [None, True, False, False,], ['make1']),
                T(1, 4, [None, False, True, False,], ['get_iron',]),
                T(1, 5, [None, True, True, False,], ['make0',]),
                T(1, 6, [None, None, None, True,], ['make0',]),
                ],
            2: [
                T(2, 2, [False, None, False, False,], ['get_wood',]),
                T(2, 3, [True, None, False, False,], ['make1',]),
                T(2, 5, [None, True, True, False,], ['make0',]),
                T(2, 6, [None, None, None, True,], ['make0',]),
                ],
            3: [
                T(3, 3, [None, None, False, False,], ['make1',]),
                T(3, 5, [None, None, True, False,], ['make0',]),
                T(3, 6, [None, None, None, True,], ['make0',]),
                ],
            4: [
                T(4, 4, [None, False, None, False,], ['get_iron',]),
                T(4, 5, [None, True, None, False,], ['make0',]),
                T(4, 6, [None, None, None, True,], ['make0',]),
                ],
            5: [
                T(5, 5, [None, None, None, False,], ['make0',]),
                T(5, 6, [None, None, None, True,], ['make0']),
                ],
            6: [
                T(6, 6, [None,] * 4, ['make0'])
                ],
            }

class Gem(CraftWorldAutomatonDomainKnowledge):
    ap_io = ['wood', 'iron', 'stick', 'axe', 'gem',]
    states = {
            0: N(0, False),
            1: N(1, False),
            4: N(4, False),
            5: N(5, False),
            6: N(6, False),
            7: N(7, True),
            }
    transitions = {
            0: [
                T(0, 0, [False, None, False, False, False,], ['get_wood',]),
                T(0, 1, [True, None, False, False, False,], ['get_iron',]),
                T(0, 4, [None, False, True, False, False,], ['get_iron',]),
                T(0, 5, [None, True, True, False, False,], ['make0',]),
                T(0, 6, [None, None, None, True, False,], ['make0',]),
                T(0, 7, [None, None, None, None, True,], ['make0',]),
                ],
            1: [
                T(1, 1, [None, None, False, False, False,], ['make1']),
                T(1, 4, [None, False, True, False, False,], ['get_iron',]),
                T(1, 5, [None, True, True, False, False,], ['make0',]),
                T(1, 6, [None, None, None, True, False,], ['make0',]),
                T(1, 7, [None, None, None, None, True,], ['make0',]),
                ],
            4: [
                T(4, 4, [None, False, None, False, False,], ['get_iron',]),
                T(4, 5, [None, True, None, False, False,], ['make0',]),
                T(4, 6, [None, None, None, True, False,], ['make0',]),
                T(4, 7, [None, None, None, None, True,], ['make0',]),
                ],
            5: [
                T(5, 5, [None, None, None, False, False,], ['make0',]),
                T(5, 6, [None, None, None, True, False,], ['make0']),
                T(5, 7, [None, None, None, None, True,], ['make0']),
                ],
            6: [
                T(6, 6, [None, None, None, None, False], ['make0']),
                T(6, 7, [None, None, None, None, True], ['make0'])
                ],
            7: [
                T(7, 7, [None,] * 5, ['make0']),
                ]
            }

class PGem(CraftWorldAutomatonDomainKnowledge):
    ap_io = ['wood', 'iron', 'stick', 'axe', 'gem']
    states = {
            0: N(0, False),
            1: N(1, False),
            2: N(2, False),
            3: N(3, False),
            4: N(4, False),
            5: N(5, False),
            6: N(6, False),
            7: N(7, True),
            }
    transitions = {
            0: [
                T(0, 0, [False,] * 5, ['get_wood', 'get_iron']),
                T(0, 1, [True, False, False, False, False,], ['get_iron', 'make1']),
                T(0, 2, [False, True, False, False, False,], ['get_wood',]),
                T(0, 3, [True, True, False, False, False,], ['make1',]),
                T(0, 4, [None, False, True, False, False,], ['get_iron',]),
                T(0, 5, [None, True, True, False, False,], ['make0',]),
                T(0, 6, [None, None, None, True, False,], ['make0',]),
                T(0, 7, [None, None, None, None, True,], ['make0',]),
                ],
            1: [
                T(1, 1, [None, False, False, False, False,], ['get_iron', 'make1']),
                T(1, 3, [None, True, False, False, False,], ['make1']),
                T(1, 4, [None, False, True, False, False,], ['get_iron',]),
                T(1, 5, [None, True, True, False, False,], ['make0',]),
                T(1, 6, [None, None, None, True, False,], ['make0',]),
                T(1, 7, [None, None, None, None, True,], ['make0',]),
                ],
            2: [
                T(2, 2, [False, None, False, False, False,], ['get_wood',]),
                T(2, 3, [True, None, False, False, False,], ['make1',]),
                T(2, 5, [None, True, True, False, False,], ['make0',]),
                T(2, 6, [None, None, None, True, False,], ['make0',]),
                T(2, 7, [None, None, None, None, True,], ['make0',]),
                ],
            3: [
                T(3, 3, [None, None, False, False, False,], ['make1',]),
                T(3, 5, [None, None, True, False, False,], ['make0',]),
                T(3, 6, [None, None, None, True, False,], ['make0',]),
                T(3, 7, [None, None, None, None, True,], ['make0',]),
                ],
            4: [
                T(4, 4, [None, False, None, False, False, None, None], ['get_iron',]),
                T(4, 5, [None, True, None, False, False, None, None], ['make0',]),
                T(4, 6, [None, None, None, True, False, None, None], ['make0',]),
                T(4, 7, [None, None, None, None, True, None, None], ['make0',]),
                ],
            5: [
                T(5, 5, [None, None, None, False, False,], ['make0',]),
                T(5, 6, [None, None, None, True, False,], ['make0']),
                T(5, 7, [None, None, None, None, True,], ['make0']),
                ],
            6: [
                T(6, 6, [None, None, None, None, False,], ['get_gem',]),
                T(6, 7, [None, None, None, None, True,], ['get_gem',]),
                ],
            7: [
                T(7, 7, [None,] * 5, ['get_gem',]),
                ],
            }

class Shears(CraftWorldAutomatonDomainKnowledge):
    ap_io = ['wood', 'iron', 'stick', 'shears']
    states = {
            0: N(0, False),
            1: N(1, False),
            4: N(4, False),
            5: N(5, False),
            6: N(6, True),
            }
    transitions = {
            0: [
                T(0, 0, [False, None, False, False], ['get_wood',]),
                T(0, 1, [True, None, False, False,], ['make1']),
                T(0, 4, [None, False, True, False,], ['get_iron',]),
                T(0, 5, [None, True, True, False,], ['make1',]),
                T(0, 6, [None, None, None, True,], ['make1',]),
                ],
            1: [
                T(1, 1, [None, None, False, False,], ['make1']),
                T(1, 4, [None, False, True, False,], ['get_iron',]),
                T(1, 5, [None, True, True, False,], ['make1',]),
                T(1, 6, [None, None, None, True,], ['make1',]),
                ],
            4: [
                T(4, 4, [None, False, None, False,], ['get_iron',]),
                T(4, 5, [None, True, None, False,], ['make1',]),
                T(4, 6, [None, None, None, True,], ['make1',]),
                ],
            5: [
                T(5, 5, [None, None, None, False,], ['make1',]),
                T(5, 6, [None, None, None, True,], ['make1']),
                ],
            6: [
                T(6, 6, [None,] * 4, ['make1'])
                ],
            }


class PShears(CraftWorldAutomatonDomainKnowledge):
    ap_io = ['wood', 'iron', 'stick', 'shears']
    states = {
            0: N(0, False),
            1: N(1, False),
            2: N(2, False),
            3: N(3, False),
            4: N(4, False),
            5: N(5, False),
            6: N(6, True),
            }
    transitions = {
            0: [
                T(0, 0, [False,] * 4, ['get_wood', 'get_iron']),
                T(0, 1, [True, False, False, False,], ['get_iron', 'make1']),
                T(0, 2, [False, True, False, False,], ['get_wood',]),
                T(0, 3, [True, True, False, False,], ['make1',]),
                T(0, 4, [None, False, True, False,], ['get_iron',]),
                T(0, 5, [None, True, True, False,], ['make1',]),
                T(0, 6, [None, None, None, True,], ['make1',]),
                ],
            1: [
                T(1, 1, [None, False, False, False,], ['get_iron', 'make1']),
                T(1, 3, [None, True, False, False,], ['make1']),
                T(1, 4, [None, False, True, False,], ['get_iron',]),
                T(1, 5, [None, True, True, False,], ['make1',]),
                T(1, 6, [None, None, None, True,], ['make1',]),
                ],
            2: [
                T(2, 2, [False, None, False, False,], ['get_wood',]),
                T(2, 3, [True, None, False, False,], ['make1',]),
                T(2, 5, [None, True, True, False,], ['make1',]),
                T(2, 6, [None, None, None, True,], ['make1',]),
                ],
            3: [
                T(3, 3, [None, None, False, False,], ['make1',]),
                T(3, 5, [None, None, True, False,], ['make1',]),
                T(3, 6, [None, None, None, True,], ['make1',]),
                ],
            4: [
                T(4, 4, [None, False, None, False,], ['get_iron',]),
                T(4, 5, [None, True, None, False,], ['make1',]),
                T(4, 6, [None, None, None, True,], ['make1',]),
                ],
            5: [
                T(5, 5, [None, None, None, False,], ['make1',]),
                T(5, 6, [None, None, None, True,], ['make1']),
                ],
            6: [
                T(6, 6, [None,] * 4, ['make1'])
                ],
            }

class PBridge(CraftWorldAutomatonDomainKnowledge):
    ap_io = ['wood', 'iron', 'bridge']
    states = {
            0: N(0, False),
            1: N(1, False),
            2: N(2, False),
            3: N(3, False),
            4: N(4, True),
            }
    
    # source_node.id -> [transition,]
    transitions = {
            0: [
                T(0, 0, [False, False, False,] , ['get_wood', 'get_iron',]),
                T(0, 1, [True, False, False,], ['get_iron']),
                T(0, 2, [False, True, False,], ['get_wood']),
                T(0, 3, [True, True, False,], ['make2']),
                T(0, 4, [None, None, True,], ['make2']),
                ],
            1: [
                T(1, 1, [None, False, False,], ['get_iron']),
                T(1, 3, [None, True, False,], ['make2']),
                T(1, 4, [None, None, True,], ['make2']),
                ],
            2: [
                T(2, 2, [False, None, False,], ['get_wood']),
                T(2, 3, [True, None, False,], ['make2']),
                T(2, 4, [None, None, True,], ['make2']),
                ],
            3: [
                T(3, 3, [None, None, False,], ['make2']),
                T(3, 4, [None, None, True,], ['make2']),
                ],

            4: [
                T(4, 4, [None, ] * 3, ['make2']),
                ],
            }

class Bridge(CraftWorldAutomatonDomainKnowledge):
    ap_io = ['wood', 'iron', 'bridge']
    states = {
            0: N(0, False),
            1: N(1, False),
            3: N(3, False),
            4: N(4, True),
            }
    
    # source_node.id -> [transition,]
    transitions = {
            0: [
                T(0, 0, [False, None, False,] , ['get_wood',]),
                T(0, 1, [True, None, False,], ['get_iron']),
                T(0, 3, [True, True, False,], ['make2']),
                T(0, 4, [None, None, True,], ['make2']),
                ],
            1: [
                T(1, 1, [None, False, False,], ['get_iron']),
                T(1, 3, [None, True, False,], ['make2']),
                T(1, 4, [None, None, True,], ['make2']),
                ],
            3: [
                T(3, 3, [None, None, False,], ['make2']),
                T(3, 4, [None, None, True,], ['make2']),
                ],

            4: [
                T(4, 4, [None, ] * 3, ['make2']),
                ],
            }

class Gold(CraftWorldAutomatonDomainKnowledge):
    ap_io = ['wood', 'iron', 'bridge', 'gold']
    states = {
            0: N(0, False),
            1: N(1, False),
            3: N(3, False),
            4: N(4, False),
            5: N(5, True),
            }
    
    # source_node.id -> [transition,]
    transitions = {
            0: [
                T(0, 0, [False, None, False, False,] , ['get_wood',]),
                T(0, 1, [True, False, False, False,], ['get_iron']),
                T(0, 3, [True, True, False, False,], ['make2']),
                T(0, 4, [None, None, True, False,], ['make2']),
                T(0, 5, [None, None, None, True,], ['make2']),
                ],
            1: [
                T(1, 1, [None, None, False, False,], ['get_iron']),
                T(1, 3, [None, True, False, False,], ['make2']),
                T(1, 4, [None, None, True, False,], ['make2']),
                T(1, 5, [None, None, None, True,], ['make2']),
                ],
            3: [
                T(3, 3, [None, None, False, False], ['make2']),
                T(3, 4, [None, None, True, False], ['make2']),
                T(3, 5, [None, None, None, True], ['make2']),
                ],
            4: [
                T(4, 4, [None, None, None, False,], ['make2']),
                T(4, 5, [None, None, None, True,], ['make2']),
                ],
            5: [
                T(5, 5, [None,]*4, ['make2']),
                ]
            }


class PGold(CraftWorldAutomatonDomainKnowledge):
    ap_io = ['wood', 'iron', 'bridge', 'gold']
    states = {
            0: N(0, False),
            1: N(1, False),
            2: N(2, False),
            3: N(3, False),
            4: N(4, False),
            5: N(5, True),
            }
    
    # source_node.id -> [transition,]
    transitions = {
            0: [
                T(0, 0, [False, False, False, False,] , ['get_wood', 'get_iron',]),
                T(0, 1, [True, False, False, False,], ['get_iron']),
                T(0, 2, [False, True, False, False,], ['get_wood']),
                T(0, 3, [True, True, False, False,], ['make2']),
                T(0, 4, [None, None, True, False,], ['make2']),
                T(0, 5, [None, None, None, True,], ['make2']),
                ],
            1: [
                T(1, 1, [None, False, False, False,], ['get_iron']),
                T(1, 3, [None, True, False, False,], ['make2']),
                T(1, 4, [None, None, True, False,], ['make2']),
                T(1, 5, [None, None, None, True,], ['make2']),
                ],
            2: [
                T(2, 2, [False, None, False, False,], ['get_wood']),
                T(2, 3, [True, None, False, False,], ['make2']),
                T(2, 4, [None, None, True, False,], ['make2']),
                T(2, 5, [None, None, None, True,], ['make2']),
                ],
            3: [
                T(3, 3, [None, None, False, False,], ['make2']),
                T(3, 4, [None, None, True, False,], ['make2']),
                T(3, 5, [None, None, None, True,], ['make2']),
                ],

            4: [
                T(4, 4, [None, None, None, None, False,], ['get_gold']),
                T(4, 5, [None, None, None, None, True,], ['get_gold']),
                ],
            5: [
                T(5, 5, [None,] * 4, ['get_gold']),
                ],
            }


class Stick(CraftWorldAutomatonDomainKnowledge):
    ap_io = ['wood', 'stick']
    states = {
            0: N(0, False),
            1: N(1, False),
            2: N(2, True),
            }

    # source_node.id -> [transition,]
    transitions = {
            0: [
                T(0, 0, [False, False,], ['get_wood',]),
                T(0, 1, [True, False,], ['make1',]),
                T(0, 2, [None, True,], ['make1',]),
                ],
            1: [
                T(1, 1, [None, False,], ['make1']),
                T(1, 2, [None, True,], ['make1']),
                ],
            2: [T(2, 2, [None, None,], ['make1',]),],
            }

class Rope(CraftWorldAutomatonDomainKnowledge):
    ap_io = ['grass', 'rope']
    states = {
            0: N(0, False),
            1: N(1, False),
            2: N(2, True),
            }

    # source_node.id -> [transition,]
    transitions = {
            0: [
                T(0, 0, [False, False,], ['get_grass',]),
                T(0, 1, [True, False,], ['make0',]),
                T(0, 2, [None, True,], ['make0',]),
                ],
            1: [
                T(1, 1, [None, False,], ['make0']),
                T(1, 2, [None, True,], ['make0']),
                ],
            2: [T(2, 2, [None, None,], ['make0',]),],
            }

class Cloth(CraftWorldAutomatonDomainKnowledge):
    ap_io = ['grass', 'cloth']
    states = {
            0: N(0, False),
            1: N(1, False),
            2: N(2, True),
            }

    # source_node.id -> [transition,]
    transitions = {
            0: [
                T(0, 0, [False, False,], ['get_grass',]),
                T(0, 1, [True, False,], ['make2',]),
                T(0, 2, [None, True,], ['make2',]),
                ],
            1: [
                T(1, 1, [None, False,], ['make2']),
                T(1, 2, [None, True,], ['make2']),
                ],
            2: [T(2, 2, [None, None,], ['make2',]),],
            }


class Plank(CraftWorldAutomatonDomainKnowledge):
    ap_io = ['wood', 'plank']
    states = {
            0: N(0, False),
            1: N(1, False),
            2: N(2, True),
            }

    # source_node.id -> [transition,]
    transitions = {
            0: [
                T(0, 0, [False, False,], ['get_wood',]),
                T(0, 1, [True, False,], ['make0',]),
                T(0, 2, [None, True,], ['make0',]),
                ],
            1: [
                T(1, 1, [None, False,], ['make0']),
                T(1, 2, [None, True,], ['make0']),
                ],
            2: [T(2, 2, [None, None,], ['make0',]),],
            }


class CraftWorldAutomatonDomainKnowledgeFactory():
    symbolic_actions = [
            'get_iron',
            'get_wood',
            'get_grass',
            'get_gold',
            'get_gem',
            'make0',
            'make1',
            'make2',
            ]
    goal_metacontroller_map = {
            'plank': Plank,
            'stick': Stick,
            'cloth': Cloth,
            'rope': Rope,
            'bridge': Bridge,
            'bed': Bed,
            'axe': Axe,
            'shears': Shears,
            'gold': Gold,
            'gem': Gem,
            }

    def make(goal, subgoals, cookbook, *args, **kwargs):
        goal = cookbook.index.get(goal)
        return CraftWorldAutomatonDomainKnowledgeFactory.goal_metacontroller_map[goal](goal, subgoals, cookbook)


class CraftWorldPAutomatonDomainKnowledgeFactory():
    symbolic_actions = [
            'get_iron',
            'get_wood',
            'get_grass',
            'get_gold',
            'get_gem',
            'make0',
            'make1',
            'make2',
            ]
    goal_metacontroller_map = {
            'plank': Plank,
            'stick': Stick,
            'cloth': Cloth,
            'rope': Rope,
            'bridge': PBridge,
            'bed': PBed,
            'axe': PAxe,
            'shears': PShears,
            'gold': PGold,
            'gem': PGem,
            }

    def make(goal, subgoals, cookbook, *args, **kwargs):
        goal = cookbook.index.get(goal)
        return CraftWorldPAutomatonDomainKnowledgeFactory.goal_metacontroller_map[goal](goal, subgoals, cookbook)

def domain_model(config):
    world = config.world.name
    world_models = {
            'LightWorld': LightWorldDomainKnowledge,
            'CraftWorld': CraftWorldDomainKnowledge,
            }
    #try:
    if 'parallel' in config.world and config.world.parallel:
        world_models = {
                # TODO FdH: implement factory for lightworld
#                'LightWorld': LightWorldParallelDomainKnowledgeFactory,
                'CraftWorld': CraftWorldPAutomatonDomainKnowledgeFactory,
                }
    else:
        world_models = {
                # TODO FdH: implement factory for lightworld
#                'LightWorld': LightWorldParallelDomainKnowledgeFactory,
                'CraftWorld': CraftWorldAutomatonDomainKnowledgeFactory,
                }
    #except:
   #    pass
    return world_models[world]

