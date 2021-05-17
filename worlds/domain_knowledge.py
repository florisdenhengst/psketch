from worlds import light
from worlds import craft

import logging

class LightWorldDomainKnowledge():
    def __init__(self, goals, cookbook):
        self.prev_action_label = None
        self.prev_observation = None
        # a sequence of symbolic high-level subgoals
        self.goals = goals
        self.current_goal_i = 0
    
    def advance(self):
        self.current_goal_i = min(len(self.goals), self.current_goal_i + 1)
    
    def tick(self, state, action):
        observation = state.features()
        subgoal_met = False
        #last_goal = self.current_goal_i == len(self.goals)
        if self.prev_action_label != None:
            subgoal_met = self.subgoal_met(observation)
            if subgoal_met:
                self.current_goal_i = min(len(self.goals) - 1, self.current_goal_i + 1)
        self.prev_observation = observation
        self.prev_action_label = action
        return subgoal_met #and not last_goal

    def in_door(self, observation):
        return observation[:4].sum() == 4.0

    def subgoal_met(self, observation):
        if not self.in_door(self.prev_observation):
            return False
        elif self.goals[self.current_goal_i] == 'left':
            return True
        elif self.goals[self.current_goal_i] == 'up':
            return True
        elif self.goals[self.current_goal_i] == 'down':
            return True
        elif self.goals[self.current_goal_i] == 'right':
            return True
        return False

class CraftWorldDomainKnowledge():
    def __init__(self, goals, cookbook):
        self.prev_action_label = None
        self.prev_observation = None
        # a sequence of symbolic high-level subgoals
        self.goals = goals
        self.current_goal_i = 0
        self.cookbook = cookbook

    def advance(self):
        self.current_goal_i = min(len(self.goals), self.current_goal_i + 1)

    def tick(self, state, action):
        subgoal_met = self.subgoal_met(state, action)
        #last_goal = self.current_goal_i == len(self.goals)
        if subgoal_met:
            self.current_goal_i = min(len(self.goals), self.current_goal_i + 1)
        self.prev_state = state
        self.prev_action_label = action
        return subgoal_met #and not last_goal

    def check_inventory(self, state, keyword):
        # TODO FdH:
        #  * Change > 0.0 to == 1.0?
        return state.inventory[self.cookbook.index[keyword]] > 0.0

    def subgoal_met(self, state, action):
        subgoal = self.goals[self.current_goal_i]
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

def domain_model(config):
    world = config.world.name
    world_models = {
            'LightWorld': LightWorldDomainKnowledge,
            'CraftWorld': CraftWorldDomainKnowledge,
            }
    return world_models[world]

