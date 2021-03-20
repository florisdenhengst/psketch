from worlds import light
from worlds import craft

class LightWorldDomainKnowledge():
    def __init__(self, goals):
        self.prev_action_label = None
        self.prev_observation = None
        # a sequence of symbolic high-level subgoals
        self.goals = goals
        self.current_goal_i = 0
    
    def tick(self, observation, action):
        subgoal_met = False
        #last_goal = self.current_goal_i == len(self.goals)
        if self.prev_action_label != None:
            subgoal_met = self.subgoal_met(observation)
            if subgoal_met:
                self.current_goal_i = min(len(self.goals), self.current_goal_i + 1)
        self.prev_observation = observation
        self.prev_action_label = action
        return subgoal_met #and not last_goal

    def in_door(self, observation):
        return observation[:4].sum() == 4.0

    def subgoal_met(self, observation):
        if not self.in_door(self.prev_observation) or self.in_door(observation):
            return False
        elif self.goals[self.current_goal_i] == 'l' and self.prev_action_label == light.LEFT:
            return True
        elif self.goals[self.current_goal_i] == 'u' and self.prev_action_label == light.UP:
            return True
        elif self.goals[self.current_goal_i] == 'd' and self.prev_action_label == light.DOWN:
            return True
        elif self.goals[self.current_goal_i] == 'r' and self.prev_action_label == light.RIGHT:
            return True
        return False

def domain_model(config):
    world = config.world.name
    world_models = {
            'LightWorld': LightWorldDomainKnowledge,
            }
    return world_models[world]

