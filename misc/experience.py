from collections import namedtuple

# State, model state, action, state', model state', reward
Transition = namedtuple("Transition", ["s1", "m1", "a", "s2", "m2", "r"])
