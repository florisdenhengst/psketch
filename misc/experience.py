from collections import namedtuple

# State, transducer state, model state, symbolic action, action, state', transducer state', model state', reward
Transition = namedtuple("Transition", ["s1", "m1", "sa1", "a", "s2", "m2", "r", "md"])
