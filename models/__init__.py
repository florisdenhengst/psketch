from .reflex import ReflexModel
from .attentive import AttentiveModel
from .modular import ModularModel
from .ac import ActorCriticModel
from .modular_ac import ModularACModel
from .ma_mc import ModularActorModularCriticModel
from .keyboard import KeyboardModel

def load(config):
    cls_name = config.model.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        print('error loading model')
        raise Exception("No such model: {}".format(cls_name))
