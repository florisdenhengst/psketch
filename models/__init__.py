from .reflex import ReflexModel
from .attentive import AttentiveModel
from .modular import ModularModel
from .modular_ac import ModularACModel
from .keyboard import KeyboardModel

def load(config):
    print('Loading Model')
    cls_name = config.model.name
    print(cls_name)
    print(globals()[cls_name])
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        print('error loading model')
        raise Exception("No such model: {}".format(cls_name))
