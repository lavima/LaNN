import jax
import jax.random as jr

from dataclasses import dataclass
from jax.tree_util import register_dataclass


class Variable:
    def __init__(self, value):
        self.value = value

@register_dataclass
@dataclass
class Module():
    # def __init__(self):

    # def __init_subclass__(cls, **kwargs):
    #     super().__init_subclass__(**kwargs)
    #     register_dataclass(cls)

    # def tree_flatten(self):
    #     children = []
    #     aux = []
    #
    #     for var in vars(self):
    #         if isinstance(var, (list, set, dict, Module)):
    #             children.append(var)
    #         else:
    #             aux.append(var)
    #
    #     return (children, aux)
    #
    # @classmethod
    # def tree_unflatten(cls, aux_data, children):
    #     module = object.__new__(cls)
    #
    #     for var in vars(module):
    #         if isinstance(var, (list, set, dict, Module)):
    #
    #     return module 

    def describe():
        print('test')

def iter_modules(module:Module):
    yield module
    for name, value in vars(module).items():
        # print(f'{name} {type(value)} {isinstance(value, Module)}')
        if isinstance(value, Module):
            yield from iter_modules(value)
        elif isinstance(value, (list,set,tuple)):
            for item in value:
                if isinstance(item, Module):
                    yield from iter_modules(item)
    
@register_dataclass
@dataclass
class Sequence(Module):
    layers : list[Module]

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def add(self, layer):
        self.layers.append(layer)

