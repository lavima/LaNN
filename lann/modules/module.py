import dataclasses
import jax
import jax.random as jr

from functools import partial
from abc import ABCMeta
from dataclasses import Field, field, dataclass
from typing import dataclass_transform, Any
from jax.tree_util import register_pytree_node

from ..pytree import Pytree


# _FLATTEN_NAME = 'flatten'
# _UNFLATTEN_NAME = 'unflatten'
# _SOURCE_FLATTEN = '''
# def flatten(self):
#     return ({children}, {aux_data})
# '''
# _SOURCE_UNFLATTEN = '''
# def unflatten(cls, aux_data, children):
#     obj = object.__new__(cls)
#     {setters_children}
#     {setters_aux_data}
#     return obj
# '''
#
# def _generate_pytree_funs(cls:type, fields:tuple[Field[Any], ...]):
#     children = []
#     aux_data = []
#     for f in fields:
#         if f.metadata.get('static', False):
#             aux_data.append(f.name)
#         else:
#             children.append(f.name)
#
#     source_children = '()' if len(children)==0 else f'({', '.join(['self.' + a for a in children])},)'
#     source_aux_data = '()' if len(aux_data)==0 else f'({', '.join(['self.' + a for a in aux_data])},)'
#     source_flatten = _SOURCE_FLATTEN.format(name=_FLATTEN_NAME, children=source_children, aux_data=source_aux_data)
#     print(source_flatten)
#
#     source_setters_children = [a + b for a, b in zip(['obj.'+a for a in children], [f'=children[{a}]' for a in range(len(children))])]
#     source_setters_aux_data = [a + b for a, b in zip(['obj.'+a for a in aux_data], [f'=aux_data[{a}]' for a in range(len(aux_data))])]
#     source_unflatten = _SOURCE_UNFLATTEN.format(
#         setters_children='\n    '.join(source_setters_children), 
#         setters_aux_data='\n    '.join(source_setters_aux_data))
#     print(source_unflatten)
#
#     namespace = {}
#     exec(compile(source_flatten, _FLATTEN_NAME, 'exec'), namespace)
#     exec(compile(source_unflatten, _UNFLATTEN_NAME, 'exec'), namespace)
#     flatten_fun = namespace[_FLATTEN_NAME]
#     unflatten_fun = namespace[_UNFLATTEN_NAME]
#
#     return flatten_fun, unflatten_fun
#
#
# @dataclass_transform(field_specifiers=(Field, field))
# class ModuleMeta(ABCMeta):
#     def __new__(mcs, name, bases, dct):
#         cls = super().__new__(mcs, name, bases, dct)
#         cls = dataclass(init=False)(cls)
#         flatten_fun, unflatten_fun = _generate_pytree_funs(cls, dataclasses.fields(cls))
#         register_pytree_node(cls, flatten_fun, partial(unflatten_fun, cls))
#         return cls


# class Module(metaclass=ModuleMeta):
class Module(Pytree):

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
    
class Sequence(Module):
    # layers : list[Module]

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def add(self, layer):
        self.layers.append(layer)

