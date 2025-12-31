import jax
import jax.random as jr

from lann.utils import dot_dict

class Variable:
    def __init__(self, value):
        self.value = value


@jax.tree_util.register_pytree_node_class
class Module():
    # def __init__(self):

    def tree_flatten(self):
        children = []
        aux = []

        for var in vars(self):
            if isinstance(var, (list, set, dict, Module)):
                children.append(var)
            else:
                aux.append(var)

        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

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
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def add(self, layer):
        self.layers.append(layer)

