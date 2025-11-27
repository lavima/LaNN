import jax
import jax.random as jr

class Module:
    def __init__(self, random_key=jr.key(0)):
        self._random_key = random_key

    def make_random_key(self, num=1):
        if num==1:
            self._random_key, key = jr.split(self._random_key, num=num+1)
            return key
        else:
            self._random_key, *keys = jr.split(self._random_key, num=num+1)
            return keys

    def describe():
        print('test')

class Sequence(Module):
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def add(self, layer):
        self.layers.append(layer)

