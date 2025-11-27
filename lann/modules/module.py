import jax
import jax.random as jr

class Module:
    def __init__(self, random_key=jr.key(0)):
        self._random_key = random_key

    def make_random_key(num=1):
        self._random_key, *keys = jr.split(num)
        return keys

    def describe():
        print('test')
