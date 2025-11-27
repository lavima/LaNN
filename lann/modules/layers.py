import jax
import jax.random as jr
import jax.numpy as jnp

from activation import relu
from module import Module

class Dense(Module):
    def __init__(self, num, activation=relu):
        self.num = num
        self.activation = activation

        self.weights = jr.uniform(make_random_key(), num)

