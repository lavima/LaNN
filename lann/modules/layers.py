import jax
import jax.random as jr
import jax.numpy as jnp

from ..activation import relu
from .module import Module

class Linear(Module):
    def __init__(self, num_in, num_out, activation=relu, random_key=jr.key(0)):
        self.num_in = num_in
        self.num_out = num_out
        self.activation = activation

        key_weights, key_bias = jr.split(random_key)

        self.weights = jr.uniform(key_weights, (num_in, num_out))
        self.bias = jr.uniform(key_bias, (num_out,))

    def __call__(self, x):
        return self.activation(jnp.dot(x, self.weights) + self.bias)
