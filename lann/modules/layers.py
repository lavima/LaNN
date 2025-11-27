import jax
import jax.random as jr
import jax.numpy as jnp

from ..activation import relu
from .module import Module

class Dense(Module):
    def __init__(self, num_in, num_out, activation=relu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_in = num_in
        self.num_out = num_out
        self.activation = activation

        self.weights = jr.uniform(self.make_random_key(), (num_in, num_out))

    def __call__(self, x):
        return self.activation(jnp.dot(x, self.weights))
