import jax
import jax.random as jr
import jax.numpy as jnp

from typing import Callable
from dataclasses import dataclass,field
from jax.tree_util import register_dataclass
from jax.typing import ArrayLike

from ..activation import linear,relu
from .module import Module

@register_dataclass
@dataclass
class Linear(Module):
    weights : jax.Array
    bias : jax.Array
    num_in : int = field(metadata=dict(static=True))
    num_out : int = field(metadata=dict(static=True))
    activation : Callable[[jax.Array], jax.Array] = field(metadata=dict(static=True))

    def __init__(self, num_in:int, num_out:int, activation:Callable[[jax.Array], jax.Array]=linear, random_key:ArrayLike=jr.key(0)):
        self.num_in = num_in
        self.num_out = num_out
        self.activation = activation

        key_weights, key_bias = jr.split(random_key)

        self.weights = jr.uniform(key_weights, (num_in, num_out))
        self.bias = jr.uniform(key_bias, (num_out,))

    def __call__(self, x):
        return self.activation(jnp.dot(x, self.weights) + self.bias)
