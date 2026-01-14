import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

from dataclasses import dataclass
from jax.tree_util import register_dataclass

def accuracy(logits, y):
    classification = jnp.argmax(logits)
    num_correct = classification == y
    return num_correct/len(logits)

class Metric:
    def __init__(self, metric_fun, state):
        self.metric_fun = metric_fun
        self.state = state

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree_node_class(cls)
        # cls.tree_flatten = Metric.tree_flatten
        # cls.tree_unflatten = Metric.tree_unflatten


    def update(self, logits, y):
        raise NotImplementedError()
    def compute(self):
        raise NotImplementedError()
    def reset(self):
        raise NotImplementedError()

class Accuracy(Metric):
    def __init__(self):
        super().__init__(accuracy, jax.Array(2))

    def update(self, logits, y):
        classification = jnp.argmax(logits)
        self.state[0] += jnp.sum(classification == y)

    def compute(self):
        raise NotImplementedError()
    def reset(self):
        raise NotImplementedError()


