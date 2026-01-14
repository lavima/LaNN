import logging
import jax
import jax.numpy as jnp

from .pytree import Pytree, static_field

logger = logging.getLogger(__name__)

def log_accuracy_update(num_correct, num_total, classification_shape, y_shape):
    logger.debug(f'num_correct: {num_correct} num_total: {num_total} classification_shape: {classification_shape} y_shape {y_shape}')


class Metric(Pytree):
    def update(self, logits, y):
        raise NotImplementedError()
    def compute(self):
        raise NotImplementedError()
    def reset(self):
        raise NotImplementedError()

class Accuracy(Metric):
    r"""
    The Accuracy metric is defined as follow.
    
    $$\mbox{NUM_CORRECT}/\mbox{NUM_TOTAL}

    If using for binary classification with a single output, the target $y$ must be of 
    shape $(\mbox{batch_size},1)$.
    """
    def __init__(self):
        self.reset()

    def update(self, logits, y):
        classification = jnp.where(logits>0.0, 1, 0)
        self.num_correct += jnp.sum(classification == y)
        self.num_total += logits.shape[0] 
        jax.debug.callback(log_accuracy_update, self.num_correct, self.num_total, classification.shape, y.shape)
        return self

    def compute(self):
        return self.num_correct/self.num_total
        
    def reset(self):
        self.num_total = jnp.zeros((), dtype=jnp.float32)
        self.num_correct = jnp.zeros((), dtype=jnp.int32)
        return self


