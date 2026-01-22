import logging
import jax
import jax.numpy as jnp

from .pytree import Pytree, static_field

logger = logging.getLogger(__name__)


class Metric(Pytree):
    """Base class for all metrics. Internal states must be non-static to pass Pytree boundary."""
    def update(self, logits, y):
        """
        Update the state of the metric.
        """
        # TODO Change to named arguments needs to be considered, but for now, this works
        raise NotImplementedError()
    def compute(self):
        """Compute the metrics from the accumulated states

        Raises:
            NotImplementedError: 
        """
        raise NotImplementedError()
    def reset(self):
        """Reset the metric states

        Raises:
            NotImplementedError: 
        """
        raise NotImplementedError()
    def reduce(self):
        """Reduce the device states

        Raises:
            NotImplementedError: 
        """
        raise NotImplementedError()
    def merge(self, other):
        """Merge two metrics

        Args:
            other (): 

        Raises:
            NotImplementedError: 
        """
        raise NotImplementedError()
    def batch_update(self, *args):
        """
        Batch update. Used for manual sharding.
            *args: 

        Returns:
            
        """
        return self.reset().update(*args)


def _log_accuracy_update(num_correct, num_total, classification_shape, y_shape):
    logger.debug(f'num_correct: {num_correct} num_total: {num_total} classification_shape: {classification_shape} y_shape {y_shape}')

class Accuracy(Metric):
    r"""
    The Accuracy metric is defined as follow.
    
    $$\mbox{NUM_CORRECT}/\mbox{NUM_TOTAL}

    If using for binary classification with a single output, the target $y$ must be of 
    shape $(\mbox{batch_size},1)$. 
    """
    # TODO test if making the output layer (n,) instead of (n,1) fixes this

    def __init__(self):
        self.reset()

    def update(self, logits, y):
        """
        Update the counts according to given logits and labels

        Args:
            logits (): 
            y (): 

        Returns: A new copy with updated states
            
        """
        # If the logits have a single output neuron, we consider it binary classification
        if logits.shape[-1] == 1:
            classification = jnp.where(logits>0.0, 1, 0)
        else:
            classification = jnp.argmax(logits, axis=1) 
        self.num_correct += jnp.sum(classification == y)
        self.num_total += logits.shape[0] 

        jax.debug.callback(_log_accuracy_update, self.num_correct, self.num_total, classification.shape, y.shape)

        return self

    def compute(self):
        return self.num_correct/self.num_total
        
    def reset(self):
        self.num_total = jnp.zeros((), dtype=jnp.float32)
        self.num_correct = jnp.zeros((), dtype=jnp.int32)
        return self

    def reduce(self):
        return jax.tree.map(lambda x: jnp.sum(x, axis=0), self)

    def merge(self, other):
        return jax.tree.map(lambda x, y: x + y, self, other)



