import logging
import jax
import jax.numpy as jnp

from typing import Sequence, Literal
from jax.tree import map

from .pytree import Pytree, static_field

logger = logging.getLogger(__name__)


class Metric(Pytree):
    """
    Base class for all metrics. Internal states must be non-static to pass 
    Pytree boundary."""

    def __init__(self):
        self.reset()

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

def log_metrics_init(metrics):
    logger.debug(f'Metrics.__init__: metrics: {metrics}')

def log_metrics_reset(metrics):
    logger.debug(f'Metrics.reset: metrics: {metrics}')

class Metrics(Metric):
    def __init__(self, metrics: Sequence[Metric]):
        jax.debug.callback(log_metrics_init, metrics)
        self.metrics = list(metrics)
        super().__init__()

    def update(self, logits, y):
        self.metrics = [x.update(logits, y) for x in self.metrics]
        return self
    
    def compute(self):
        scores = [x.compute() for x in self.metrics]
        return {name: value for score in scores for name, value in score.items()}

    def reset(self):
        jax.debug.callback(log_metrics_init, self.metrics)
        self.metrics = [x.reset() for x in self.metrics]
        return self
    

class _SummedState(Metric):
    def reduce(self):
        return jax.tree.map(lambda x: jnp.sum(x, axis=0), self)

    def merge(self, other):
        return jax.tree.map(lambda x, y: x + y, self, other)


def _log_accuracy_update(num_correct, num_total, classification_shape, y_shape):
    logger.debug(f'num_correct: {num_correct} num_total: {num_total} classification_shape: {classification_shape} y_shape {y_shape}')

class Accuracy(_SummedState):
    r"""
    The Accuracy metric is defined as follow.
    
    $$\mbox{NUM_CORRECT}/\mbox{NUM_TOTAL}

    If using for binary classification with a single output, the target $y$ must be of 
    shape $(\mbox{batch_size},1)$. 
    """
    # TODO test if making the output layer (n,) instead of (n,1) fixes this

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
        return {'accuracy': self.num_correct/self.num_total }
        
    def reset(self):
        self.num_total = jnp.zeros((), dtype=jnp.float32)
        self.num_correct = jnp.zeros((), dtype=jnp.int32)
        return self

def _log_precisionrecallfmeasure_update(true_positives, false_positives, false_negatives):
    logger.debug(f'PrecisionRecallFMeasure.update: true_positives: {true_positives} false_positives: {false_positives} false_negatives: {false_negatives}')

class PrecisionRecallFMeasure(_SummedState):
    """
    A combined metric class for precision, recall and f measure
    """

    num_classes : int = static_field()
    average : Literal['micro']|Literal['macro'] = static_field() 
    beta : float = static_field() 

    def __init__(self, num_classes=2, average:Literal['micro']|Literal['macro']='micro', beta:float=1):
        self.num_classes = num_classes
        self.average = average
        self.beta = beta
        super().__init__()

    def update(self, logits, y):
        if logits.shape[-1] == 1:
            classification = jnp.where(logits>0.0, 1, 0)
        else:
            classification = jnp.argmax(logits, axis=1) 

        def _compute_class_stats(c):
            correct = classification == y
            incorrect = ~correct
            classification_is_c = classification == c
            classification_not_c = ~classification_is_c

            tp = jnp.sum(correct & classification_is_c, dtype=jnp.float32)
            fp = jnp.sum(incorrect & classification_is_c, dtype=jnp.float32)
            fn = jnp.sum(incorrect & classification_not_c, dtype=jnp.float32)

            return tp, fp, fn

        if self.num_classes==2:
            tp, fp, fn = _compute_class_stats(1)
            self.true_positives = tp
            self.false_positives = fp
            self.false_negatives = fn
        elif self.num_classes>2 and self.average=='micro':
            correct = classification == y
            incorrect = ~correct
            self.true_positives += jnp.sum(correct, dtype=jnp.float32)
            self.false_positives += jnp.sum(incorrect, dtype=jnp.float32)
            self.false_negatives += jnp.sum(incorrect, dtype=jnp.float32)
        elif self.num_classes>2 and self.average=='macro':
            classes = jnp.arange(self.num_classes)
            tps, fps, fns = jax.vmap(_compute_class_stats)(classes)
            self.true_positives = tps
            self.false_positives = fps
            self.false_negatives = fns
        else:
            raise ValueError(f'Incompatible values for num_classes and/or average. Got ({self.num_classes}, {self.average})')

        jax.debug.callback(_log_precisionrecallfmeasure_update, self.true_positives, self.false_positives, self.false_negatives)
        return self

    def reset(self):
        if self.num_classes==2 or self.average=='micro':
            self.true_positives = jnp.zeros((), dtype=jnp.float32)
            self.false_positives = jnp.zeros((), dtype=jnp.float32)
            self.false_negatives = jnp.zeros((), dtype=jnp.float32)
        elif self.num_classes>2 and self.average=='macro':
            self.true_positives = jnp.zeros((self.num_classes,), dtype=jnp.float32)
            self.false_positives = jnp.zeros((self.num_classes,), dtype=jnp.float32)
            self.false_negatives = jnp.zeros((self.num_classes,), dtype=jnp.float32)
        else:
            raise ValueError(f'Incompatible values for num_classes and/or average. Got ({self.num_classes}, {self.average})')
        return self

    def compute(self):
        if self.num_classes==2 or self.average=='micro':
            precision = self.true_positives/(self.true_positives + self.false_positives)
            recall = self.true_positives/(self.true_positives + self.false_negatives)
            fmeasure = ((1 + self.beta**2)*precision*recall)/(self.beta*precision + recall)
            return {
                'precision': precision,
                'recall': recall,
                'fmeasure': fmeasure
            }
        elif self.num_classes>2 and self.average=='macro':
            precisions = self.true_positives/(self.true_positives + self.false_positives)
            recalls = self.true_positives/(self.true_positives + self.false_negatives)
            fmeasures = ((1 + self.beta**2)*precisions*recalls)/(self.beta*precisions + recalls)
            return {
                'precision': jnp.mean(precisions),
                'recall': jnp.mean(recalls),
                'fmeasure': jnp.mean(fmeasures)
            }
        else:
            raise ValueError(f'Incompatible values for num_classes and/or average. Got ({self.num_classes}, {self.average})')
