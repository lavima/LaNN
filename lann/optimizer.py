import jax
import optax

class Optimizer:
    def __init__(self, algorithm=optax.adam, model=None, loss=None):
        if model==None:
            raise ValueError('model must be set')
        if loss==None:
            raise ValueError('loss must be set')
        self.algorithm = algorithm
        self.model = model
        self.loss = loss
