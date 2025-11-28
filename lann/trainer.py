import jax
import optax

class Trainer:
    def __init__(self, optimizer=optax.adam, model=None, loss=None, num_epochs=5):
        if model==None:
            raise ValueError('model must be set')
        if loss==None:
            raise ValueError('loss must be set')
        self.optimizer = optimizer
        self.model = model
        self.loss = loss
        self.num_epochs = num_epochs

    def _train_step(self, epoch):
        print(f'epoch {epoch}/{self.num_epochs}')

