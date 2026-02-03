import logging
import jax
import jax.numpy as jnp
import jax.random as jr

from jax import jit, value_and_grad
from jax.lax import scan
from jax.tree import flatten
from optax import adam, apply_updates
from optax.losses import softmax_cross_entropy_with_integer_labels

from lann.metrics import PrecisionRecallFMeasure, Accuracy, Metrics
from lann.models import Sequence
from lann.modules import LSTMCell, RNN
from lann.activation import relu, sigmoid


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random_key = jr.key(13)

random_key, random_linear1, random_linear2, random_linear3, random_conv1, random_conv2 = jr.split(random_key, 6)

random_key, random_x, random_y = jr.split(random_key, 3)

x = jr.normal(random_x, (1000, 100, 10))
y = jr.normal(random_y, (1000,))

model = Sequence([
    RNN(LSTMCell(num_features_in=10, num_features_hidden=10)),
    RNN(LSTMCell(num_features_in=10, num_features_hidden=10))
    ])

params, treedef = flatten(model)
print(treedef)
print(params)

optimizer = adam(1e-3)

def log_batch_loss_and_eval(x, y, logits):
    logger.debug(f'x.shape: {x.shape} y.shape: {y.shape} logits.shape: {logits.shape}')

def log_epoch(epoch, loss, scores):
    score_strings = [f'{name}: {value:.4f}' for name, value in scores.items()]
    logger.info(f'epoch {epoch}: loss: {loss:.4f} '+', '.join(score_strings))

def batch_loss_and_eval(model, loss, metric, x, y):
    logits = model(x)
    jax.debug.callback(log_batch_loss_and_eval, x, y, logits)
    loss_values = loss(logits, y)
    metric = metric.update(logits, y)
    return jnp.mean(loss_values), metric

def train_step(model, loss, metric, optimizer_state, x, y):
    (loss_value, metric), gradients = value_and_grad(batch_loss_and_eval, has_aux=True)(model, loss, metric, x, y)

    updates, optimizer_state = optimizer.update(gradients, optimizer_state, model)
    model = apply_updates(model, updates)

    return model, optimizer_state, loss_value, metric


def train(model, loss, metric, optimizer, x, y, num_epochs=10, batch_size=10, random_key=jr.key(0)):
    def _train_step(state, batch_indices):
        model, metric, optimizer_state = state
        model, optimizer_state, loss_value, metric = train_step(model, loss, metric, optimizer_state, x[batch_indices], y[batch_indices])
        return (model, metric, optimizer_state), loss_value

    @jit
    def _epoch(model, metric, optimizer_state, random_key):
        indices = jr.permutation(random_key, x_shape[0])
        batched_indices = indices.reshape(-1, batch_size)

        (model, metric, optimizer_state), accumulated_loss = scan(_train_step, (model, metric, optimizer_state), batched_indices)

        return model, optimizer_state, jnp.mean(accumulated_loss), metric

    x = jnp.array(x)
    y = jnp.array(y)

    x_shape = x.shape

    optimizer_state = optimizer.init(model)
    for epoch in range(num_epochs):
        random_key, random_epoch = jr.split(random_key, 2)
        model, optimizer_state, average_loss, metric = _epoch(model, metric, optimizer_state, random_epoch)
        scores = metric.compute()
        jax.debug.callback(log_epoch, epoch, average_loss, scores)
        metric = metric.reset()

    return model

metric = Metrics([Accuracy(), PrecisionRecallFMeasure(num_classes=10, average='macro')])

train(model, softmax_cross_entropy_with_integer_labels, metric, optimizer, x, y, num_epochs=100, batch_size=100)
