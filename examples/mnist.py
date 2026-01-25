import os
import logging
import jax
import jax.numpy as jnp
import jax.random as jr

from jax import jit, value_and_grad
from jax.lax import scan
from jax.tree import flatten
from jax.sharding import Mesh, PartitionSpec
from optax import adam, apply_updates
from optax.losses import softmax_cross_entropy_with_integer_labels

from lann.datasets import load_mnist
from lann.metrics import PrecisionRecallFMeasure, Accuracy, Metrics
from lann.models import Sequence
from lann.module import Dense, Conv, MaxPool, Flatten
from lann.activation import relu, sigmoid

P = PartitionSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random_key = jr.key(13)

random_key, random_linear1, random_linear2, random_linear3, random_conv1, random_conv2 = jr.split(random_key, 6)

(train_images, train_labels), (test_images, test_labels) = load_mnist()
# train_labels = train_labels.astype(jnp.int32)
# test_labels = test_labels.astype(jnp.int32)

print(test_images.shape)
print(test_labels.shape)


model = Sequence([
    Conv(num_channels_in=1, num_channels_out=8, window_size=(3, 3), random_key=random_conv1),
    MaxPool(),
    Conv(num_channels_in=8, num_channels_out=8, window_size=(3, 3), random_key=random_conv2),
    MaxPool(),
    Flatten(),
    Dense(num_in=392, num_out=196, activation=relu, random_key=random_linear1),
    Dense(num_in=196, num_out=98, activation=relu, random_key=random_linear2),
    Dense(num_in=98, num_out=10, random_key=random_linear3)])

# model = Sequence([
#     Conv(num_channels_in=1, num_channels_out=4, window_size=(3, 3), random_key=random_conv1),
#     MaxPool(),
#     Conv(num_channels_in=4, num_channels_out=4, window_size=(3, 3), strides=(1, 1), random_key=random_conv2),
#     MaxPool(),
#     Flatten(),
#     Dense(num_in=196, num_out=98, activation=sigmoid, random_key=random_linear2),
#     Dense(num_in=98, num_out=10, random_key=random_linear3)])

# params, treedef = flatten(model)
# print(treedef)
# print(params)

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

metric = Metrics([Accuracy(), PrecisionRecallFMeasure(num_classes=10, average='macro')])

train(model, softmax_cross_entropy_with_integer_labels, metric, optimizer, train_images, train_labels, num_epochs=100, batch_size=1000)
