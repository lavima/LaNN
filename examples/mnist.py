import os
import logging
import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import jax.random as jr

from jax import jit, value_and_grad
from jax.lax import scan
from jax.tree import flatten
from jax.sharding import Mesh, PartitionSpec
from optax import adam, apply_updates
from optax.losses import sigmoid_binary_cross_entropy
from tensorflow.keras.datasets.mnist import load_data

from lann.metrics import Accuracy
from lann.models import Sequence
from lann.module import Dense, Conv, MaxPool, Flatten

# os.environ["JAX_PLATFORM_NAME"] = "cpu" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

P = PartitionSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random_key = jr.key(13)

random_key, random_linear1, random_conv1, random_conv2, random_conv3 = jr.split(random_key, 5)

(train_images, train_labels), (test_images, test_labels) = load_data()

# Normalize images to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape(train_images.shape + (1,))
test_images = test_images.reshape(test_images.shape + (1,))
train_labels = train_labels.reshape(-1, 1)
test_labels = test_labels.reshape(-1, 1)
print(test_images.shape)
print(test_labels.shape)


model = Sequence([
    Conv(num_channels_in=1, num_channels_out=1, window_size=(3, 3), random_key=random_conv1),
    MaxPool(),
    Conv(num_channels_in=1, num_channels_out=1, window_size=(3, 3), strides=(1, 1), random_key=random_conv2),
    MaxPool(),
    Flatten(),
    Dense(num_in=49, num_out=10, random_key=random_linear1)])

optimizer = adam(1e-3)

def batch_loss_and_eval(model, loss, metric, x, y):
    logits = model(x)
    loss_values = loss(logits, y)
    metric = metric.update(logits, y)
    return jnp.mean(loss_values), metric

def train_step(model, loss, metric, optimizer_state, x, y):
    (loss_value, metric), gradients = value_and_grad(batch_loss_and_eval, has_aux=True)(model, loss, metric, x, y)

    updates, optimizer_state = optimizer.update(gradients, optimizer_state, model)
    model = apply_updates(model, updates)

    return model, optimizer_state, loss_value, metric


def epoch_callback(epoch, loss, score):
    logger.info(f'epoch {epoch}: loss: {loss:.4f} score: {score:.4f}')

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
        score = metric.compute()
        jax.debug.callback(epoch_callback, epoch, average_loss, score)
        metric = metric.reset()

train(model, sigmoid_binary_cross_entropy, Accuracy(), optimizer, train_images, train_labels, num_epochs=200, batch_size=20)
