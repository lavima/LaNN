import logging
import jax
import jax.numpy as jnp
import jax.random as jr

from jax import jit, value_and_grad
from jax.lax import scan
from jax.tree import flatten
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from optax import adam, apply_updates
from optax.losses import sigmoid_binary_cross_entropy
from sklearn.datasets import make_classification

from lann.metrics import Accuracy
from lann.activation import linear, relu
from lann.modules import Sequence
from lann.modules.layers import Dense

P = PartitionSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random_key = jr.key(13)

random_key, random_linear1, random_linear2, random_linear3, random_x, random_y = jr.split(random_key, 6)

x, y = make_classification(n_samples=10000, n_features=20, n_informative=5)
y = (y > 0).astype(jnp.float32)
x = jnp.array(x)
y = jnp.array(y).reshape(y.shape[0], 1)

model = Sequence([
    Dense(20, 10, activation=relu, random_key=random_linear1),
    Dense(10, 5, activation=relu, random_key=random_linear2),
    Dense(5, 1, activation=linear, random_key=random_linear3)])

# children, treedef = flatten(model)
# print(children)
# print(treedef)

# for module in iter_modules(model):
#     print(module)

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
        x_batch, y_batch = x[batch_indices], y[batch_indices]

        x_sharded = jax.device_put(x_batch, sharding)
        y_sharded = jax.device_put(y_batch, sharding)

        model, optimizer_state, loss_value, metric = train_step(model, loss, metric, optimizer_state, x_sharded, y_sharded)

        return (model, metric, optimizer_state), loss_value

    @jit
    def _epoch(model, metric, optimizer_state, random_key):
        indices = jr.permutation(random_key, x_shape[0])
        batched_indices = indices.reshape(-1, batch_size)

        (model, metric, optimizer_state), accumulated_loss = scan(_train_step, (model, metric, optimizer_state), batched_indices)

        return model, optimizer_state, jnp.mean(accumulated_loss), metric

    mesh = jax.make_mesh((2,), axis_names=('batches',))
    sharding = NamedSharding(mesh, P('batches', None))
    x_shape = x.shape
    optimizer_state = optimizer.init(model)
    for epoch in range(num_epochs):
        random_key, random_epoch = jr.split(random_key, 2)
        model, optimizer_state, average_loss, metric = _epoch(model, metric, optimizer_state, random_epoch)
        score = metric.compute()
        jax.debug.callback(epoch_callback, epoch, average_loss, score)
        metric = metric.reset()

train(model, sigmoid_binary_cross_entropy, Accuracy(), optimizer, x, y, num_epochs=200, batch_size=200)
