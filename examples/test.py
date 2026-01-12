import jax.numpy as jnp
import jax.random as jr

from jax import jit, value_and_grad
from jax.lax import scan
from jax.tree import flatten, unflatten, leaves
from optax import adam, apply_updates
from optax.losses import sigmoid_binary_cross_entropy
from sklearn.datasets import make_classification

from lann.activation import linear, relu
from lann.modules import Sequence
from lann.modules.layers import Linear

random_key = jr.key(13)

random_key, random_linear1, random_linear2, random_x, random_y = jr.split(random_key, 5)

# x = jr.normal(random_x, (100, 3))
# y = jr.randint(random_y, (100, 1), minval=0, maxval=1).astype(jnp.float32)
x, y = make_classification(n_samples=10000, n_features=20)
y = (y > 0).astype(jnp.float32)
x = jnp.array(x)
y = jnp.array(y)

print(x[0:5])
print(y[0:5])

model = Sequence([
    Linear(20, 10, activation=relu, random_key=random_linear1),
    Linear(10, 5, activation=relu, random_key=random_linear1),
    Linear(5, 1, activation=linear, random_key=random_linear2)])

# for module in iter_modules(model):
#     print(module)

params, treedef = flatten(model)
print(treedef)
print(params)

optimizer = adam(1e-1)

def batch_loss(params, x, y):
    model = unflatten(treedef, params)
    loss_values = sigmoid_binary_cross_entropy(model(x), y)
    return jnp.mean(loss_values)

@jit
def train_step(params, optimizer_state, x, y):
    loss_value, gradients = value_and_grad(batch_loss)(params, x, y)

    updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
    params = apply_updates(params, updates)

    return params, optimizer_state, loss_value


def train(model, optimizer, x, y, num_epochs=100, batch_size=5, random_key=jr.key(0)):
    def _train_step(state, batch_indices):
        params, optimizer_state = state
        params, optimizer_state, loss_value = train_step(params, optimizer_state, x[batch_indices], y[batch_indices])
        return (params, optimizer_state), loss_value

    @jit
    def _epoch(epoch, params, optimizer_state, random_key):
        indices = jr.permutation(random_key, x_shape[0])
        batched_indices = indices.reshape(-1, batch_size)
        (params, optimizer_state), accumulated_loss = scan(_train_step, (params, optimizer_state), batched_indices)

        return params, optimizer_state, jnp.mean(accumulated_loss)

    x_shape = x.shape
    params, treedef = flatten(model)
    optimizer_state = optimizer.init(params)
    for epoch in range(num_epochs):
        random_key, random_epoch = jr.split(random_key, 2)
        params, optimizer_state, average_loss = _epoch(epoch, params, optimizer_state, random_epoch)
        print(f'epoch {epoch}: loss: {average_loss:.4f}')

train(model, optimizer, x, y)
