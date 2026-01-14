import jax.numpy as jnp
import jax.random as jr

from jax import jit, value_and_grad
from jax.lax import scan
from jax.tree import flatten
from optax import adam, apply_updates
from optax.losses import sigmoid_binary_cross_entropy
from sklearn.datasets import make_classification

from lann.activation import linear, relu
from lann.modules import Sequence
from lann.modules.layers import Linear

random_key = jr.key(13)

random_key, random_linear1, random_linear2, random_linear3, random_x, random_y = jr.split(random_key, 6)

x, y = make_classification(n_samples=10000, n_features=20, n_informative=5)
y = (y > 0).astype(jnp.float32)
x = jnp.array(x)
y = jnp.array(y)

print(x[0:5])
print(y[0:5])

model = Sequence([
    Linear(20, 10, activation=relu, random_key=random_linear1),
    Linear(10, 5, activation=relu, random_key=random_linear2),
    Linear(5, 1, activation=linear, random_key=random_linear3)])

children, treedef = flatten(model)
print(children)
print(treedef)

# for module in iter_modules(model):
#     print(module)

optimizer = adam(1e-4)

def batch_loss(model, x, y):
    loss_values = sigmoid_binary_cross_entropy(model(x), y)
    return jnp.mean(loss_values)

def train_step(model, optimizer_state, x, y):
    loss_value, gradients = value_and_grad(batch_loss)(model, x, y)

    updates, optimizer_state = optimizer.update(gradients, optimizer_state, model)
    model = apply_updates(model, updates)

    return model, optimizer_state, loss_value


def train(model, optimizer, x, y, num_epochs=10, batch_size=10, random_key=jr.key(0)):
    def _train_step(state, batch_indices):
        model, optimizer_state = state
        model, optimizer_state, loss_value = train_step(model, optimizer_state, x[batch_indices], y[batch_indices])
        return (model, optimizer_state), loss_value

    @jit
    def _epoch(epoch, model, optimizer_state, random_key):
        indices = jr.permutation(random_key, x_shape[0])
        batched_indices = indices.reshape(-1, batch_size)
        (model, optimizer_state), accumulated_loss = scan(_train_step, (model, optimizer_state), batched_indices)

        return model, optimizer_state, jnp.mean(accumulated_loss)

    x_shape = x.shape
    optimizer_state = optimizer.init(model)
    for epoch in range(num_epochs):
        random_key, random_epoch = jr.split(random_key, 2)
        model, optimizer_state, average_loss = _epoch(epoch, model, optimizer_state, random_epoch)
        print(f'epoch {epoch}: loss: {average_loss:.4f}')

train(model, optimizer, x, y, num_epochs=100)
