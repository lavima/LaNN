import jax.numpy as jnp
import jax.random as jr

from jax import value_and_grad
from jax.tree import flatten, unflatten
from optax import adam, apply_updates
from optax.losses import sigmoid_binary_cross_entropy

from lann.activation import softmax, relu
from lann.modules import Sequence, iter_modules
from lann.modules.layers import Linear

random_key = jr.key(13)

random_linear1, random_linear2, random_x, random_y = jr.split(random_key, 4)

x = jr.normal(random_x, (100, 3))
y = jr.randint(random_y, (100, 1), minval=0, maxval=1).astype(jnp.float32)

print(x[0:5])
print(y[0:5])

model = Sequence([
    Linear(3, 2, activation=relu, random_key=random_linear1),
    Linear(2, 1, activation=softmax, random_key=random_linear2)])

# for module in iter_modules(model):
#     print(module)

params, treedef = flatten(model)
print(params)
print(treedef)

optimizer = adam(1e-1)
optimizer_state = optimizer.init(params)

def train_step(params, x, y):
    model = unflatten(treedef, params)
    loss_values = sigmoid_binary_cross_entropy(model(x), y)
    return jnp.mean(loss_values)

num_epochs = 10
for epoch in range(num_epochs):
    loss_value, gradients = value_and_grad(train_step)(params, x, y)
    updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
    params = apply_updates(params, updates)
    print(f'epoch {epoch}: loss: {loss_value:.4f}')
