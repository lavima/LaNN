import optax
import jax.numpy as jnp
import jax.random as jr

from lann.activation import softmax
from lann.modules import Sequence, iter_modules
from lann.modules.layers import Linear

random_key = jr.key(13)

random_key, random_split = jr.split(random_key)

x = jr.normal(random_key, (10, 3))
print(x)

model = Sequence([
    Linear(3, 4, random_key=random_key),
    Linear(4, 2, activation=softmax, random_key=random_key)])

for module in iter_modules(model):
    print(module)

optimizer = optax.adam(1e-1)
