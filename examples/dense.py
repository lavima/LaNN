import optax
import jax.numpy as jnp
import jax.random as jr

from lann.activation import softmax
from lann.modules import Dense, Sequence

random_key = jr.key(13)

random_key, random_split = jr.split(random_key)

x = jr.normal(random_key, (10, 3))
print(x)

model = Sequence([
    Dense(3, 4, random_key=random_key),
    Dense(4, 2, activation=softmax, random_key=random_key)])

optimizer = optax.adam(1e-1)
