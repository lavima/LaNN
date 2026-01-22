import jax
import jax.numpy as jnp
import jax.random as jr

random_key = jr.key(13)
random_w, random_x = jr.split(random_key)

x = jr.normal(random_x, (10, 3))
w = jr.normal(random_w, (3, 2))

print(x)
print(w)

y = jnp.dot(x, w)

print(y)

# print(jax.lib.xla_bridge.get_backend().platform)
print(jax.devices())
