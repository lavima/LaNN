import jax
import jax.numpy as jnp

def relu(x):
    return jnp.maximum(x, 0)
