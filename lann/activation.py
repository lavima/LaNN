import jax
import jax.numpy as jnp

def sigmoid(x):
    # 1/(1+e^x)
    jax.lax.logistic(x)

def softmax(x):
    # ensures numerical stability by making the values range (-inf, 0]
    x = x - jnp.max(x, axis=-1, keepdims=True)
    return jnp.exp(x) / jnp.sum(jnp.exp(x), axis=-1, keepdims=True)
    
def relu(x):
    return jnp.maximum(x, 0)

def linear(x):
    return x
