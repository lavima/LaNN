import logging
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

def log_sigmoid(x):
    logger.debug(f'sigmoid x.shape: {x.shape}')

def sigmoid(x):
    # 1/(1+e^x)
    jax.debug.callback(log_sigmoid, x)
    return jax.lax.logistic(x)

def softmax(x):
    # ensures numerical stability by making the values range (-inf, 0]
    x = x - jnp.max(x, axis=-1, keepdims=True)
    return jnp.exp(x) / jnp.sum(jnp.exp(x), axis=-1, keepdims=True)

def tanh(x):
    return jax.lax.tanh(x)
    
def relu(x):
    return jnp.maximum(x, 0)

def linear(x):
    return x
