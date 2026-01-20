import jax.numpy as jnp

from typing import Sequence, Union, Tuple
from jax.lax import max, reduce_window

def max_pool(
        inputs, 
        window_size: Sequence[int], 
        strides: Sequence[int]|None=None, 
        padding:Union[str, Sequence[Tuple[int, int]]]='VALID'):
    """max_pool is a wrapper around jax.lax.reduce_window

    Args:
        inputs (): 
        window_size: 
        strides: 
        padding: 

    Returns:
        
    """
    return reduce_window(
        inputs, 
        computation=max, 
        init_value=-jnp.inf, 
        window_dimensions=(1,)+tuple(window_size)+(1,), 
        window_strides=(1,)+tuple(strides)+(1,), 
        padding=padding)

def flatten(inputs):
    return inputs.reshape(inputs.shape[0], -1)

