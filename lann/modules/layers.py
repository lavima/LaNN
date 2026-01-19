import jax
import jax.random as jr
import jax.numpy as jnp

from typing import Callable, Sequence, Union, Tuple
from jax.typing import ArrayLike
from jax.lax import conv_general_dilated

from ..pytree import static_field
from ..activation import linear,relu
from .module import Module

class Dense(Module):
    """Dense is a fully connected linear combination with bias module, with an optional activation function

    Attributes:
        num_in: 
        num_out: 
        activation: 
        weights: 
        bias: 
    """
    num_in : int = static_field()
    num_out : int = static_field()
    activation : Callable[[jax.Array], jax.Array] = static_field()

    def __init__(self, num_in:int, num_out:int, activation:Callable[[jax.Array], jax.Array]=linear, random_key:ArrayLike=jr.key(0)):
        """Construct a new Dense module

        Args:
            num_in: 
            num_out: 
            activation: 
            random_key: 
        """
        self.num_in = num_in
        self.num_out = num_out
        self.activation = activation

        key_weights, key_bias = jr.split(random_key)

        self.weights = jr.uniform(key_weights, (num_in, num_out))
        self.bias = jr.uniform(key_bias, (num_out,))

    def __call__(self, x):
        return self.activation(jnp.dot(x, self.weights) + self.bias)

class Conv(Module):
    """Conv is a convolution module. The dimension numbers used are
    ('NHWC', 'IOHW, 'NHWC') (see Jax.lax.conv_general_dilated documentation
    for more details). 

    Would it be better to just specify the kernel size directly?

    Attributes:
        num_channels_in: 
        num_channels_out: 
        window_size: 
        strides: 
        padding: 
        kernel: 
    """
    num_channels_in : int = static_field()
    num_channels_out : int = static_field()
    window_size : Sequence[int] = static_field()
    strides : Sequence[int] = static_field()
    padding : Union[str, Sequence[Tuple[int, int]]] = static_field()

    def __init__(
            self, 
            num_channels_in:int, 
            num_channels_out:int, 
            window_size:Sequence[int], 
            strides:Sequence[int], 
            padding:Union[str, 
            Sequence[Tuple[int, int]]], 
            random_key:ArrayLike=jr.key(0)):
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out
        self.window_size = window_size
        self.strides = strides
        self.padding = padding

        random_kernel = random_key
        self.kernel = jr.uniform(random_kernel, (num_channels_in, num_channels_out) + tuple(window_size))

    def __call__(self, inputs):
        return conv_general_dilated(inputs, self.kernel, self.strides, self.padding, dimension_numbers=('NHWC', 'IOHW', 'NHWC'))

class MaxPooling(Module):
    window_size : Sequence[int] = static_field()

