import logging
import jax
import jax.random as jr
import jax.numpy as jnp

from functools import reduce
from operator import mul
from typing import Any, Sequence, Literal, Protocol, runtime_checkable

from .typing import Shape

logger = logging.getLogger(__name__)

@runtime_checkable
class Initializer(Protocol):
    """
    Initializer protocol. Inspired by jax.nn.Initializer. Rewritten from 
    scratch to learn the python mechanics.
    """
    def __call__(self, random_key: jax.Array, shape: Shape, dtype: Any|None=None):
        raise NotImplementedError()

def zeros(random_key:jax.Array, shape:Shape, dtype:Any|None=None):
    return jnp.zeros(shape, dtype=dtype)

def scaled_variance(
        type: Literal['glorot'] | Literal['he'], 
        distribution: Literal['uniform'] | Literal['normal'] | Literal['truncated_normal'], 
        in_axis: Sequence[int]=-2, 
        out_axis: Sequence[int]=-1, 
        batch_axis: Sequence[int] = ()):
    r"""
    Returns an initializer that scales the variance of the specified distribution.

    For normal distribution the distribution is scaled by the standard deviation as 
    calculated for the specified type.
    
    For truncated normal distribution the standard deviation is adjusted to account 
    for the truncated values (the tails). The variation of a truncated normal 
    distribution can be calculated using the following equation. 

    ..  math::

        \operatorname{Var}(X \mid a<X<b) = \sigma^2\left[ 1 - \frac{\beta\varphi(\beta) - \alpha\varphi(\alpha)}{\Phi(\beta)-\Phi(\alpha)}
        -\left(\frac{\varphi(\beta) - \varphi(\alpha)}{\Phi(\beta)-\Phi(\alpha)}\right)^2\right]

    For unit distribution (mean 0 and std 1), :math:'\Alpha=a' and :math:'\Beta=b'.
    """

    def _compute_fans(shape):
        """
        Compute fan_in and fan_out for weights. The default values for the axis 
        indices work for linear combination and convolutional kernel weights under
        the assumption that the kernel ordering is HWIO. 

        The product of the sizes of all axes not included in neither in_axis, out_axis 
        nor batch_axis are multiplied with both the input and output size. For linear 
        combination weights this yields.

        $$\mbox{fan_in} = \mbox{num_in}$$
        $$\mbox{fan_out} = \mbox{num_out}$$

        For a convolutional kernel it yields

        $$\mbox{fan_in} = \mbox{num_in} * \mbox{kernel_height} * \mbox{kernel_width}$$
        $$\mbox{fan_out} = \mbox{num_out} * \mbox{kernel_height} * \mbox{kernel_width}$$
        """
        if isinstance(in_axis, int):
            num_in = shape[in_axis]
        else:
            num_in = reduce(mul, [shape[x] for x in in_axis])
        if isinstance(out_axis, int):
            num_out = shape[out_axis]
        else:
            num_out = reduce(mul, [shape[x] for x in out_axis])
        if isinstance(batch_axis, int):
            num_batch = shape[batch_axis]
        else:
            num_batch = reduce(mul, [shape[x] for x in batch_axis], 1)

        remaining = reduce(mul, shape)/num_in/num_out/num_batch
        
        fan_in = num_in * remaining
        fan_out = num_out * remaining
        
        return fan_in, fan_out

    def init(random_key:jax.Array, shape:Shape, dtype:Any|None=None) -> jax.Array:
        fan_in, fan_out = _compute_fans(shape)
        if type=='glorot':
            scale = 2.0/(fan_in+fan_out)
        elif type=='he':
            scale = 2.0/(fan_in)
        else:
            raise ValueError(f'Unknown variance scaler {type}')
            
        # TODO is this the best way to fallback
        if not dtype:
            dtype = jnp.float32

        if distribution == 'normal':
            return jr.normal(random_key, shape, dtype=dtype) * jnp.sqrt(scale)
        elif distribution == 'truncated_normal':
            # Scole the variance scale factor to account for the truncated variation 
            return jr.truncated_normal(random_key, -2, 2, shape, dtype=dtype) * (jnp.sqrt(scale) / jnp.array(0.87962566103423978, dtype=dtype))
        elif distribution == 'uniform':
            return jr.uniform(random_key, shape, dtype=dtype) * jnp.sqrt(3 * scale)
        else:
            raise ValueError(f'Unknown distribution {distribution}')

    return init

def glorot_normal(
        in_axis: Sequence[int]=-2, 
        out_axis: Sequence[int]=-1, 
        batch_axis: Sequence[int] = (),
        truncate: bool = False):
    return scaled_variance(type='glorot', distribution='normal' if not truncate else 'trancated_normal', in_axis=in_axis, out_axis=out_axis, batch_axis=batch_axis)

def glorot_uniform(
        in_axis: Sequence[int]=-2, 
        out_axis: Sequence[int]=-1, 
        batch_axis: Sequence[int] = ()):
    return scaled_variance(type='glorot', distribution='uniform', in_axis=in_axis, out_axis=out_axis, batch_axis=batch_axis)

def he_normal(
        in_axis: Sequence[int]=-2, 
        out_axis: Sequence[int]=-1, 
        batch_axis: Sequence[int] = (),
        truncate: bool = False):
    return scaled_variance(type='he', distribution='normal' if not truncate else 'truncated_normal', in_axis=in_axis, out_axis=out_axis, batch_axis=batch_axis)

def he_uniform(
        in_axis: Sequence[int]=-2, 
        out_axis: Sequence[int]=-1, 
        batch_axis: Sequence[int] = ()):
    return scaled_variance(type='he', distribution='uniform', in_axis=in_axis, out_axis=out_axis, batch_axis=batch_axis)
