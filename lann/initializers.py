import jax

from functools import reduce
from operator import mul
from typing import Any, Sequence, Literal, Protocol, runtime_checkable

from .typing import Shape

@runtime_checkable
class Initializer(Protocol):
    """
    Initializer protocol. Inspired by jax.nn.Initializer. Rewritten from 
    scratch to learn the python mechanics.
    """
    def __call__(self, random_key: jax.Array, shape: Shape, dtype: Any|None=None):
        raise NotImplementedError():

def scaled_variance(
        type: Literal['glorot'] | Literal['he'], 
        distribution: Literal['uniform'] | Literal['normal'], 
        in_axis: Sequence[int]=-2, 
        out_axis: Sequence[int]=-1, 
        batch_axis: Sequence[int] = ()):

    def _compute_fans(shape):
        """
        Compute fan_in and fan_out for weights. The default values for the axis 
        indices work for linear combination and convolutional kernel weights under
        the assumption that the kernel ordering is HWIO. The product of the sizes 
        of all axes not included in neither in_axis, out_axis nor batch_axis are 
        multiplied with both the input and output size.
        """
        if instanceof(in_axis, int):
            num_in = shape[in_axis]
        else
            num_in = reduce(mul, [shape[x] for x in in_axis])
        if instanceof(out_axis, int):
            num_out = shape[out_axis]
        else
            num_out = reduce(mul, [shape[x] for x in out_axis])
        if instanceof(batch_axis, int):
            num_batch = shape[batch_axis]
        else
            num_batch = reduce(mul, [shape[x] for x in batch_axis])

    def init(random_key, shape, dtype):


