import logging
import jax
import jax.random as jr
import jax.numpy as jnp

from typing import Any, Callable, Sequence, Union, Tuple
from functools import partial
from jax.typing import ArrayLike
from jax.lax import conv_general_dilated

from .typing import Shape
from .activation import linear
from .pytree import Pytree, static_field
from .functions import max_pool, flatten
from .initializers import Initializer, zeros, glorot_normal, he_normal

logger = logging.getLogger(__name__)

class Module(Pytree):

    """
    Module is the base class for all LaNN modules. Inherit from this to create modules, 
    where a module is a modular part of an architecture. All modules are Pytrees, so they
    can easily be used with JAX functions
    """
    def describe():
        print('test')

def iter_modules(module:Module):
    """iter_modules is an iterator for recursively traversing a module hierarchy

    Args:
        module: the module to travers 

    Yields: Traverses the hiearchy in a depth-first manner
        
    """
    yield module
    for name, value in vars(module).items():
        # print(f'{name} {type(value)} {isinstance(value, Module)}')
        if isinstance(value, Module):
            yield from iter_modules(value)
        elif isinstance(value, (list,set,tuple)):
            for item in value:
                if isinstance(item, Module):
                    yield from iter_modules(item)
    
class Linear(Module):
    """Dense is a fully connected linear combination with bias module

    Attributes:
        num_in: 
        num_out: 
        weights: 
        bias: 
    """
    num_in : int = static_field()
    num_out : int = static_field()
    init_weights : Callable[[jax.Array, Shape, Any|None], jax.Array] = static_field()
    init_bias : Callable[[jax.Array, Shape, Any|None], jax.Array] = static_field()

    def __init__(
            self, 
            *, 
            num_in:int, 
            num_out:int, 
            init_weights:Initializer=glorot_normal(), 
            init_bias:Initializer=zeros, 
            random_key:ArrayLike=jr.key(0)):
        """Construct a new Dense module

        Args:
            num_in: 
            num_out: 
            random_key: 
        """
        self.num_in = num_in
        self.num_out = num_out
        self.init_weights = init_weights
        self.init_bias = init_bias

        key_weights, key_bias = jr.split(random_key)

        self.weights = init_weights(key_weights, (num_in, num_out))
        self.bias = init_bias(key_bias, (num_out,))

    def __call__(self, x):
        return jnp.dot(x, self.weights) + self.bias

class Dense(Linear):
    activation : Callable[[jax.Array], jax.Array] = static_field()
    def __init__(self, activation: Callable[[jax.Array], jax.Array]=linear, **kwargs):
        super().__init__(**kwargs)
        self.activation = activation

    def __call__(self, x):
        return self.activation(super().__call__(x))

def log_conv_shape(input_shape, kernel_shape, output_shape):
    logger.debug(f'input_shape: {input_shape} kernel_shape: {kernel_shape} output_shape: {output_shape}')

class Conv(Module):
    """Conv is a convolution module. The dimension numbers used are
    ('NHWC', 'HWIO, 'NHWC') (see Jax.lax.conv_general_dilated documentation
    for details on meaning of each character). This ordering has been chosen
    partly because it is intuitive (NHWC) and partly because it makes the 
    implementation of the initializers easy (HWIO). I think it also matches 
    most popular JAX libraries.

    Would it be better to just speci    fy the kernel size directly?

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
    init_kernel : Initializer = static_field()

    def __init__(
            self, 
            *,
            num_channels_in:int, 
            num_channels_out:int, 
            window_size:Sequence[int]=(3, 3), 
            strides:Sequence[int]=(1, 1), 
            padding:Union[str, Sequence[Tuple[int, int]]]='SAME', 
            init_kernel:Initializer=he_normal(), 
            random_key:ArrayLike=jr.key(0)):
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out
        self.window_size = window_size
        self.strides = strides
        self.padding = padding
        self.init_kernel = init_kernel

        random_kernel = random_key
        self.kernel = init_kernel(random_kernel, tuple(window_size) + (num_channels_in, num_channels_out))
        # self.kernel = jr.normal(random_kernel, tuple(window_size) + (num_channels_in, num_channels_out))

    def __call__(self, inputs):
        outputs = conv_general_dilated(inputs, self.kernel, self.strides, self.padding, dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
        jax.debug.callback(log_conv_shape, inputs.shape, self.kernel.shape, outputs.shape)
        return outputs

class LSTM(Module):
    num_features_in:int = static_field()
    num_features_hidden:int = static_field()

    def __init__(self, num_features_in, num_features_hidden, random_key=jr.key(0)):
        self.num_features_in = num_features_in
        self.num_features_hidden = num_features_hidden

        linear_ih = partial(Linear, num_in=num_features_in, num_out=num_features_hidden)
        linear_hh = partial(Linear, num_in=num_features_hidden, num_out=num_features_hidden)



        

class MaxPool(Module):
    """MaxPool provides a module wrapper for lann.functions.max_pool

    Attributes:
        window_size: 
        strides: 
        padding: 
        window_size: 
        strides: 
        padding: 
    """
    window_size: Sequence[int] = static_field()
    strides: Sequence[int]|None = static_field() 
    padding: Union[str, Sequence[Tuple[int, int]]] = static_field()

    def __init__(self, window_size: Sequence[int]=(2, 2), strides: Sequence[int]=(2, 2), padding: Union[str, Sequence[Tuple[int, int]]]='Valid'):
        self.window_size = window_size
        self.strides = strides
        self.padding = padding

    def __call__(self, inputs):
        return max_pool(inputs, self.window_size, self.strides, self.padding)

class Flatten(Module):
    def __call__(self, inputs):
        return flatten(inputs)

# Pytree implementation inspired by equinox. I don't like the code generation. 
# Replaced for now by Pytree baseclass
#
# TODO It seems like it should be possible to do without the code generation.
#
# _FLATTEN_NAME = 'flatten'
# _UNFLATTEN_NAME = 'unflatten'
# _SOURCE_FLATTEN = '''
# def flatten(self):
#     return ({children}, {aux_data})
# '''
# _SOURCE_UNFLATTEN = '''
# def unflatten(cls, aux_data, children):
#     obj = object.__new__(cls)
#     {setters_children}
#     {setters_aux_data}
#     return obj
# '''
#
# def _generate_pytree_funs(cls:type, fields:tuple[Field[Any], ...]):
#      children = []
#     aux_data = []
#     for f in fields:
#         if f.metadata.get('static', False):
#             aux_data.append(f.name)
#         else:
#             children.append(f.name)
#
#     source_children = '()' if len(children)==0 else f'({', '.join(['self.' + a for a in children])},)'
#     source_aux_data = '()' if len(aux_data)==0 else f'({', '.join(['self.' + a for a in aux_data])},)'
#     source_flatten = _SOURCE_FLATTEN.format(name=_FLATTEN_NAME, children=source_children, aux_data=source_aux_data)
#     print(source_flatten)
#
#     source_setters_children = [a + b for a, b in zip(['obj.'+a for a in children], [f'=children[{a}]' for a in range(len(children))])]
#     source_setters_aux_data = [a + b for a, b in zip(['obj.'+a for a in aux_data], [f'=aux_data[{a}]' for a in range(len(aux_data))])]
#     source_unflatten = _SOURCE_UNFLATTEN.format(
#         setters_children='\n    '.join(source_setters_children), 
#         setters_aux_data='\n    '.join(source_setters_aux_data))
#     print(source_unflatten)
#
#     namespace = {}
#     exec(compile(source_flatten, _FLATTEN_NAME, 'exec'), namespace)
#     exec(compile(source_unflatten, _UNFLATTEN_NAME, 'exec'), namespace)
#     flatten_fun = namespace[_FLATTEN_NAME]
#     unflatten_fun = namespace[_UNFLATTEN_NAME]
#
#     return flatten_fun, unflatten_fun
#
#
# @dataclass_transform(field_specifiers=(Field, field))
# class ModuleMeta(ABCMeta):
#     def __new__(mcs, name, bases, dct):
#         cls = super().__new__(mcs, name, bases, dct)
#         cls = dataclass(init=False)(cls)
#         flatten_fun, unflatten_fun = _generate_pytree_funs(cls, dataclasses.fields(cls))
#         register_pytree_node(cls, flatten_fun, partial(unflatten_fun, cls))
#         return cls

