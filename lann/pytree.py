import typing as tp

from types import MappingProxyType
from dataclasses import field, Field, MISSING
from functools import partial
from abc import ABCMeta
from jax.tree_util import register_pytree_node

def static_field(
    *,
    default: tp.Any = MISSING,
    default_factory: tp.Any = MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
    metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None):

    if metadata is None:
        metadata = dict(static=True)
    else:
        metadata['static'] = True
    
    return field(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata)
    
class PytreeMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        obj = object.__new__(cls, *args, **kwargs)
        try:
            obj.__init__(*args, **kwargs)
        finally:
            pass

        obj_vars = vars(obj)
        obj._pytree_fields = tuple(field for field in obj_vars if field not in obj._pytree_static_fields)
        
        return obj

class Pytree(metaclass=PytreeMeta):
    _pytree_fields = tp.Tuple[str, ...]
    _pytree_static_fields = tp.Tuple[str, ...]
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        static_fields = set()
        class_vars = vars(cls)
        for field, value in class_vars.items():
            if isinstance(value, Field) and value.metadata.get("static", False):
                static_fields.add(field)
        static_fields.add('_pytree_fields')
        static_fields = tuple(static_fields)
        cls._pytree_static_fields = static_fields
        register_pytree_node(cls, cls._flatten, cls._unflatten, flatten_with_keys_func=partial(cls._flatten, with_keys=True))

    @classmethod
    def _flatten(cls, pytree, with_keys=False):
        all_vars = vars(pytree).copy()
        if with_keys:
            children = ((field, all_vars[field]) for field in pytree._pytree_fields)
        else:
            children = (all_vars[field] for field in pytree._pytree_fields)
        aux_data = {field: all_vars[field] for field in pytree._pytree_static_fields}
        return children, MappingProxyType(aux_data)
    
    @classmethod
    def _unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.__dict__.update(aux_data)
        obj.__dict__.update(zip(aux_data['_pytree_fields'], children))
        return obj



