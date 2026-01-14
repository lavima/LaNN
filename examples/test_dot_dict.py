import jax
from jax.tree import flatten, unflatten

from lann.utils import dot_dict
from lann.metrics import accuracy

x = dot_dict({'metric_fun':accuracy, 'state':jax.Array()})
leaves, treedef = flatten(x)
print(leaves)
print(treedef)
