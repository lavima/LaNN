import logging
import jax
import jax.numpy as jnp
import jax.random as jr

from optax import adam
from optax.losses import sigmoid_binary_cross_entropy
from sklearn.datasets import make_classification

from lann.metrics import Accuracy
from lann.activation import linear, relu
from lann.models import Sequence
from lann.modules import LinearActivation
from lann.trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random_key = jr.key(13)

random_key, random_linear1, random_linear2, random_linear3, random_x, random_y = jr.split(random_key, 6)

x, y = make_classification(n_samples=10000, n_features=20, n_informative=5)
y = (y > 0).astype(jnp.float32)
x = jnp.array(x)
y = jnp.array(y).reshape(y.shape[0], 1)

model = Sequence([
    LinearActivation(num_in=20, num_out=10, activation=relu, random_key=random_linear1),
    LinearActivation(num_in=10, num_out=5, activation=relu, random_key=random_linear2),
    LinearActivation(num_in=5, num_out=1, activation=linear, random_key=random_linear3)])

optimizer = adam(1e-3)

trainer = Trainer(model, sigmoid_binary_cross_entropy, Accuracy(), optimizer)
trainer.train(x, y, num_epochs=200, batch_size=200)
