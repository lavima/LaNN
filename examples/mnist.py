import logging
import jax
import jax.numpy as jnp
import jax.random as jr

from jax.tree import flatten
from optax import adam
from optax.losses import softmax_cross_entropy_with_integer_labels

from lann.datasets import load_mnist
from lann.metrics import PrecisionRecallFMeasure, Accuracy, Metrics
from lann.models import Sequence
from lann.modules import Linear, LinearActivation, Conv, MaxPool, Flatten
from lann.activation import relu, sigmoid
from lann.trainer import Trainer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random_key = jr.key(13)

random_key, random_linear1, random_linear2, random_linear3, random_conv1, random_conv2 = jr.split(random_key, 6)

(train_images, train_labels), (test_images, test_labels) = load_mnist()
# train_labels = train_labels.astype(jnp.int32)
# test_labels = test_labels.astype(jnp.int32)

print(test_images.shape)
print(test_labels.shape)

model = Sequence([
    Conv(num_channels_in=1, num_channels_out=8, window_size=(3, 3), random_key=random_conv1),
    MaxPool(),
    Conv(num_channels_in=8, num_channels_out=8, window_size=(3, 3), random_key=random_conv2),
    MaxPool(),
    Flatten(),
    LinearActivation(num_in=392, num_out=196, activation=relu, random_key=random_linear1),
    LinearActivation(num_in=196, num_out=98, activation=relu, random_key=random_linear2),
    Linear(num_in=98, num_out=10, random_key=random_linear3)])

params, treedef = flatten(model)
print(treedef)
print(params)

optimizer = adam(1e-3)

metric = Metrics([Accuracy(), PrecisionRecallFMeasure(num_classes=10, average='macro')])

trainer = Trainer(model, softmax_cross_entropy_with_integer_labels, metric, optimizer)
trainer.train(train_images, train_labels, num_epochs=10, batch_size=100)
