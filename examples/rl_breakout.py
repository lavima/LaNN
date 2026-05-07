import logging
import jax
import jax.numpy as jnp
import jax.random as jr

from datetime import datetime
from tensorflow.summary import create_file_writer
from tensorboard import program
from optax import adam
from optax.losses import huber_loss

from lann.trainer import RLTrainer
from lann.reinforcement_learning import make_env
from lann.models import Sequence
from lann.modules import Linear, LinearActivation, Conv, MaxPool, Flatten
from lann.activation import relu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = 'logs/' + current_time + '/train'
summary_writer = create_file_writer(log_dir)
# tensorboard = program.TensorBoard()
# tensorboard.configure(argv=[None, '--logdir', log_dir, '--port', '6006'])

random_key = jr.key(13)

random_key, random_linear1, random_linear2, random_linear3, random_conv1, random_conv2, random_conv3= jr.split(random_key, 7)

environment = make_env('ALE/Breakout-v5', resolution=(84, 84))

model = Sequence([
    Conv(num_channels_in=4, num_channels_out=8, window_size=(3, 3), random_key=random_conv1),
    MaxPool(),
    Conv(num_channels_in=8, num_channels_out=16, window_size=(3, 3), random_key=random_conv2),
    MaxPool(),
    Conv(num_channels_in=16, num_channels_out=32, window_size=(3, 3), random_key=random_conv3),
    MaxPool(),
    Flatten(),
    LinearActivation(num_in=3200, num_out=196, activation=relu, random_key=random_linear1),
    LinearActivation(num_in=196, num_out=98, activation=relu, random_key=random_linear2),
    Linear(num_in=98, num_out=environment.action_space.n, random_key=random_linear3)])

# tensorboard_url = tensorboard.launch()
# print(f'Tensorboard started at {tensorboard_url}')

optimizer = adam(1e-3)
trainer = RLTrainer(environment, model, huber_loss, optimizer, summary_writer=summary_writer)
trainer.train()
