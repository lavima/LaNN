import logging
import mlflow
import jax
import jax.numpy as jnp
import jax.random as jr

from datetime import datetime
from optax import adam
from optax.losses import huber_loss

from lann.trainer import RLTrainer
from lann.reinforcement_learning import make_env
from lann.models import Sequence
from lann.modules import Linear, LinearActivation, Conv, Relu, Flatten
from lann.activation import relu


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri('sqlite:///lann_examples.db')
mlflow.set_experiment('rl-breakout')

# current_time = datetime.now().strftime('%Y%m%d-%H%M%S')

random_key = jr.key(13)

random_key, random_linear1, random_linear2, random_linear3, random_conv1, random_conv2, random_conv3= jr.split(random_key, 7)

environment = make_env('ALE/Breakout-v5', resolution=(84, 84))

model = Sequence([
    Conv(num_channels_in=4, num_channels_out=32, window_size=(8, 8), strides=(4, 4), padding='VALID', random_key=random_conv1),
    Relu(),
    Conv(num_channels_in=32, num_channels_out=64, window_size=(4, 4), strides=(2, 2), padding='VALID', random_key=random_conv2),
    Relu(),
    Conv(num_channels_in=64, num_channels_out=64, window_size=(3, 3), strides=(1, 1), padding='VALID', random_key=random_conv3),
    Relu(),
    Flatten(),
    LinearActivation(num_in=3136, num_out=512, activation=relu, random_key=random_linear2),
    Linear(num_in=512, num_out=environment.action_space.n, random_key=random_linear3)])

def epoch_callback(model, epoch, loss, reward, steps):
    print(f'epoch {epoch} loss: {loss} reward: {reward} steps: {steps}')
    mlflow.log_metric('loss', loss, step=epoch)
    mlflow.log_metric('reward', reward, step=epoch)

learning_rate = 2e-4
optimizer = adam(learning_rate)
trainer = RLTrainer(environment, model, huber_loss, optimizer, update='hard', update_rate=10000, replay_buffer_size=200000, callbacks=[epoch_callback])

with mlflow.start_run():
    mlflow.log_param('learning-rate', learning_rate)
    trainer.train(num_epochs=10000)
