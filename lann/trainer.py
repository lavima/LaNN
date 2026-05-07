import logging
import jax
import jax.random as jr
import jax.numpy as jnp
import gymnasium as gym

from copy import deepcopy
from random import random, sample
from collections import deque
from itertools import count
from tensorflow.summary import scalar 
from jax import jit, value_and_grad
from jax.lax import scan
from jax.tree_util import tree_map
from optax import apply_updates

from lann.reinforcement_learning import Transition

logger = logging.getLogger(__name__)

class Trainer:

    def __init__(self, model, loss, metric, optimizer):
        self.model = model
        self.loss = loss
        self.metric = metric
        self.optimizer = optimizer


    def train(self, x, y, num_epochs=10, batch_size=10, random_key=jr.key(0)):

        def _batch_loss_and_eval(model, loss, metric, x, y):
            logits = model(x)
            jax.debug.callback(_log_batch_loss_and_eval, x, y, logits)
            loss_values = loss(logits, y)
            metric = metric.update(logits, y)
            return jnp.mean(loss_values), metric

        def _train_step(model, loss, metric, optimizer_state, x, y):
            (loss_value, metric), gradients = value_and_grad(_batch_loss_and_eval, has_aux=True)(model, loss, metric, x, y)

            updates, optimizer_state = self.optimizer.update(gradients, optimizer_state, model)
            model = apply_updates(model, updates)

            return model, optimizer_state, loss_value, metric

        def _train_step_wrapper(state, batch_indices):
            model, metric, optimizer_state = state
            model, optimizer_state, loss_value, metric = _train_step(model, self.loss, metric, optimizer_state, x[batch_indices], y[batch_indices])
            return (model, metric, optimizer_state), loss_value

        @jit
        def _epoch(model, metric, optimizer_state, random_key):
            indices = jr.permutation(random_key, x_shape[0])
            batched_indices = indices.reshape(-1, batch_size)

            (model, metric, optimizer_state), accumulated_loss = scan(_train_step_wrapper, (model, metric, optimizer_state), batched_indices)

            return model, optimizer_state, jnp.mean(accumulated_loss), metric

        x = jnp.array(x)
        y = jnp.array(y)

        x_shape = x.shape

        optimizer_state = self.optimizer.init(self.model)
        for epoch in range(num_epochs):
            random_key, random_epoch = jr.split(random_key, 2)
            self.model, optimizer_state, average_loss, self.metric = _epoch(self.model, self.metric, optimizer_state, random_epoch)
            scores = self.metric.compute()
            jax.debug.callback(_log_epoch, epoch, average_loss, scores)
            self.metric = self.metric.reset()

        return self.model

class RLTrainer:

    def __init__(self, environment, model, loss, optimizer, replay_buffer_size=1000000, update_rate=0.01, reward_when_truncated=True, summary_writer=None):
        self.environment = environment
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.replay_buffer_size = replay_buffer_size
        self.update_rate = update_rate
        self.reward_when_truncated = reward_when_truncated
        self.summary_writer = summary_writer

    def train(self, num_epochs=10, batch_size=32, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_rate=1e-3, discount_factor=0.99):

        def _batch_loss_and_eval(model, loss, x, y):
            logits = model(x)
            loss_values = loss(logits, y)
            return jnp.mean(loss_values)

        replay_buffer = deque([], self.replay_buffer_size)
        target_model = deepcopy(self.model)
        
        optimizer_state = self.optimizer.init(self.model)

        epsilon_counter = 0

        for epoch in range(num_epochs):
            state, _ = self.environment.reset()
            state = jnp.array(state, dtype=jnp.float32).transpose(1, 2, 0)

            for step_count in count():

                epsilon = max(epsilon_end, epsilon_start - epsilon_counter * epsilon_decay_rate)
                epsilon_counter += 1
                if epsilon > random():
                    action = jnp.array(self.environment.action_space.sample())
                else:
                    action = jnp.argmax(self.model(state[None, ...]), axis=1)[0]
                    

                observation, reward, terminated, truncated, _ = self.environment.step(action)
                reward = jnp.clip(reward, -1.0, 1.0)

                next_state = jnp.array(observation, dtype=jnp.float32).transpose(1, 2, 0)

                done = terminated if self.reward_when_truncated else terminated or truncated 
                
                replay_buffer.append(Transition(state, action, next_state, reward, done))

                state = next_state

                if len(replay_buffer) < batch_size:
                    continue

                transitions = sample(replay_buffer, batch_size) 
                batched_transitions = tree_map(lambda *xs: jnp.stack(xs), *transitions)


                quality_values = self.model(batched_transitions.state)
                quality_values = jax.vmap(lambda v, a: v[a], in_axes=(0,0))(quality_values, batched_transitions.action)

                next_quality_values = jnp.max(target_model(batched_transitions.next_state), axis=1)

                targets = batched_transitions.reward + (jnp.invert(batched_transitions.done) * discount_factor * next_quality_values)

                loss_value, gradients = value_and_grad(_batch_loss_and_eval)(self.model, self.loss, batched_transitions.state, targets[:, None])
                updates, optimizer_state = self.optimizer.update(gradients, optimizer_state, self.model)
                model = apply_updates(self.model, updates)

                tree_map(lambda a, b: a*(1-self.update_rate) + b*self.update_rate, target_model, self.model)

                if terminated or truncated:
                    break

            # if self.summary_writer is not None:
            #     with self.summary_writer.as_default():
            #         scalar('loss', loss_value, step=epoch)


def _log_batch_loss_and_eval(x, y, logits):
    logger.debug(f'x.shape: {x.shape} y.shape: {y.shape} logits.shape: {logits.shape}')

def _log_epoch(epoch, loss, scores):
    score_strings = [f'{name}: {value:.4f}' for name, value in scores.items()]
    logger.info(f'epoch {epoch}: loss: {loss:.4f} '+', '.join(score_strings))
        

