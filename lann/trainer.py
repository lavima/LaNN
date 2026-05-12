import logging
import jax
import numpy as np
import jax.random as jr
import jax.numpy as jnp
import gymnasium as gym

from copy import deepcopy
from random import random, sample
from collections import deque
from itertools import count
from jax import jit, value_and_grad
from jax.lax import scan
from jax.tree_util import tree_map
from optax import apply_updates

from lann.reinforcement_learning import Transition

logger = logging.getLogger(__name__)

class _TrainerBase:

    def __init__(self, callbacks=None):
        self._callbacks = callbacks

class Trainer(_TrainerBase):

    def __init__(self, model, loss, metric, optimizer, callbacks=None):
        super().__init__(callbacks)
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

class RLTrainer(_TrainerBase):

    def __init__(self, environment, model, loss, optimizer, replay_buffer_size=1000000, update='soft', update_rate=0.00005, reward_when_truncated=True, summary_writer=None, callbacks=None):
        super().__init__(callbacks)
        self.environment = environment
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.replay_buffer_size = replay_buffer_size

        if update == 'soft' and type(update_rate) is not float:
            raise ValueError('update_rate must be float when using soft updates')
        elif update == 'hard' and type(update_rate) is not int:
            raise ValueError('update_rate must be int when using hard updates')
        elif update not in ['soft', 'hard']:
            raise ValueError('unknown update type')

        self.update = update
        self.update_rate = update_rate
        self.reward_when_truncated = reward_when_truncated
        self.summary_writer = summary_writer

    def train(self, num_epochs=100, batch_size=32, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_rate=1e-6, discount_factor=0.99):

        def _select_action(state):
            epsilon = max(epsilon_end, epsilon_start - total_step_counter * epsilon_decay_rate)
            if epsilon > random():
                action = np.array(self.environment.action_space.sample())
            else:
                action = np.argmax(self.model(state[None, ...]), axis=1)[0]
            return action

        @jax.jit
        def _train_step(model, target_model, optimizer_state, transition_batch):
            def _batch_loss(model, states, actions, targets):
                all_quality_values = model(states)
                quality_values = jnp.take_along_axis(all_quality_values, actions[:, None], axis=1).squeeze()
                loss_values = self.loss(targets, quality_values)
                return jnp.mean(loss_values)

            next_quality_values = jnp.max(target_model(transition_batch.next_state), axis=1)

            targets = transition_batch.reward + ((1.0 - transition_batch.done) * discount_factor * next_quality_values)

            loss_value, gradients = value_and_grad(_batch_loss)(model, transition_batch.state, transition_batch.action, targets)
            updates, optimizer_state = self.optimizer.update(gradients, optimizer_state, model)
            model = apply_updates(model, updates)

            return model, optimizer_state, loss_value


        replay_buffer = deque([], self.replay_buffer_size)
        target_model = tree_map(jnp.array, self.model)
        
        optimizer_state = self.optimizer.init(self.model)

        total_step_counter = 0
        loss_value = 0.0


        for epoch in range(num_epochs):
            state, _ = self.environment.reset()
            num_lives = self.environment.unwrapped.ale.lives()
            state = np.array(state, dtype=np.float32).transpose(1, 2, 0) / 255.0

            total_reward = 0

            for step_count in count():

                action = _select_action(state) 
                observation, reward, terminated, truncated, info = self.environment.step(action)
                total_step_counter += 1

                reward = np.clip(reward, -1.0, 1.0)
                total_reward += reward

                next_state = np.array(observation, dtype=np.float32).transpose(1, 2, 0) / 255.0

                if info['lives'] < num_lives:
                    done = np.array(1.0, dtype=np.float32)
                    num_lives = info['lives']
                else:
                    done = np.array(terminated or truncated if self.reward_when_truncated else terminated, dtype=np.float32)
                
                replay_buffer.append(Transition(state, action, next_state, reward, done))

                state = next_state

                if len(replay_buffer) < batch_size:
                    continue

                transitions = sample(replay_buffer, batch_size) 
                transition_batch = tree_map(lambda *xs: np.stack(xs), *transitions)

                self.model, optimizer_state, loss_value = _train_step(self.model, target_model, optimizer_state, transition_batch)

                if self.update == 'soft':
                    target_model = tree_map(lambda a, b: a*(1-self.update_rate) + b*self.update_rate, target_model, self.model)
                elif total_step_counter % self.update_rate == 0:
                    target_model = tree_map(jnp.array, self.model)

                if terminated or truncated:
                    break


            for callback in self._callbacks:
                callback(self.model, epoch, loss_value, total_reward, total_step_counter)



def _log_batch_loss_and_eval(x, y, logits):
    logger.debug(f'x.shape: {x.shape} y.shape: {y.shape} logits.shape: {logits.shape}')

def _log_epoch(epoch, loss, scores):
    score_strings = [f'{name}: {value:.4f}' for name, value in scores.items()]
    logger.info(f'epoch {epoch}: loss: {loss:.4f} '+', '.join(score_strings))
        

