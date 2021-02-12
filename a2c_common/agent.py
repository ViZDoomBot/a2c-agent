#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : agent.py.py
# @Author: harry
# @Date  : 2/12/21 3:27 AM
# @Desc  : A2C Agent

import os
import numpy as np
import tensorflow as tf

from a2c_common.model import ActorCritic
from a2c_common.game_wrapper import GameWrapper
from a2c_common.utils import get_expected_return
from a2c_common.loss import compute_loss
from typing import Tuple, List


class A2CAgent(object):
    def __init__(
            self,
            model: ActorCritic,
            game: GameWrapper,
            # num_actions: int,
            # input_shape: Tuple[int, int] = (120, 120),
    ):
        self.model = model
        self.game = game
        # self.num_actions = num_actions
        # self.input_shape = input_shape

    def set_game_wrapper(self, game: GameWrapper):
        self.game = game

    def get_action(self, state: np.ndarray, stochastic: bool = True) -> int:
        """
        Get action to take given current state
        :param state: game state of shape (height, width, num_channels).
        :param stochastic: randomly sample an action from policy probs if True, choose argmax action o.w.
        :return: chosen action id.
        """
        state = tf.expand_dims(state, 0)
        action_probs, _ = self.model(state)

        if stochastic:
            action = tf.random.categorical(action_probs, 1)[0, 0]
        else:
            action = tf.math.argmax(action_probs, -1)[0]

        return int(action)

    def run_batch(self, batch_size: int, reward_shaping: bool = False) -> List[tf.Tensor]:
        """
        Runs a single batch to collect training data of length batch_size.
        :param batch_size
        :param reward_shaping
        :return:
        """
        action_probs_raw = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        dones = tf.TensorArray(dtype=tf.bool, size=0, dynamic_size=True)

        state = self.game.get_state()
        state_shape = state.shape
        for t in tf.range(batch_size):
            # run the model and to get action probabilities and critic value
            state = tf.expand_dims(state, axis=0)
            action_probs_raw_t, value = self.model(state)

            # sample next action from the action probability distribution
            action = tf.random.categorical(action_probs_raw_t, 1)[0, 0]
            # print(action_probs_raw_t)
            action_probs_t = action_probs_raw_t[0, action]

            action_probs_raw = action_probs_raw.write(t, action_probs_raw_t)
            action_probs = action_probs.write(t, action_probs_t)
            values = values.write(t, tf.squeeze(value))

            # perform action in game env
            state, reward, done, shaping_reward = self.game.tf_step(action)
            state.set_shape(state_shape)
            if reward_shaping:
                reward += shaping_reward

            rewards = rewards.write(t, reward)
            dones = dones.write(t, done)

            # reset game env if necessary
            if done:
                state = self.game.reset()

        action_probs_raw = action_probs_raw.stack()
        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()
        dones = dones.stack()

        # if the episode didn't end on the last step we need to compute the value for the last state
        if dones[-1]:
            next_value = tf.constant(0.0, dtype=tf.float32)
        else:
            state = tf.expand_dims(state, axis=0)
            _, value = self.model(state)
            next_value = tf.stop_gradient(value[0, 0])

        return action_probs_raw, action_probs, values, rewards, dones, next_value

    # XXX: currently using tf.function will cause ViZDoom running abnormally...
    # @tf.function
    def train_step(
            self,
            max_steps_per_episode: int,
            batch_size: int,
            optimizer: tf.keras.optimizers.Optimizer,
            gamma: float = 0.99,
            entropy_coff: float = 0.0001,
            reward_shaping: bool = False
    ) -> tf.Tensor:
        """
        Run a model training episode for max_steps_per_episode steps.
        :param max_steps_per_episode
        :param batch_size
        :param optimizer
        :param gamma
        :param entropy_coff
        :param reward_shaping
        :return:
        """
        # divide episode steps into batches (ignoring remainder)
        batch_n = max_steps_per_episode // batch_size
        episode_reward = tf.constant(0.0, dtype=tf.float32)

        for _ in tf.range(batch_n):
            with tf.GradientTape() as tape:
                # run the model for one batch to collect training data
                action_probs_raw, action_probs, values, rewards, dones, next_value = self.run_batch(
                    batch_size, reward_shaping
                )
                # calculate expected returns
                returns = get_expected_return(rewards, dones, next_value, gamma)
                # convert training data to appropriate TF tensor shapes
                action_probs, values, returns = [
                    tf.expand_dims(x, -1) for x in [action_probs, values, returns]]
                # calculate loss
                loss = compute_loss(action_probs_raw, action_probs, values, returns, entropy_coff)
                # tf.print("rewards:", rewards)
                # print("dones:", dones)
                # print("next_value:", next_value)
                # print("returns:", returns)
                # tf.print("loss:", loss)

            grads = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            batch_reward = tf.math.reduce_sum(rewards)
            episode_reward += batch_reward
            # tf.print("batch_reward:", batch_reward)

        return episode_reward

    def save(self, folder_name):
        """
        Save model.
        """
        if folder_name is None:
            print('WARNING: folder_name not given, skipping save')
            return

        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        # self.model.save(os.path.join(folder_name, 'a2c'), save_format='tf')
        self.model.save_weights(os.path.join(folder_name, 'a2c'), save_format='tf')

    def load(self, folder_name):
        """
        Load model.
        """
        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')

        # self.model = tf.keras.models.load_model(os.path.join(folder_name, 'a2c'))
        self.model.load_weights(os.path.join(folder_name, 'a2c'))
