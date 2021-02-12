#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : loss.py.py
# @Author: harry
# @Date  : 2/12/21 2:13 AM
# @Desc  : Compute actor-critic loss with entropy

import tensorflow as tf

from tensorflow.keras.losses import Huber

_huber_loss = Huber()


def compute_loss(
        action_probs_raw: tf.Tensor,
        action_probs: tf.Tensor,
        values: tf.Tensor,
        returns: tf.Tensor,
        entropy_coff: float = 0.0001
):
    """
    Compute actor-critic loss with entropy.
    :param action_probs_raw: shape (batch_size, num_actions), raw outputs of policy network.
    :param action_probs: shape (batch_size,), selected action probs of true actions taken by agent for the episode.
    :param values: shape (batch_size,), outputs of value network.
    :param returns: shape (batch_size,), discounted expected rewards for each time step of the episode.
    :param entropy_coff: coefficient for entropy loss.
    :return: the computed total loss.
    """
    # compute actor loss (policy loss + entropy loss)
    advantages = returns - values
    action_log_probs = tf.math.log(action_probs)
    policy_loss = tf.math.reduce_mean(action_log_probs * advantages)
    action_log_probs_raw = tf.math.log(action_probs_raw)
    entropy_loss = -tf.reduce_mean(tf.reduce_sum(action_probs_raw * action_log_probs_raw, axis=-1))
    # minus sign comes from the fact that we want to MAXIMIZE the expected discounted rewards
    actor_loss = -(policy_loss + entropy_coff * entropy_loss)
    actor_loss = tf.cast(actor_loss, dtype='float32')
    # print('policy_loss:', policy_loss)
    # print('entropy_loss:', entropy_loss)
    # print('actor_loss:', actor_loss)

    # compute critic loss
    critic_loss = tf.cast(_huber_loss(returns, values), dtype='float32')
    # print('critic_loss:', critic_loss)

    return actor_loss + critic_loss
