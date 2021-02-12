#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: harry
# @Date  : 2/4/21 6:43 PM
# @Desc  : utils

import cv2
import numpy as np
import tensorflow as tf


def process_frame(frame, shape=(120, 120), normalize=True):
    """Preprocesses a frame to shape[0] x shape[1] x 1 grayscale
    :param frame: The frame to process.  Must have values ranging from 0-255.
    :param shape: Desired shape to return.
    :param normalize: Whether to normalize the frame by dividing 255.
    """
    frame = frame.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work

    if len(frame.shape) < 3:
        frame = np.expand_dims(frame, axis=-1)

    if frame.shape[-1] > 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    frame = frame.reshape((*shape, 1))

    if normalize:
        frame = frame.astype('float32') / 255.0

    return frame.astype('float32')


def get_expected_return(
        rewards: tf.Tensor,
        dones: tf.Tensor,
        next_value: tf.Tensor,
        gamma: float,
) -> tf.Tensor:
    """
    Compute expected returns per timestep.
    :param rewards:
    :param dones:
    :param next_value: the value of the next state given by the critic network.
    :param gamma
    :return:
    """
    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    dones = tf.cast(dones[::-1], dtype=tf.float32)

    prev_returns_i = next_value
    prev_returns_i_shape = prev_returns_i.shape
    for i in tf.range(n):
        returns_i = rewards[i] + gamma * prev_returns_i * (1.0 - dones[i])
        returns = returns.write(i, returns_i)
        prev_returns_i = returns_i
        prev_returns_i.set_shape(prev_returns_i_shape)

    returns = returns.stack()[::-1]

    return returns