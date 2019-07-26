# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import cv2
import numpy as np
from libs.label_name_dict.label_dict import NAME_LABEL_MAP
from libs.configs import cfgs


def max_length_limitation(length, length_limitation):
    return tf.cond(tf.less(length, length_limitation),
                   true_fn=lambda: length,
                   false_fn=lambda: length_limitation)


def short_side_resize(img_tensor, gtboxes_and_label, target_shortside_len, length_limitation=1200):
    '''

    :param img_tensor:[h, w, c], gtboxes_and_label:[-1, 9].
    :param target_shortside_len:
    :param length_limitation: set max length to avoid OUT OF MEMORY
    :return:
    '''
    img_h, img_w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
    new_h, new_w = tf.cond(tf.less(img_h, img_w),
                           true_fn=lambda: (target_shortside_len,
                                            max_length_limitation(target_shortside_len * img_w // img_h, length_limitation)),
                           false_fn=lambda: (max_length_limitation(target_shortside_len * img_h // img_w, length_limitation),
                                             target_shortside_len))

    img_tensor = tf.expand_dims(img_tensor, axis=0)
    img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])

    x1, y1, x2, y2, x3, y3, x4, y4, label = tf.unstack(gtboxes_and_label, axis=1)

    x1, x2, x3, x4 = x1 * new_w // img_w, x2 * new_w // img_w, x3 * new_w // img_w, x4 * new_w // img_w
    y1, y2, y3, y4 = y1 * new_h // img_h, y2 * new_h // img_h, y3 * new_h // img_h, y4 * new_h // img_h

    img_tensor = tf.squeeze(img_tensor, axis=0)  # ensure image tensor rank is 3

    return img_tensor, tf.transpose(tf.stack([x1, y1, x2, y2, x3, y3, x4, y4, label], axis=0)), new_h, new_w


def short_side_resize_for_inference_data(img_tensor, target_shortside_len, length_limitation=1200, is_resize=True):
    if is_resize:
      img_h, img_w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

      new_h, new_w = tf.cond(tf.less(img_h, img_w),
                             true_fn=lambda: (target_shortside_len,
                                              max_length_limitation(target_shortside_len * img_w // img_h, length_limitation)),
                             false_fn=lambda: (max_length_limitation(target_shortside_len * img_h // img_w, length_limitation),
                                               target_shortside_len))

      img_tensor = tf.expand_dims(img_tensor, axis=0)
      img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])

      img_tensor = tf.squeeze(img_tensor, axis=0)  # ensure image tensor rank is 3
    return img_tensor


def flip_left_to_right(img_tensor, gtboxes_and_label):

    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

    img_tensor = tf.image.flip_left_right(img_tensor)

    x1, y1, x2, y2, x3, y3, x4, y4, label = tf.unstack(gtboxes_and_label, axis=1)
    new_x1 = w - x1
    new_x2 = w - x2
    new_x3 = w - x3
    new_x4 = w - x4

    return img_tensor, tf.transpose(tf.stack([new_x1, y1, new_x2, y2, new_x3, y3, new_x4, y4, label], axis=0))


def random_flip_left_right(img_tensor, gtboxes_and_label):
    img_tensor, gtboxes_and_label= tf.cond(tf.less(tf.random_uniform(shape=[], minval=0, maxval=1), 0.5),
                                           lambda: flip_left_to_right(img_tensor, gtboxes_and_label),
                                           lambda: (img_tensor, gtboxes_and_label))

    return img_tensor,  gtboxes_and_label
