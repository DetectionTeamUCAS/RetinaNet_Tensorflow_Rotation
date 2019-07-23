# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs.configs import cfgs
from libs.networks.resnet import fusion_two_layer


def fpn(feature_dict, scope):
    pyramid_dict = {}
    with tf.variable_scope('build_pyramid_{}'.format(scope)):
        with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY),
                            activation_fn=None, normalizer_fn=None):

            P5 = slim.conv2d(feature_dict['C5'],
                             num_outputs=cfgs.FPN_CHANNEL,
                             kernel_size=[1, 1],
                             stride=1, scope='build_P5')

            pyramid_dict['P5'] = P5

            for level in range(4, 2, -1):  # build [P4, P3]

                pyramid_dict['P%d' % level] = fusion_two_layer(C_i=feature_dict["C%d" % level],
                                                               P_j=pyramid_dict["P%d" % (level + 1)],
                                                               scope='build_P%d' % level)
            for level in range(5, 2, -1):
                pyramid_dict['P%d' % level] = slim.conv2d(pyramid_dict['P%d' % level],
                                                          num_outputs=cfgs.FPN_CHANNEL,
                                                          kernel_size=[3, 3], padding="SAME",
                                                          stride=1, scope="fuse_P%d" % level)

            p6 = slim.conv2d(pyramid_dict['P5'] if cfgs.USE_P5 else feature_dict['C5'],
                             num_outputs=cfgs.FPN_CHANNEL, kernel_size=[3, 3], padding="SAME",
                             stride=2, scope='p6_conv')
            pyramid_dict['P6'] = p6

            p7 = tf.nn.relu(p6, name='p6_relu')

            p7 = slim.conv2d(p7,
                             num_outputs=cfgs.FPN_CHANNEL, kernel_size=[3, 3], padding="SAME",
                             stride=2, scope='p7_conv')

            pyramid_dict['P7'] = p7
            return pyramid_dict


def gp(fm1, fm2, scope):
    h, w = tf.shape(fm2)[1], tf.shape(fm2)[2]
    global_ctx = tf.reduce_mean(fm1, axis=[1, 2], keep_dims=True)
    global_ctx = tf.sigmoid(global_ctx)
    output = (global_ctx * fm2) + tf.image.resize_bilinear(fm1, size=[h, w], name='resize_' + scope)
    return output


def rcb(fm, scope):
    fm = tf.nn.relu(fm)
    fm = slim.conv2d(fm, num_outputs=cfgs.FPN_CHANNEL, kernel_size=[3, 3],
                     padding="SAME", stride=1, scope='RCB_%s' % scope,
                     activation_fn=None)
    fm = slim.batch_norm(fm, scope='BN_%s' % scope)
    return fm


def sum_fm(fm1, fm2, scope):
    h, w = tf.shape(fm2)[1], tf.shape(fm2)[2]
    output = fm2 + tf.image.resize_bilinear(fm1, size=[h, w], name='resize_' + scope)
    return output


def nas_fpn(feature_dict, scope):
    GP_P6_P4 = gp(feature_dict['P6'], feature_dict['P6'], 'GP_P6_P4_{}'.format(scope))
    GP_P6_P4_RCB = rcb(GP_P6_P4, 'GP_P6_P4_RCB_{}'.format(scope))
    SUM1 = sum_fm(GP_P6_P4_RCB, feature_dict['P4'], 'SUM1_{}'.format(scope))
    SUM1_RCB = rcb(SUM1, 'SUM1_RCB_{}'.format(scope))
    SUM2 = sum_fm(SUM1_RCB, feature_dict['P3'], 'SUM2_{}'.format(scope))
    SUM2_RCB = rcb(SUM2, 'SUM2_RCB_{}'.format(scope))  # P3
    SUM3 = sum_fm(SUM2_RCB, SUM1_RCB, 'SUM3_{}'.format(scope))
    SUM3_RCB = rcb(SUM3, 'SUM3_RCB_{}'.format(scope))  # P4
    SUM3_RCB_GP = gp(SUM2_RCB, SUM3_RCB, 'SUM3_RCB_GP_{}'.format(scope))
    SUM4 = sum_fm(SUM3_RCB_GP, feature_dict['P5'], 'SUM4_{}'.format(scope))
    SUM4_RCB = rcb(SUM4, 'SUM4_RCB_{}'.format(scope))  # P5
    SUM4_RCB_GP = gp(SUM1_RCB, SUM4_RCB, 'SUM4_RCB_GP_{}'.format(scope))
    SUM5 = sum_fm(SUM4_RCB_GP, feature_dict['P7'], 'SUM5_{}'.format(scope))
    SUM5_RCB = rcb(SUM5, 'SUM5_RCB_{}'.format(scope))  # P7
    h, w = tf.shape(feature_dict['P6'])[1], tf.shape(feature_dict['P6'])[2]
    SUM5_RCB_resize = tf.image.resize_bilinear(SUM5_RCB, size=[h, w], name='resize_SUM5_RCB_resize_'.format(scope))
    SUM4_RCB_GP1 = gp(SUM4_RCB, SUM5_RCB_resize, 'SUM4_RCB_GP1_{}'.format(scope))
    SUM4_RCB_GP1_RCB = rcb(SUM4_RCB_GP1, 'SUM4_RCB_GP1_RCB_{}'.format(scope))  # P6
    pyramid_dict = {'P3': SUM2_RCB, 'P4': SUM3_RCB, 'P5': SUM4_RCB,
                    'P6': SUM4_RCB_GP1_RCB, 'P7': SUM5_RCB}
    return pyramid_dict

