# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from libs.box_utils.rbbox_overlaps import rbbx_overlaps
from libs.configs import cfgs
from libs.box_utils.coordinate_convert import forward_convert
from libs.box_utils.cython_utils.cython_bbox import bbox_overlaps
from libs.box_utils.iou_cpu import get_iou_matrix


def get_horizen_minAreaRectangle(boxes):

    boxes = np.reshape(boxes, [-1, 8])
    x_list = boxes[:, 0::2]
    y_list = boxes[:, 1::2]

    y_max = np.max(y_list, axis=1)
    y_min = np.min(y_list, axis=1)
    x_max = np.max(x_list, axis=1)
    x_min = np.min(x_list, axis=1)

    return np.transpose(np.stack([x_min, y_min, x_max, y_max], axis=0))


def iou_rotate(boxes1, boxes2):
    boxes1_convert = forward_convert(boxes1, False)
    # boxes2_convert = forward_convert(boxes2, False)

    boxes1_h = get_horizen_minAreaRectangle(boxes1_convert)
    # boxes2_h = get_horizen_minAreaRectangle(boxes2_convert)

    iou_h = bbox_overlaps(np.ascontiguousarray(boxes1_h, dtype=np.float),
                          np.ascontiguousarray(boxes2, dtype=np.float))

    # argmax_overlaps_inds = np.argmax(iou_h, axis=1)
    # target_boxes = boxes2[argmax_overlaps_inds]
    #
    # delta_theta = np.abs(boxes1[:, -1] - target_boxes[:, -1])
    # iou_h[delta_theta > 10] = 0
    #
    # argmax_overlaps_inds = np.argmax(iou_h, axis=1)
    # max_overlaps = iou_h[np.arange(iou_h.shape[0]), argmax_overlaps_inds]
    # indices = max_overlaps < 0.7
    # iou_h[indices] = 0

    # boxes1 = boxes1[indices]
    #
    # overlaps = get_iou_matrix(np.ascontiguousarray(boxes1, dtype=np.float32),
    #                           np.ascontiguousarray(boxes2, dtype=np.float32))
    #
    # iou_r = np.zeros_like(iou_h)
    # iou_r[indices] = overlaps

    return iou_h

