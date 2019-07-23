# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys
sys.path.append('../../')
import xml.etree.cElementTree as ET
import numpy as np
import tensorflow as tf
import glob
import cv2
import json
from libs.label_name_dict.label_dict import *
from help_utils.tools import *

tf.app.flags.DEFINE_string('coco_dir', '/data/COCO/coco_trainvalmini.odgt', 'coco dir')
tf.app.flags.DEFINE_string('save_name', 'train', 'save name')
tf.app.flags.DEFINE_string('save_dir', '../tfrecord/', 'save name')
tf.app.flags.DEFINE_string('dataset', 'coco', 'dataset')
FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_pascal_to_tfrecord(coco_trainvalmini):
    save_path = FLAGS.save_dir + FLAGS.dataset + '_' + FLAGS.save_name + '.tfrecord'
    mkdir(FLAGS.save_dir)

    # writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    # writer = tf.python_io.TFRecordWriter(path=save_path, options=writer_options)
    writer = tf.python_io.TFRecordWriter(path=save_path)

    with open(coco_trainvalmini) as f:
        files = f.readlines()

    img_count = 0
    gt_count = 0

    for count, raw_line in enumerate(files):
        file = json.loads(raw_line)
        img_path = os.path.join('/data/COCO/train2017', file['fpath'].split('_')[-1])
        img_name = file['ID']

        if not os.path.exists(img_path):
            # print('{} is not exist!'.format(img_path))
            img_count += 1
            continue
        # img = np.array(Image.open(img_path))
        img = cv2.imread(img_path)[:, :, ::-1]

        if img is None:
            continue

        gtboxes = file['gtboxes']
        img_height = file['height']
        img_width = file['width']

        if len(gtboxes) == 0:
            # print('{}: gt is not exist!'.format(img_path))
            gt_count += 1
            continue

        gtbox_label = []
        for gt in gtboxes:
            box = gt['box']
            label = gt['tag']
            gtbox_label.append([box[0], box[1], box[0]+box[2], box[1]+box[3], NAME_LABEL_MAP[label]])

        gtbox_label = np.array(gtbox_label, np.int32)

        feature = tf.train.Features(feature={
            # do not need encode() in linux
            'img_name': _bytes_feature(img_name.encode()),
            # 'img_name': _bytes_feature(img_name),
            'img_height': _int64_feature(img_height),
            'img_width': _int64_feature(img_width),
            'img': _bytes_feature(img.tostring()),
            'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),
            'num_objects': _int64_feature(gtbox_label.shape[0])
        })

        example = tf.train.Example(features=feature)

        writer.write(example.SerializeToString())

        view_bar('Conversion progress', count + 1, len(files))
    writer.close()
    print('{} images not exist!'.format(img_count))
    print('{} gts not exist!'.format(gt_count))
    print('\nConversion is complete!')


if __name__ == '__main__':
    # xml_path = '../data/dataset/VOCdevkit/VOC2007/Annotations/000005.xml'
    # read_xml_gtbox_and_label(xml_path)

    # coco_path = '/unsullied/sharefs/_research_detection/GeneralDetection/COCO/data/MSCOCO/odformat/coco_trainvalmini.odgt'
    # convert_pascal_to_tfrecord(coco_path)
    convert_pascal_to_tfrecord(FLAGS.coco_dir)
