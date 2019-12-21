# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys
sys.path.append('../../')
import xml.etree.cElementTree as ET
import numpy as np
import tensorflow as tf
import glob
import cv2
from tqdm import tqdm
import os

from help_utils.tools import mkdir
from help_utils.tools import *


tf.app.flags.DEFINE_string('VOC_dir', '/data/dataset/HRSC2016/HRSC2016/FullDataSet/', 'Voc dir')
tf.app.flags.DEFINE_string('xml_dir', 'Annotations', 'xml dir')
tf.app.flags.DEFINE_string('image_dir', 'AllImages', 'image dir')
tf.app.flags.DEFINE_string('label_txt', '/data/dataset/HRSC2016/HRSC2016/ImageSets/trainval.txt', 'label txt')
tf.app.flags.DEFINE_string('save_name', 'train', 'save name')
tf.app.flags.DEFINE_string('save_dir', '../tfrecord/', 'save name')
tf.app.flags.DEFINE_string('img_format', '.bmp', 'format of image')
tf.app.flags.DEFINE_string('dataset', 'HRSC2016', 'dataset')
FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def coordinate_convert_r(box):
    w, h = box[2:-1]
    theta = -box[-1]
    x_lu, y_lu = -w/2, h/2
    x_ru, y_ru = w/2, h/2
    x_ld, y_ld = -w/2, -h/2
    x_rd, y_rd = w/2, -h/2

    x_lu_ = math.cos(theta)*x_lu + math.sin(theta)*y_lu + box[0]
    y_lu_ = -math.sin(theta)*x_lu + math.cos(theta)*y_lu + box[1]

    x_ru_ = math.cos(theta) * x_ru + math.sin(theta) * y_ru + box[0]
    y_ru_ = -math.sin(theta) * x_ru + math.cos(theta) * y_ru + box[1]

    x_ld_ = math.cos(theta) * x_ld + math.sin(theta) * y_ld + box[0]
    y_ld_ = -math.sin(theta) * x_ld + math.cos(theta) * y_ld + box[1]

    x_rd_ = math.cos(theta) * x_rd + math.sin(theta) * y_rd + box[0]
    y_rd_ = -math.sin(theta) * x_rd + math.cos(theta) * y_rd + box[1]

    convert_box = [x_lu_, y_lu_, x_ru_, y_ru_, x_rd_, y_rd_, x_ld_, y_ld_]

    return convert_box

def read_xml_gtbox_and_label(xml_path):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 9],
           and has [x1, y1, x2, y2, x3, y3, x4, y4, label] in a per row
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    for child_of_root in root:
        # if child_of_root.tag == 'filename':
        #     assert child_of_root.text == xml_path.split('/')[-1].split('.')[0] \
        #                                  + FLAGS.img_format, 'xml_name and img_name cannot match'

        # if child_of_root.tag == 'size':
        #     for child_item in child_of_root:
        #         if child_item.tag == 'width':
        #             img_width = int(child_item.text)
        #         if child_item.tag == 'height':
        #             img_height = int(child_item.text)
        #
        # if child_of_root.tag == 'object':
        #     label = None
        #     for child_item in child_of_root:
        #         if child_item.tag == 'name':
        #             label = NAME_LABEL_MAP[child_item.text]
        #         if child_item.tag == 'bndbox':
        #             tmp_box = []
        #             for node in child_item:
        #                 tmp_box.append(float(node.text))
        #             assert label is not None, 'label is none, error'
        #             tmp_box.append(label)
        #             box_list.append(tmp_box)

        # ship
        if child_of_root.tag == 'Img_SizeWidth':
            img_width = int(child_of_root.text)
        if child_of_root.tag == 'Img_SizeHeight':
            img_height = int(child_of_root.text)
        if child_of_root.tag == 'HRSC_Objects':
            box_list = []
            for child_item in child_of_root:
                if child_item.tag == 'HRSC_Object':
                    label = 1
                    # for child_object in child_item:
                    #     if child_object.tag == 'Class_ID':
                    #         label = NAME_LABEL_MAP[child_object.text]
                    tmp_box = [0., 0., 0., 0., 0.]
                    for node in child_item:
                        if node.tag == 'mbox_cx':
                            tmp_box[0] = float(node.text)
                        if node.tag == 'mbox_cy':
                            tmp_box[1] = float(node.text)
                        if node.tag == 'mbox_w':
                            tmp_box[2] = float(node.text)
                        if node.tag == 'mbox_h':
                            tmp_box[3] = float(node.text)
                        if node.tag == 'mbox_ang':
                            tmp_box[4] = float(node.text)

                    tmp_box = coordinate_convert_r(tmp_box)
                        # assert label is not None, 'label is none, error'
                    tmp_box.append(label)
                    # if len(tmp_box) != 0:
                    box_list.append(tmp_box)
            # box_list = coordinate_convert(box_list)
            # print(box_list)
    gtbox_label = np.array(box_list, dtype=np.int32)

    return img_height, img_width, gtbox_label


def convert_pascal_to_tfrecord():
    xml_path = os.path.join(FLAGS.VOC_dir, FLAGS.xml_dir)
    image_path = os.path.join(FLAGS.VOC_dir, FLAGS.image_dir)
    save_path = os.path.join(FLAGS.save_dir, FLAGS.dataset + '_' + FLAGS.save_name + '.tfrecord')
    mkdir(FLAGS.save_dir)

    # writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    # writer = tf.python_io.TFRecordWriter(path=save_path, options=writer_options)
    writer = tf.python_io.TFRecordWriter(path=save_path)

    fr = open(FLAGS.label_txt, 'r')
    lines = fr.readlines()

    real_cnt = 0
    pbar = tqdm(glob.glob(xml_path + '/*.xml'))

    for xml in pbar:
        # to avoid path error in different development platform
        xml = xml.replace('\\', '/')
        tmp = xml.split('/')[-1].split('.')[0] + "\n"
        if tmp not in lines:
            continue

        img_name = xml.split('/')[-1].split('.')[0] + FLAGS.img_format
        img_path = image_path + '/' + img_name

        if not os.path.exists(img_path):
            print('{} is not exist!'.format(img_path))
            continue

        img_height, img_width, gtbox_label = read_xml_gtbox_and_label(xml)
        # assert gtbox_label.shape[0] > 0, 'no box'
        if gtbox_label.shape[0] == 0:
            continue

        # img = np.array(Image.open(img_path))
        img = cv2.imread(img_path)[:, :, ::-1]

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
        real_cnt += 1
        pbar.set_description("Conversion progress")

    print('\nConversion is complete! {} images.'.format(real_cnt))
    writer.close()


if __name__ == '__main__':
    # xml_path = '../data/dataset/VOCdevkit/VOC2007/Annotations/000005.xml'
    # read_xml_gtbox_and_label(xml_path)

    convert_pascal_to_tfrecord()
