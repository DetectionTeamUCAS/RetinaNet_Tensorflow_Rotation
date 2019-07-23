# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import tensorflow as tf
import cv2
import numpy as np
import math
from tqdm import tqdm
import argparse
from multiprocessing import Queue, Process
sys.path.append("../")

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.networks import build_whole_network
from help_utils import tools
from libs.label_name_dict.label_dict import *
from libs.box_utils import draw_box_in_img
from libs.box_utils.coordinate_convert import forward_convert, backward_convert


def worker(gpu_id, images, det_net, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not BGR
    img_batch = tf.cast(img_plac, tf.float32)

    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                     length_limitation=cfgs.IMG_MAX_LENGTH)
    if cfgs.NET_NAME in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:
        img_batch = (img_batch / 255 - tf.constant(cfgs.PIXEL_MEAN_)) / tf.constant(cfgs.PIXEL_STD)
    else:
        img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)

    img_batch = tf.expand_dims(img_batch, axis=0)

    detection_boxes, detection_scores, detection_category = det_net.build_whole_detection_network(
        input_img_batch=img_batch,
        gtboxes_batch_h=None,
        gtboxes_batch_r=None)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = det_net.get_restorer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model %d ...' % gpu_id)
        for a_img in images:
            raw_img = cv2.imread(a_img)
            raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]

            resized_img, detected_boxes, detected_scores, detected_categories = \
                sess.run(
                    [img_batch, detection_boxes, detection_scores, detection_category],
                    feed_dict={img_plac: raw_img[:, :, ::-1]}
                )
            detected_boxes = forward_convert(detected_boxes, False)

            detected_indices = detected_scores >= cfgs.VIS_SCORE
            detected_scores = detected_scores[detected_indices]
            detected_boxes = detected_boxes[detected_indices]
            detected_categories = detected_categories[detected_indices]

            resized_h, resized_w = resized_img.shape[1], resized_img.shape[2]
            scales = [raw_w / resized_w, raw_h / resized_h]
            result_dict = {'scales': scales, 'boxes': detected_boxes,
                           'scores': detected_scores, 'labels': detected_categories,
                           'image_id': a_img}
            result_queue.put_nowait(result_dict)


def test_icdar2015(det_net, real_test_img_list, gpu_ids, show_box):

    save_path = os.path.join('./test_icdar2015', cfgs.VERSION)
    tools.mkdir(save_path)

    nr_records = len(real_test_img_list)
    pbar = tqdm(total=nr_records)
    gpu_num = len(gpu_ids.strip().split(','))

    nr_image = math.ceil(nr_records / gpu_num)
    result_queue = Queue(500)
    procs = []

    for i in range(gpu_num):
        start = i * nr_image
        end = min(start + nr_image, nr_records)
        split_records = real_test_img_list[start:end]
        proc = Process(target=worker, args=(i, split_records, det_net, result_queue))
        print('process:%d, start:%d, end:%d' % (i, start, end))
        proc.start()
        procs.append(proc)

    for i in range(nr_records):
        res = result_queue.get()

        x1, y1, x2, y2, x3, y3, x4, y4 = res['boxes'][:, 0], res['boxes'][:, 1], res['boxes'][:, 2], res['boxes'][:, 3],\
                                         res['boxes'][:, 4], res['boxes'][:, 5], res['boxes'][:, 6], res['boxes'][:, 7]

        x1, y1 = x1 * res['scales'][0], y1 * res['scales'][1]
        x2, y2 = x2 * res['scales'][0], y2 * res['scales'][1]
        x3, y3 = x3 * res['scales'][0], y3 * res['scales'][1]
        x4, y4 = x4 * res['scales'][0], y4 * res['scales'][1]

        boxes = np.transpose(np.stack([x1, y1, x2, y2, x3, y3, x4, y4]))

        if show_box:
            boxes = backward_convert(boxes, False)
            nake_name = res['image_id'].split('/')[-1]
            draw_path = os.path.join(save_path, nake_name)
            draw_img = np.array(cv2.imread(res['image_id']), np.float32)

            final_detections = draw_box_in_img.draw_boxes_with_label_and_scores(draw_img,
                                                                                boxes=boxes,
                                                                                labels=res['labels'],
                                                                                scores=res['scores'],
                                                                                method=1,
                                                                                in_graph=False)
            cv2.imwrite(draw_path, final_detections)

        else:
            fw_txt_dt = open(os.path.join(save_path, 'res_{}.txt'.format(res['image_id'].split('/')[-1].split('.')[0])), 'w')

            for box in boxes:
                line = '%d,%d,%d,%d,%d,%d,%d,%d\n' % (box[0], box[1], box[2], box[3],
                                                      box[4], box[5], box[6], box[7])
                fw_txt_dt.write(line)
            fw_txt_dt.close()

        pbar.set_description("Test image %s" % res['image_id'].split('/')[-1])

        pbar.update(1)

    for p in procs:
        p.join()


def eval(num_imgs, test_dir, gpu_ids, show_box):

    test_imgname_list = [os.path.join(test_dir, img_name) for img_name in os.listdir(test_dir)
                         if img_name.endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))]
    assert len(test_imgname_list) != 0, 'test_dir has no imgs there.' \
                                        ' Note that, we only support img format of (.jpg, .png, and .tiff) '

    if num_imgs == np.inf:
        real_test_img_list = test_imgname_list
    else:
        real_test_img_list = test_imgname_list[: num_imgs]

    retinanet = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                     is_training=False)
    test_icdar2015(det_net=retinanet, real_test_img_list=real_test_img_list, gpu_ids=gpu_ids, show_box=show_box)


def parse_args():

    parser = argparse.ArgumentParser('evaluate the result with Pascal2007 stdand')

    parser.add_argument('--test_dir', dest='test_dir',
                        help='evaluate imgs dir ',
                        default='/data/ICDAR2015/test', type=str)
    parser.add_argument('--gpus', dest='gpus',
                        help='gpu id',
                        default='0,1,2,3,4,5,6,7', type=str)
    parser.add_argument('--eval_num', dest='eval_num',
                        help='the num of eval imgs',
                        default=np.inf, type=int)
    parser.add_argument('--show_box', '-s', default=False,
                        action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print(20*"--")
    print(args)
    print(20*"--")
    eval(args.eval_num,
         test_dir=args.test_dir,
         gpu_ids=args.gpus,
         show_box=args.show_box)



















