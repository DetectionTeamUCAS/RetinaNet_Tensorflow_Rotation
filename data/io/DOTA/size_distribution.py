import os
import sys
import numpy as np
import cv2

sys.path.append('../../..')
from libs.box_utils.coordinate_convert import backward_convert


class_list = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
              'small-vehicle', 'large-vehicle', 'ship',
              'tennis-court', 'basketball-court',
              'storage-tank', 'soccer-ball-field',
              'roundabout', 'harbor',
              'swimming-pool', 'helicopter']

distribution = {}


def format_label(txt_list):
    format_data = []
    for i in txt_list[2:]:
        format_data.append(
            [int(xy) for xy in i.split(' ')[:8]] + [class_list.index(i.split(' ')[8])]
        )
        if i.split(' ')[8] not in class_list:
            print('warning found a new label :', i.split(' ')[8])
            exit()
    return np.array(format_data)


print('class_list', len(class_list))
raw_data = '/data/DOTA/train/'
raw_images_dir = os.path.join(raw_data, 'images', 'images')
raw_label_dir = os.path.join(raw_data, 'labelTxt')

save_dir = '/data/DOTA/DOTA_TOTAL/train800/'

images = [i for i in os.listdir(raw_images_dir) if 'png' in i]
labels = [i for i in os.listdir(raw_label_dir) if 'txt' in i]

print('find image', len(images))
print('find label', len(labels))

min_length = 1e10
max_length = 1

for idx, img in enumerate(images):
    img_data = cv2.imread(os.path.join(raw_images_dir, img))

    txt_data = open(os.path.join(raw_label_dir, img.replace('png', 'txt')), 'r').readlines()
    box = format_label(txt_data)
    box = backward_convert(box)
    for b in box:
        if class_list[int(b[-1])] not in distribution:
            distribution[class_list[int(b[-1])]] = {'s': 0, 'm': 0, 'l': 0}
        if np.sqrt(b[2] * b[3]) < 32:
            distribution[class_list[int(b[-1])]]['s'] += 1
        elif np.sqrt(b[2] * b[3]) < 96:
            distribution[class_list[int(b[-1])]]['m'] += 1
        else:
            distribution[class_list[int(b[-1])]]['l'] += 1
print(distribution)

