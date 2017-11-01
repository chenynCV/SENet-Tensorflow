#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright 2017 challenger.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import cv2
import logging
import PIL.Image as Image
import json
import os
import random

class scene_data_fn(object):

    def __init__(self, image_path, label_path):
        # get all the image name and its label
        self.data_dict = {}
        with open(label_path, 'r') as f:
            label_list = json.load(f)
        for image in label_list:
            self.data_dict[image['image_id']] = int(image['label_id'])
        self.start = 0
        self.end = 0
        self.Length = len(self.data_dict)
        self.img_name = list(self.data_dict.keys())
        self.image_path = image_path
        random.shuffle(self.img_name)

    def img_resize(self, imgpath, img_size):
        # resize the image to the specific size
        img = Image.open(imgpath)
        if (img.width > img.height):
            scale = float(img_size) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (
            int(img.width * scale + 1), img_size))).astype(np.float32)
        else:
            scale = float(img_size) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (
            img_size, int(img.height * scale + 1)))).astype(np.float32)
        # crop the proper size and scale to [-1, 1]
        img = (img[
                  (img.shape[0] - img_size) // 2:
                  (img.shape[0] - img_size) // 2 + img_size,
                  (img.shape[1] - img_size) // 2:
                  (img.shape[1] - img_size) // 2 + img_size,
                  :]-127)/255
        return img

    def next_batch(self, batch_size, img_size=32):
        # fetch a batch of images
        self.start = self.end
        if self.start >= self.Length:
            self.start = 0
            random.shuffle(self.img_name)
        img_data = []
        img_label = []
        index = self.start
        while len(img_data) < batch_size:
            if index >= self.Length:
                index = 0
            img_data.append(self.img_resize(os.path.join(self.image_path, self.img_name[index]), img_size))
            img_label.append(self.data_dict[self.img_name[index]])
            index += 1   
        self.end = index
        return np.array(img_data), np.array(img_label)


def img_resize(imgpath, img_size):
        img = Image.open(imgpath)
        if (img.width > img.height):
            scale = float(img_size) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (
            int(img.width * scale + 1), img_size))).astype(np.float32)
        else:
            scale = float(img_size) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (
            img_size, int(img.height * scale + 1)))).astype(np.float32)
        img = (img[
                  (img.shape[0] - img_size) // 2:
                  (img.shape[0] - img_size) // 2 + img_size,
                  (img.shape[1] - img_size) // 2:
                  (img.shape[1] - img_size) // 2 + img_size,
                  :]-127)/255
        return img


def train_log(filename='logfile'):
    # create logger
    logger_name = "filename"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # create file handler
    log_path = './' + filename + '.log'
    fh = logging.FileHandler(log_path)
    ch = logging.StreamHandler()

    # create formatter
    fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
    datefmt = "%a %d %b %Y %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)

    # add handler and formatter to logger
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ========================================================== #
# ├─ _random_crop()
# ├─ _random_flip_leftright()
# ├─ data_augmentation()
# └─ color_preprocessing()
# ========================================================== #

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def _random_multiply(batch, x, y):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = (random.uniform(x, y) * batch[i]).astype(np.uint8)
    return batch


def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])

    return x_train, x_test


def data_augmentation(batch, img_size=32):
    batch = _random_flip_leftright(batch)
    batch = _random_multiply(batch, 0.8, 1.2)
    batch = _random_crop(batch, [img_size, img_size], padding=4)
    return batch