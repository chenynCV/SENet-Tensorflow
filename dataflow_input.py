import numpy as np
import cv2
import json
import os
import random
from tensorpack import imgaug, dataset, ModelDesc, InputDesc
from tensorpack.dataflow import (
    AugmentImageComponent, PrefetchDataZMQ,
    BatchData, MultiThreadMapData, DataFlow)
from IPython import embed

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
            batch[i] = random.uniform(x, y) * batch[i]
            batch[i][batch[i] > 255] = 255
            batch[i] = batch[i].astype(np.uint8)
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


def data_augmentation(batch, img_size=(224, 224)):
    batch = _random_flip_leftright(batch)
    # batch = _random_multiply(batch, 0.8, 1.2)
    # batch = _random_crop(batch, img_size, padding=None)
    return np.array(batch)


class MyDataFlow(DataFlow):
    def __init__(self, image_path, label_path, is_training=True, batch_size=8, img_size=(224, 224)):
        # get all the image name and its label
        self.data_dict = {}
        with open(label_path, 'r') as f:
            label_list = json.load(f)
        for image in label_list:
            self.data_dict[image['image_id']] = int(image['label_id'])
        self.img_name = list(self.data_dict.keys())
        self.image_path = image_path
        self.is_training = is_training
        self.batch_size = batch_size
        self.img_size = img_size

    def get_data(self):
        np.random.seed()
        img_batch = np.random.choice(self.img_name, self.batch_size)
        img_data = []
        img_label = []
        for item in img_batch:
            img_data.append(cv2.resize(cv2.imread(os.path.join(self.image_path, item)), self.img_size))
            img_label.append(self.data_dict[item])
        if self.is_training:
            img_data = data_augmentation(np.array(img_data), img_size=self.img_size)
        yield {'data': np.array(img_data), 'label': np.array(img_label)}


class MyDataFlowEval(DataFlow):
    def __init__(self, image_path, label_path, img_size=(224, 224)):
        # get all the image name and its label
        self.data_dict = {}
        with open(label_path, 'r') as f:
            label_list = json.load(f)
        for image in label_list:
            self.data_dict[image['image_id']] = int(image['label_id'])
        self.img_name = list(self.data_dict.keys())
        self.image_path = image_path
        self.img_size = img_size
        self.Length = len(self.data_dict)

    def get_data(self):
        for index, item in enumerate(self.img_name):
            data = cv2.resize(cv2.imread(os.path.join(self.image_path, item)), self.img_size)
            label = self.data_dict[item]
            yield {
                'name': item, 
                'data': np.expand_dims(np.array(data), axis=0),
                'label': np.array(label),
                'epoch': (index+1) == self.Length
            }