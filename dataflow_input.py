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

class GoogleNetResize(imgaug.ImageAugmentor):
    """
    crop 8%~100% of the original image
    See `Going Deeper with Convolutions` by Google.
    """
    def __init__(self, crop_area_fraction=0.08,
                 aspect_ratio_low=0.75, aspect_ratio_high=1.333,
                 target_shape=224):
        self._init(locals())

    def _augment(self, img, _):
        h, w = img.shape[:2]
        area = h * w
        for _ in range(10):
            targetArea = self.rng.uniform(self.crop_area_fraction, 1.0) * area
            aspectR = self.rng.uniform(self.aspect_ratio_low, self.aspect_ratio_high)
            ww = int(np.sqrt(targetArea * aspectR) + 0.5)
            hh = int(np.sqrt(targetArea / aspectR) + 0.5)
            if self.rng.uniform() < 0.5:
                ww, hh = hh, ww
            if hh <= h and ww <= w:
                x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                out = img[y1:y1 + hh, x1:x1 + ww]
                out = cv2.resize(out, (self.target_shape, self.target_shape), interpolation=cv2.INTER_CUBIC)
                return out
        out = imgaug.ResizeShortestEdge(self.target_shape, interp=cv2.INTER_CUBIC).augment(img)
        out = imgaug.CenterCrop(self.target_shape).augment(out)
        return out

def fbresnet_augmentor(isTrain, target_shape=224):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    if isTrain:
        augmentors = [
            GoogleNetResize(crop_area_fraction=0.32, target_shape=target_shape),
            # GoogleNetResize(target_shape=target_shape),
            imgaug.RandomOrderAug(
                [# imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 # imgaug.Contrast((0.6, 1.4), clip=False),
                 # imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion for the constants copied from fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(int(256 / 224 * target_shape), cv2.INTER_CUBIC),
            imgaug.CenterCrop((target_shape, target_shape)),
        ]
    return augmentors

def data_augmentation(im, augmentors):
    """
    See explanations in the tutorial:
    http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html
    """
    assert isinstance(augmentors, list)
    aug = imgaug.AugmentorList(augmentors)
    im = aug.augment(im)
    return im

class MyDataFlow(DataFlow):
    def __init__(self, image_path, label_path, is_training=True, batch_size=64, img_size=224):
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
        self.augmentors = fbresnet_augmentor(isTrain=is_training, target_shape=img_size)

    def get_data(self):
        np.random.seed()
        img_batch = np.random.choice(self.img_name, self.batch_size)
        img_data = []
        img_label = []
        for item in img_batch:
            im = cv2.imread(os.path.join(self.image_path, item), cv2.IMREAD_COLOR)  
            im = data_augmentation(im, self.augmentors)
            img_data.append(im)
            img_label.append(self.data_dict[item])
        yield {'data': np.array(img_data), 'label': np.array(img_label)}


class MyDataFlowEval(DataFlow):
    def __init__(self, image_path, label_path, img_size=224):
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
        self.augmentors = fbresnet_augmentor(isTrain=False, target_shape=img_size)

    def get_data(self):
        for index, item in enumerate(self.img_name):
            im = cv2.imread(os.path.join(self.image_path, item), cv2.IMREAD_COLOR)  
            im = data_augmentation(im, self.augmentors)
            label = self.data_dict[item]
            yield {
                'name': item, 
                'data': np.expand_dims(np.array(im), axis=0),
                'label': np.array(label),
                'epoch': (index+1) == self.Length
            }