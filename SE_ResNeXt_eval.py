import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
import os
import json
from tensorpack import imgaug, dataset, ModelDesc, InputDesc
from tensorpack.dataflow import (
    AugmentImageComponent, PrefetchDataZMQ,
    BatchData, MultiThreadMapData, DataFlow)
from dataflow_input import MyDataFlowEval
from IPython import embed

os.environ['CUDA_VISIBLE_DEVICES']= '2'

weight_decay = 0.0005
momentum = 0.9

init_learning_rate = 0.1
cardinality = 2 # how many split ?
blocks = 3 # res_block ! (split + transition)
depth = 64 # out channel

"""
So, the total number of layers is (3*blokcs)*residual_layer_num + 2
because, blocks = split(conv 2) + transition(conv 1) = 3 layer
and, first conv layer 1, last dense layer 1
thus, total number of layers = (3*blocks)*residual_layer_num + 2
"""

reduction_ratio = 4

total_epochs = 100

batch_size = 64
image_size = 224
img_channels = 3
class_num = 80

iteration = 421
# 128 * 421 ~ 53,879

test_iteration = 10

def conv_layer(input, filter, kernel, stride, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        return network

def deconv_layer(input, filter, kernel, stride, padding='SAME', layer_name="deconv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d_transpose(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        return network

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Average_pooling(x, pool_size=[2,2], stride=2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Max_pooling(x, pool_size=[3,3], stride=2, padding='VALID') :
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x) :
    return tf.nn.sigmoid(x)

def tanh(x):
    return tf.tanh(x)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Fully_connected(x, units=class_num, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=False, units=units)

def Evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0

    for it in range(test_iteration):
        batch_data = next(scene_data_val)
        test_batch_x = batch_data['data']
        test_batch_y = batch_data['label']

        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_
        test_acc += acc_

    test_loss /= test_iteration # average loss
    test_acc /= test_iteration # average accuracy

    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary

class SE_ResNeXt():
    def __init__(self, x, training):
        self.training = training
        self.model = self.Build_SEnet(x)

    def first_layer(self, x, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=64, kernel=[7, 7], stride=2, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = Max_pooling(x)

            return x

    def transform_layer(self, x, stride, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=depth, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            x = conv_layer(x, filter=depth, kernel=[3,3], stride=stride, layer_name=scope+'_conv2')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            return x

    def transition_layer(self, x, out_dim, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=out_dim, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            # x = Relu(x)

            return x

    def split_layer(self, input_x, stride, layer_name):
        with tf.name_scope(layer_name) :
            layers_split = list()
            for i in range(cardinality) :
                splits = self.transform_layer(input_x, stride=stride, scope=layer_name + '_splitN_' + str(i))
                layers_split.append(splits)

            return Concatenation(layers_split)

    def squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name) :
            squeeze = Global_Average_Pooling(input_x)

            excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
            excitation = Relu(excitation)
            excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
            excitation = Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1,1,1,out_dim])
            scale = input_x * excitation

            return scale

    def residual_layer(self, input_x, out_dim, layer_num, res_block=blocks):
        # split + transform(bottleneck) + transition + merge
        # input_dim = input_x.get_shape().as_list()[-1]

        for i in range(res_block):
            input_dim = int(np.shape(input_x)[-1])

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1
            x = self.split_layer(input_x, stride=stride, layer_name='split_layer_'+layer_num+'_'+str(i))
            x = self.transition_layer(x, out_dim=out_dim, scope='trans_layer_'+layer_num+'_'+str(i))
            x = self.squeeze_excitation_layer(x, out_dim=out_dim, ratio=reduction_ratio, layer_name='squeeze_layer_'+layer_num+'_'+str(i))

            if flag is True :
                pad_input_x = Average_pooling(input_x)
                pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]]) # [?, height, width, channel]
            else :
                pad_input_x = input_x

            input_x = Relu(x + pad_input_x)

        return x

    def generator(self, x, scope="generator"):
        with tf.variable_scope(scope):
            n_downsampling = 5
            for i in range(n_downsampling):
                mult = pow(2, (n_downsampling - i))
                x = deconv_layer(x, filter=int((32 * mult) / 2), kernel=[3, 3], stride=2, layer_name='deconv' + str(i))
                x = Relu(x)

            x = conv_layer(x, filter=3, kernel=[7,7], stride=1, layer_name='conv1')
            x = 128 * Batch_Normalization(x, training=self.training, scope=scope+'_batch1') + 128

            return x

    def Build_SEnet(self, input_x):
        # only cifar10 architecture

        input_x = self.first_layer(input_x, scope='first_layer')

        x = self.residual_layer(input_x, out_dim=64, layer_num='1')
        x = self.residual_layer(x, out_dim=128, layer_num='2')
        x = self.residual_layer(x, out_dim=256, layer_num='3')
        x = self.residual_layer(x, out_dim=512, layer_num='4')

        recon_x = self.generator(x)
        # recon_x = tf.cast(recon_x, dtype=tf.uint8)

        x = Global_Average_Pooling(x)
        x = flatten(x)

        x = Fully_connected(x, layer_name='final_fully_connected')
        return x, recon_x

# image_size = 32, img_channels = 3, class_num = 10 in cifar10
x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels])
label = tf.placeholder(tf.float32, shape=[None,])
one_hot_labels = tf.one_hot(indices=tf.cast(label, tf.int32), depth=class_num)

training_flag = tf.placeholder(tf.bool)

learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits, recon_x = SE_ResNeXt(x, training=training_flag).model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=logits))

G_loss = 1e-2*tf.reduce_mean(tf.abs(x - recon_x))

l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
train = optimizer.minimize(cost + l2_loss * weight_decay + G_loss)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

values, indices = tf.nn.top_k(logits, 3)

val_dir = '/data0/AIChallenger/ai_challenger_scene_validation_20170908/scene_validation_images_20170908/'
annotations = '/data0/AIChallenger/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json'
# a DataFlow you implement to produce [tensor1, tensor2, ..] lists from whatever sources:
df = MyDataFlowEval(val_dir, annotations, img_size=(image_size, image_size))
# start 3 processes to run the dataflow in parallel
df = PrefetchDataZMQ(df, nr_proc=1)
df.reset_state()
scene_data_val = df.get_data()

saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model_release')
    print("loading checkpoint...")
    saver.restore(sess, ckpt.model_checkpoint_path)

    result = []
    for it in scene_data_val:
        temp_dict = {}
        feed_dict = {x: it['data'], training_flag: False}
        predictions = np.squeeze(sess.run(indices, feed_dict=feed_dict), axis=0)
        temp_dict['image_id'] = it['name']
        temp_dict['label_id'] = predictions.tolist()
        result.append(temp_dict)
        print('image %s is %d,%d,%d, label: %d' % (it['name'], predictions[0], predictions[1], predictions[2], it['label']))
        if it['epoch']:
            break

    with open('submit.json', 'w') as f:
        json.dump(result, f)
        print('write result json, num is %d' % len(result))

