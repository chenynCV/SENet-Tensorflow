import tensorflow as tf
import numpy as np
import os
import json
from tensorpack import imgaug, dataset, ModelDesc, InputDesc
from tensorpack.dataflow import (PrefetchDataZMQ, BatchData)
from dataflow_input import MyDataFlowEval
import resnet_model
from IPython import embed

os.environ['CUDA_VISIBLE_DEVICES']= '2'

init_learning_rate = 0.1
batch_size = 64
image_size = 224
img_channels = 3
class_num = 80

weight_decay = 1e-4
momentum = 0.9

total_epochs = 100
iteration = 421
# 128 * 421 ~ 53,879
test_iteration = 10

def dist_top_k(feat, centers):
    feat = feat[0, ]
    diff = centers_class - feat
    diff = - tf.reduce_sum(diff*diff, axis=1)
    _, predictions = tf.nn.top_k(diff, 3)
    return predictions

def get_tensor_by_name(save_file, var_name):
    reader = tf.train.NewCheckpointReader(save_file)
    return reader.get_tensor(var_name)

def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    # centers = tf.nn.l2_normalize(centers, 1, 1e-10, name='centers_norm')
    loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers

def focal_loss(onehot_labels, cls_preds,
                alpha=0.25, gamma=2.0, name=None, scope=None):
    """Compute softmax focal loss between logits and onehot labels
    logits and onehot_labels must have same shape [batchsize, num_classes] and
    the same data type (float16, 32, 64)
    Args:
      onehot_labels: Each row labels[i] must be a valid probability distribution
      cls_preds: Unscaled log probabilities
      alpha: The hyperparameter for adjusting biased samples, default is 0.25
      gamma: The hyperparameter for penalizing the easy labeled samples
      name: A name for the operation (optional)
    Returns:
      A 1-D tensor of length batch_size of same type as logits with softmax focal loss
    """
    with tf.name_scope(scope, 'focal_loss', [cls_preds, onehot_labels]) as sc:
        logits = tf.convert_to_tensor(cls_preds)
        onehot_labels = tf.convert_to_tensor(onehot_labels)

        precise_logits = tf.cast(logits, tf.float32) if (
                        logits.dtype == tf.float16) else logits
        onehot_labels = tf.cast(onehot_labels, precise_logits.dtype)
        predictions = tf.nn.sigmoid(logits)
        predictions_pt = tf.where(tf.equal(onehot_labels, 1), predictions, 1.-predictions)
        # add small value to avoid 0
        epsilon = 1e-8
        alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
        alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1-alpha_t)
        losses = tf.reduce_sum(-alpha_t * tf.pow(1. - predictions_pt, gamma) * tf.log(predictions_pt+epsilon),
                                     name=name, axis=1)
        return losses

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

        loss_, acc_ = sess.run([Total_loss, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_
        test_acc += acc_

    test_loss /= test_iteration # average loss
    test_acc /= test_iteration # average accuracy

    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary

def resnet_model_fn(inputs, training):
    """Our model_fn for ResNet to be used with our Estimator."""

    network = resnet_model.imagenet_resnet_v2(
        resnet_size=18, num_classes=class_num, mode='se', data_format=None)
    inputs= network(inputs=inputs, is_training=training)
    feat = tf.nn.l2_normalize(inputs, 1, 1e-10, name='feat')
    inputs = tf.layers.dense(inputs=inputs, units=class_num)
    # inputs = tf.layers.dense(inputs=feat, units=class_num)
    inputs = tf.identity(inputs, 'final_dense')

    return inputs, feat

# image_size = 32, img_channels = 3, class_num = 10 in cifar10
x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels])
label = tf.placeholder(tf.float32, shape=[None,])
one_hot_labels = tf.one_hot(indices=tf.cast(label, tf.int32), depth=class_num)

training_flag = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits, feat = resnet_model_fn(x, training=training_flag)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=logits))
Focal_loss = tf.reduce_mean(focal_loss(one_hot_labels, logits, alpha=0.5))
l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
Center_loss, Centers = center_loss(feat, tf.cast(label, dtype=tf.int32), 0.95, class_num)
Total_loss = Focal_loss + l2_loss + Center_loss

optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
# Batch norm requires update_ops to be added as a train_op dependency.
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(Total_loss)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

values, indices = tf.nn.top_k(logits, 3)

val_dir = '/data0/AIChallenger/ai_challenger_scene_validation_20170908/scene_validation_images_20170908/'
annotations = '/data0/AIChallenger/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json'
# a DataFlow you implement to produce [tensor1, tensor2, ..] lists from whatever sources:
df = MyDataFlowEval(val_dir, annotations, img_size=image_size)
# start 3 processes to run the dataflow in parallel
df = PrefetchDataZMQ(df, nr_proc=1)
df.reset_state()
scene_data_val = df.get_data()

centers_class = np.load("centers.npy")
centers_class = tf.convert_to_tensor(centers_class)
indices_Center = dist_top_k(feat, centers_class)

saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model_release')
    print("loading checkpoint...")
    saver.restore(sess, ckpt.model_checkpoint_path)

    result = []
    for it in scene_data_val:
        temp_dict = {}
        feed_dict = {x: it['data'], training_flag: False}
        predictions, predictions_Center = sess.run([indices, indices_Center], feed_dict=feed_dict)
        predictions = np.squeeze(predictions, axis=0)

        predictions = predictions_Center

        temp_dict['image_id'] = it['name']
        temp_dict['label_id'] = predictions.tolist()
        result.append(temp_dict)
        print('image %s is %d,%d,%d, label: %d' % (it['name'], predictions[0], predictions[1], predictions[2], it['label']))
        if it['epoch']:
            break

    with open('submit.json', 'w') as f:
        json.dump(result, f)
        print('write result json, num is %d' % len(result))

