import tensorflow as tf
import numpy as np
import os
from tensorpack import imgaug, dataset, ModelDesc, InputDesc
from tensorpack.dataflow import (PrefetchDataZMQ, BatchData)
from dataflow_input import MyDataFlow
import resnet_model
from IPython import embed

os.environ['CUDA_VISIBLE_DEVICES']= '1'

init_learning_rate = 0.1
batch_size = 64
image_size = 224
img_channels = 3
class_num = 80

weight_decay = 1e-4
momentum = 0.9

total_epochs = 100
iteration = 2*421
# 128 * 421 ~ 53,879
test_iteration = 10

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
        resnet_size=50, num_classes=class_num, mode='se', data_format=None)
    inputs= network(inputs=inputs, is_training=training)
    feat = tf.nn.l2_normalize(inputs, 1, 1e-10, name='feat')
    inputs = tf.layers.dense(inputs=inputs, units=class_num)
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
Focal_loss = tf.reduce_mean(focal_loss(one_hot_labels, logits))
l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
Center_loss, _ = center_loss(logits, tf.cast(label, dtype=tf.int32), 0.95, class_num)
Total_loss = Focal_loss + l2_loss

optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
# Batch norm requires update_ops to be added as a train_op dependency.
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(Total_loss)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

val_dir = '/data0/AIChallenger/ai_challenger_scene_validation_20170908/scene_validation_images_20170908/'
annotations = '/data0/AIChallenger/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json'
# a DataFlow you implement to produce [tensor1, tensor2, ..] lists from whatever sources:
df = MyDataFlow(val_dir, annotations, is_training=False, batch_size=batch_size, img_size=image_size)
# start 3 processes to run the dataflow in parallel
df = PrefetchDataZMQ(df, nr_proc=10)
df.reset_state()
scene_data_val = df.get_data()

train_dir = '/data0/AIChallenger/ai_challenger_scene_train_20170904/scene_train_images_20170904/' 
annotations = '/data0/AIChallenger/ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json'
# a DataFlow you implement to produce [tensor1, tensor2, ..] lists from whatever sources:
df = MyDataFlow(train_dir, annotations, is_training=True, batch_size=batch_size, img_size=image_size)
# start 3 processes to run the dataflow in parallel
df = PrefetchDataZMQ(df, nr_proc=3)
df.reset_state()
scene_data = df.get_data()

saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("loading checkpoint...")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter('./logs', sess.graph)
    
    _x = x[:, :, :, ::-1]
    tf.summary.image('x', _x, 4)
    
    summary_op = tf.summary.merge_all()

    epoch_learning_rate = init_learning_rate
    for epoch in range(1, total_epochs + 1):
        if epoch % 30 == 0 :
            epoch_learning_rate = epoch_learning_rate / 10

        train_acc = 0.0
        train_loss = 0.0

        for step in range(1, iteration + 1):
            batch_data = next(scene_data)
            batch_x = batch_data['data']
            batch_y = batch_data['label']

            train_feed_dict = {
                x: batch_x,
                label: batch_y,
                learning_rate: epoch_learning_rate,
                training_flag: True
            }

            _, batch_loss = sess.run([train_op, Total_loss], feed_dict=train_feed_dict)
            batch_acc = accuracy.eval(feed_dict=train_feed_dict)

            print("epoch: %d/%d, iter: %d/%d, batch_loss: %.4f, batch_acc: %.4f \n" % (
                epoch, total_epochs, step, iteration, batch_loss, batch_acc))

            train_loss += batch_loss
            train_acc += batch_acc

            if step % 30 == 0 :
                summary_str = sess.run(summary_op, feed_dict=train_feed_dict)
                summary_writer.add_summary(summary=summary_str, global_step=epoch)
                summary_writer.flush()


        train_loss /= iteration # average loss
        train_acc /= iteration # average accuracy

        train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                          tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])

        test_acc, test_loss, test_summary = Evaluate(sess)

        summary_writer.add_summary(summary=train_summary, global_step=epoch)
        summary_writer.add_summary(summary=test_summary, global_step=epoch)
        summary_writer.flush()

        line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f \n" % (
            epoch, total_epochs, train_loss, train_acc, test_loss, test_acc)
        print(line)

        with open('./logs/logs.txt', 'a') as f:
            f.write(line)

        saver.save(sess=sess, save_path='./model/model.ckpt')
