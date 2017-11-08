import tensorflow as tf
import numpy as np
import os
from tensorpack import imgaug, dataset, ModelDesc, InputDesc
from tensorpack.dataflow import (PrefetchDataZMQ, BatchData)
from dataflow_input import (MyDataFlow, data_augmentation)
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
iteration = 421
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

def resnet_model_fn(inputs, training):
    """Our model_fn for ResNet to be used with our Estimator."""

    network = resnet_model.imagenet_resnet_v2(
        resnet_size=18, num_classes=class_num, data_format=None)
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
l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
C_loss, _ = center_loss(logits, tf.cast(label, dtype=tf.int32), 0.95, class_num)

optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
# Batch norm requires update_ops to be added as a train_op dependency.
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(cost + l2_loss)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

val_dir = '/data0/AIChallenger/ai_challenger_scene_validation_20170908/scene_validation_images_20170908/'
annotations = '/data0/AIChallenger/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json'
# a DataFlow you implement to produce [tensor1, tensor2, ..] lists from whatever sources:
df = MyDataFlow(val_dir, annotations, is_training=False, batch_size=batch_size, img_size=(image_size, image_size))
# start 3 processes to run the dataflow in parallel
df = PrefetchDataZMQ(df, nr_proc=3)
df.reset_state()
scene_data_val = df.get_data()

train_dir = '/data0/AIChallenger/ai_challenger_scene_train_20170904/scene_train_images_20170904/' 
annotations = '/data0/AIChallenger/ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json'
# a DataFlow you implement to produce [tensor1, tensor2, ..] lists from whatever sources:
df = MyDataFlow(train_dir, annotations, is_training=True, batch_size=batch_size, img_size=(image_size, image_size))
# start 3 processes to run the dataflow in parallel
df = PrefetchDataZMQ(df, nr_proc=8)
df.reset_state()
scene_data = df.get_data()

saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model_train')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("loading checkpoint...")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter('./logs_train', sess.graph)
    
    _x = x[:, :, :, ::-1]
    tf.summary.image('x', _x, 4)
    
    summary_op = tf.summary.merge_all()

    epoch_learning_rate = init_learning_rate
    for epoch in range(1, total_epochs + 1):
        if epoch % 5 == 0 :
            epoch_learning_rate = epoch_learning_rate / 10
            if epoch_learning_rate <= 1e-5:
                epoch_learning_rate = epoch_learning_rate * 1e3

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

            _, batch_loss = sess.run([train_op, cost], feed_dict=train_feed_dict)
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

        with open('./logs_train/logs.txt', 'a') as f:
            f.write(line)

        saver.save(sess=sess, save_path='./model_train/model.ckpt')
