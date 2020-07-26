import os, sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim

import classify_config as cfg

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# gen data
num_sample,num_batchsize = 20, 20
x_np, y_np = cfg.gen_data(num_sample*num_batchsize)

# gen model

bn_is_tf = False
if bn_is_tf:
    isTrainBn = tf.placeholder(tf.bool, name='is_train_bn')
    tfInput, tfLabel, tfoutput, end_point = cfg.model(isTrain=True, isTrainBn=isTrainBn)
else:
    tfInput, tfLabel, tfoutput, end_point = cfg.model(isTrain=True, isTrainBn=True)
if cfg.isQuant:
    tf.contrib.quantize.create_training_graph(input_graph=tf.get_default_graph(), quant_delay=500)
with tf.variable_scope('optimize'):
    pred = tf.arg_max(tf.nn.softmax(tfoutput), dimension=-1, output_type=tf.int32)
    train_acc = tf.reduce_mean(tf.cast(tf.equal(pred, tfLabel), tf.float32))
train_loss = slim.losses.sparse_softmax_cross_entropy(logits=tfoutput, labels=tfLabel)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(train_loss)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./board_train', sess.graph)
    tf.global_variables_initializer().run(session=sess)

    print([x.name.split(':')[0] for x in tf.global_variables()])
    print(sess.run(tf.global_variables()))
    for _ in range(10):
        for i in range(num_sample):
            tmpx = x_np[num_batchsize * i:num_batchsize * (i + 1), :, :, :]
            tmpy = y_np[num_batchsize * i:num_batchsize * (i + 1)]
            if bn_is_tf:
                res_acc,_ = sess.run([train_acc,train_op],feed_dict={tfInput:tmpx, tfLabel:tmpy, isTrainBn:True})
            else:
                res_acc,_ = sess.run([train_acc,train_op],feed_dict={tfInput:tmpx, tfLabel:tmpy})
            print('acc:{}'.format(res_acc))
    print(sess.run(tf.global_variables()))

    saver = tf.compat.v1.train.Saver()
    saver.save(sess=sess, save_path='./model/hxh')

print('done....')
