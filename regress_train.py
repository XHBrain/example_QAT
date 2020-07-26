import os, sys
import numpy as np
import tensorflow as tf

import regress_config as cfg

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

num_sample,num_batch = 500, 50
x_np, y_np = cfg.gen_data(num_sample*num_batch)
is_train_bn = tf.placeholder(tf.bool)
tf_input, tf_label, tf_output, end_point = cfg.model(is_train=True, is_train_bn=is_train_bn)

tf.contrib.quantize.create_training_graph(input_graph=tf.get_default_graph(), quant_delay=500)

loss = tf.reduce_mean(tf.square(tf_output-tf_label), name='Xreducemean')
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, name='Xmomentum')\
        .minimize(loss, name='Xminimize')

with tf.Session() as sess:
    tf.global_variables_initializer().run(session=sess)
    writer = tf.summary.FileWriter('./board', sess.graph)

    print([x.name.split(':')[0] for x in tf.global_variables()])
    print(sess.run(tf.global_variables()))
    for _ in range(1):
        for i in range(num_sample):
            tmpx = x_np[num_batch * i:num_batch * (i + 1), :, :, :]
            tmpy = y_np[num_batch * i:num_batch * (i + 1), :]
            res_loss, _ = sess.run([loss, train_op],
                                   # feed_dict={tf_input:tmpx, tf_label:tmpy})
                                   feed_dict={tf_input:tmpx, tf_label:tmpy, is_train_bn:True})
    print(sess.run(tf.global_variables()))

    """ proved overflow quantity bound to a boundary."""
    for float_x in np.linspace(cfg.input_min-1, cfg.input_max+1, 11):
        res = sess.run(end_point,
                       # feed_dict={tf_input: [[[[float_x]]]]})
                       feed_dict={tf_input: [[[[float_x]]]], is_train_bn:False})
        print('input {:.2f} output {}.'.format(float_x, res))

    saver = tf.compat.v1.train.Saver()
    saver.save(sess=sess, save_path='./model/hxh')

print('done....')
