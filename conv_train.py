import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

######################################
num_sample = 1
num_batch = 50
x_np = np.random.rand(num_batch*num_sample,1,1,1)-0.5
y_np = np.array(np.ceil(np.sin(x_np*np.pi*6)).reshape(-1,1), dtype=np.int32)
test_x = np.linspace(-0.5,0.5,50).reshape(-1,1,1,1)
test_y = np.array(np.ceil(np.sin(test_x*np.pi*6)).reshape(-1,1), dtype=np.int32)
# plt.scatter(x_np.reshape(-1).tolist(), y_np.reshape(-1),)
# plt.scatter(test_x.reshape(-1).tolist(), test_y.reshape(-1),)
# plt.show()
#######################################
tf_x = tf.placeholder(dtype=tf.float32, shape=[None,1,1,1], name='h_input')
tf_label = tf.placeholder(dtype=tf.float32, shape=[None,1], name='h_label')
is_training = tf.placeholder(dtype=tf.bool, shape=None, name='h_istrain')
x = tf.layers.conv2d(inputs=tf_x, filters=12,
        kernel_size=1, strides=[1, 1], kernel_initializer=tf.glorot_uniform_initializer(),
        padding='VALID',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4), use_bias=True, name='h_conv_1')
x = tf.layers.batch_normalization(inputs=x, momentum=0.5, epsilon=1e-3,
                                         scale=True,
                                         center=True,
                                         training=is_training,
                                         fused=False,
                                         name='h_bn_1')
x = tf.nn.tanh(x, name='h_act_1')
x = tf.layers.conv2d(inputs=x, filters=12,
        kernel_size=1, strides=[1, 1], kernel_initializer=tf.glorot_uniform_initializer(),
        padding='VALID',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4), use_bias=False, name='h_conv_2')
x = tf.layers.batch_normalization(inputs=x, momentum=0.5, epsilon=1e-3,
                                         scale=True,
                                         center=True,
                                         training=is_training,
                                         fused=False,
                                         name='h_bn_2')
x = tf.nn.relu6(x, name='h_act_2')
x = tf.layers.conv2d(inputs=x, filters=12,
        kernel_size=1, strides=[1, 1], kernel_initializer=tf.glorot_uniform_initializer(),
        padding='VALID',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4), use_bias=True, name='h_conv_3')
x = tf.nn.sigmoid(x, name='h_act_3')
x = tf.layers.conv2d(inputs=x, filters=1,
        kernel_size=1, strides=[1, 1], kernel_initializer=tf.glorot_uniform_initializer(),
        padding='VALID',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4), use_bias=True, name='h_conv_4')
x = tf.reshape(x,[-1,1], 'h_reshape')
y = tf.identity(x, 'h_output')
loss = tf.reduce_mean(tf.square(tf_label-y), name='h_reducemean')
#############################################

tf.contrib.quantize.create_training_graph(input_graph=tf.get_default_graph(), quant_delay=10000)

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, name='h_MomentumOptimizer').minimize(loss, name='h_minimize')

with tf.Session() as sess:
    tf.global_variables_initializer().run(session=sess)
    for _ in range(100):
        for i in range(num_sample):
            tmpx = x_np[num_batch*i:num_batch*(i+1),:,:,:]
            tmpy = y_np[num_batch*i:num_batch*(i+1),:]
            res_loss, _ = sess.run([loss, train_op], feed_dict={is_training:True, tf_x:tmpx, tf_label: tmpy })
            print(res_loss)
    print('train done.')

    res_y, res_loss = sess.run([y, loss], feed_dict={is_training: False, tf_x: test_x, tf_label: test_y})
    print('test loss: {}'.format(res_loss))

    saver = tf.compat.v1.train.Saver()
    saver.save(sess=sess, save_path='./model/hxh')

    writer = tf.summary.FileWriter('./board', sess.graph)

print('done....')
