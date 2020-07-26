import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import tensorflow as tf


isQuant = True

""" input data range """
input_min = -1.
input_max = 1.
""" q = std_dev*r-mean . the scope of q between 0 to 255.
    notice: if q equaled 256 overflowing, the input data will become 0 not 255. """
input_mean = 128 #(255-0)/(input_max-input_min)*(0-input_min)
input_std_dev = 128 #(255-0)/(input_max-input_min)

def int2float(q):
    return (q-input_mean)/input_std_dev

def float2int(r):
    return np.array(r*input_std_dev+input_mean, dtype=np.uint8)

def gen_data(num_sample, noise=0.1):
    x, y = make_moons(n_samples=num_sample, noise=noise)
    ### rescale to [-1, 1]
    x[:, 0] = 2 / 3 * x[:, 0] - 1 / 3
    x[:, 1] = (4 * x[:, 1] - 1) / 3
    # plt.scatter(x[:, 0], x[:, 1], marker='o', c=y); plt.show()
    return x.reshape(-1, 1, 1, 2), y

def model(isTrain, isTrainBn):
    end_point = []
    tf_input = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, 2], name='tf_input')
    tf_label = tf.placeholder(dtype=tf.int32, shape=[None], name='tf_label')
    if isTrain and isQuant:
        tf_input_1 = tf.fake_quant_with_min_max_args(tf_input, input_min, input_max, name='x0_1')
    else:
        tf_input_1 = tf_input
    # with tf.variable_scope('model'):
    x = tf.layers.separable_conv2d(tf_input_1, filters=10000, kernel_size=1, use_bias=False, name='L1')
    x = tf.layers.batch_normalization(x, training=isTrainBn, fused=True, name='L1_bn')
    with tf.variable_scope('L1_hard_swish'):
        x1 = tf.nn.relu6(x + 3)
        # x1 = tf.fake_quant_with_min_max_args(x1, 0, 6)
        x = x * x1 * 0.16666667
    x = tf.layers.conv2d(x, filters=4, kernel_size=1, use_bias=False, name='L2')
    x = tf.layers.batch_normalization(x, training=isTrainBn, fused=True, name='L2_bn')
    with tf.variable_scope('L2_hard_swish'):
        x1 = tf.nn.relu6(x + 3)
        # x1 = tf.fake_quant_with_min_max_args(x1, 0, 6)
        x = x * x1 * 0.16666667
    if isQuant: x = tf.fake_quant_with_min_max_args(x, 0, 6)
    x = tf.layers.conv2d(x, filters=2, kernel_size=1, use_bias=True, name='FCN')
    x = tf.layers.flatten(x, name='Xflatten')
    x = tf.identity(x, 'Xoutput')
    end_point.append(x)
    # if (not isTrain) and isQuant:
    #     x = tf.fake_quant_with_min_max_args(x, -1, 1, name='x5')
    return tf_input, tf_label, x, end_point

def checkup():
    print(input_mean, input_std_dev)
    print(int2float(0), int2float(255))
    print(float2int(input_min), float2int(input_max))
    print(gen_data(10))

if __name__ == '__main__':
    checkup()

