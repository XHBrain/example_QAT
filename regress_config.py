import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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
    return r*input_std_dev+input_mean

def f(x):
    return x*1.0

def gen_data(num_sample):
    x_np = np.random.uniform(input_min, input_max, num_sample)
    y_np = f(x_np)
    # plt.scatter(x_np, y_np); plt.show()
    return x_np.reshape(-1, 1, 1, 1), y_np.reshape(-1, 1)

def model(is_train, is_train_bn):
    end_point = []
    tf_input = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, 1], name='tf_input')
    tf_label = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='tf_label')
    if is_train:
        tf_input_1 = tf.fake_quant_with_min_max_args(tf_input, input_min, input_max, name='x0_1')
    else:
        tf_input_1 = tf_input
    with tf.variable_scope('X1'):
        x = tf.layers.conv2d(tf_input_1, 1, 1, use_bias=True, name='X1')
        end_point.append(x)
        x = tf.layers.batch_normalization(x, training=is_train_bn, name='X1/bn', fused=False)
        end_point.append(x)
        with tf.variable_scope('hard_swish'):
            x1 = tf.nn.relu6(x)-3
            x1 = tf.fake_quant_with_min_max_args(x1, -3, 3)
            # x1 = tf.nn.relu6(x + 3)
            # x1 = tf.fake_quant_with_min_max_args(x1, 0, 6)
            # x = x * x1 * 0.16666667
            end_point.append(x)
    # with tf.variable_scope("X2"):
    #     x = tf.layers.conv2d(x, 1, 1, use_bias=False, name='x2')
    #     end_point.append(x)
    #     x = tf.layers.batch_normalization(x, training=is_train_bn, name='x2/bn', fused=True)
    #     end_point.append(x)
    #     with tf.variable_scope('hard_swish'):
    #         x1 = tf.nn.relu6(x + 3)
    #         # x1 = tf.fake_quant_with_min_max_args(x1, 0, 6)
    #         x = x * x1 * 0.16666667
    #         end_point.append(x)
    x = tf.layers.flatten(x, name='Xflatten')
    x = tf.identity(x, 'Xoutput')
    end_point.append(x)
    if not is_train:
        # todo: how to ues
        x = tf.fake_quant_with_min_max_args(x, -1, 1, name='x5'); end_point.append(x)
    return tf_input, tf_label, x, end_point

def checkup():
    print(input_mean, input_std_dev)
    print(int2float(0), int2float(255))
    print(float2int(input_min), float2int(input_max))
    print(gen_data(10))

if __name__ == '__main__':
    checkup()
