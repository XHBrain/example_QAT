
import os, time
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

import classify_config as cfg

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sess = tf.Session()
with gfile.FastGFile('output_model/pb_model/frozen_model.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')  # 导入计算图

# 需要有一个初始化的过程
# sess.run(tf.global_variables_initializer())

# 需要先复原变量
# print('L1/kernel={}'.format(sess.run('L1/kernel:0')))

# 输入
input = sess.graph.get_tensor_by_name('tf_input:0')
output = sess.graph.get_tensor_by_name('Xoutput:0')

num_sample = 1000
x_np, y_np = cfg.gen_data(num_sample, noise=None)
acc = 0
total_time = 0
sess.run(output, feed_dict={input: x_np[0:1,:,:,:]})
sess.run(output, feed_dict={input: x_np[0:1,:,:,:]})
for i in range(num_sample):
    tic = time.time()
    ret = sess.run(output, feed_dict={input: x_np[i:i+1,:,:,:]})
    total_time += (time.time()-tic)
    acc += np.abs((np.argmax(ret)-y_np[i]))
print('float inference in pb file. acc={}, time={}'.format(1-(acc/num_sample), total_time/num_sample))
