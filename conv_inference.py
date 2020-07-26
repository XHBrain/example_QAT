import os
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

######################################
test_x = np.linspace(-0.5,0.5,50).reshape(-1,1,1,1)
test_y = np.array(np.ceil(np.sin(test_x*np.pi*6)).reshape(-1,1), dtype=np.int32)
# plt.scatter(test_x.reshape(-1).tolist(), test_y.reshape(-1),)
# plt.show()
#######################################
tf_x = tf.placeholder(dtype=tf.float32, shape=[None,1,1,1], name='h_input')
tf_label = tf.placeholder(dtype=tf.float32, shape=[None,1], name='h_label')
is_training = False
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
# x = tf.fake_quant_with_min_max_args(x, min=-1., max=1., name='h_fake_1')
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
x = tf.fake_quant_with_min_max_args(x, min=0., max=6., name='h_fake_2')
x = tf.layers.conv2d(inputs=x, filters=12,
        kernel_size=1, strides=[1, 1], kernel_initializer=tf.glorot_uniform_initializer(),
        padding='VALID',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4), use_bias=True, name='h_conv_3')
x = tf.nn.sigmoid(x, name='h_act_3')
# x = tf.fake_quant_with_min_max_args(x, min=0., max=1., name='h_fake_3')
x = tf.layers.conv2d(inputs=x, filters=1,
        kernel_size=1, strides=[1, 1], kernel_initializer=tf.glorot_uniform_initializer(),
        padding='VALID',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4), use_bias=True, name='h_conv_4')
x = tf.reshape(x,[-1,1], 'h_reshape')
y = tf.identity(x, 'h_output')
loss = tf.reduce_mean(tf.square(tf_label-y), name='h_reducemean')
#############################################

tf.contrib.quantize.create_eval_graph(input_graph=tf.get_default_graph())

with tf.Session() as sess:
    tf.global_variables_initializer().run(session=sess)

    saver = tf.compat.v1.train.Saver()
    saver.restore(sess=sess, save_path='./model/hxh')

    # 保存图
    tf.train.write_graph(sess.graph_def, 'output_model/pb_model', 'model.pb')
    # 把图和参数结构一起
    freeze_graph.freeze_graph('output_model/pb_model/model.pb', '', False, './model/hxh', 'h_output',
                              'save/restore_all', 'save/Const:0', 'output_model/pb_model/frozen_model.pb',
                              True, "")

    writer = tf.summary.FileWriter('./board', sess.graph)

    converter = tf.lite.TFLiteConverter.from_session(sess, input_tensors=[tf_x], output_tensors=[y])
    converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
    input_arrays = converter.get_input_arrays()
    converter.quantized_input_stats = {input_arrays[0]: (128., 118.)}  # mean, std_dev
    tflite_model = converter.convert()
    open("./output_model/converted_model.tflite", "wb").write(tflite_model)
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    res_y = []
    for tmp_x  in test_x:
        interpreter.set_tensor(input_details[0]['index'], [np.array((tmp_x+0.5)*255,dtype=np.uint8)])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        res_y.append(output_data[0])
    print('loss = {}'.format(np.mean(test_y-res_y)))



    print('done....')
