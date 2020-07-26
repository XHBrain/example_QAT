import os, time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import classify_config as cfg

""" :parameter """
is_quantized_uint8 = True

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tfInput, _, tfoutput, _ = cfg.model(isTrain=True, isTrainBn=False)

if cfg.isQuant:
    tf.contrib.quantize.create_eval_graph(input_graph=tf.get_default_graph())

sess = tf.Session()
writer = tf.summary.FileWriter('./board_inference', sess.graph)

saver = tf.compat.v1.train.Saver()
saver.restore(sess=sess, save_path='./model/hxh')

""" save with pb"""
# save graph
tf.train.write_graph(sess.graph_def, 'output_model/pb_model', 'model.pb')
# 把图和参数结构一起
freeze_graph.freeze_graph('output_model/pb_model/model.pb', '', False, './model/hxh', 'Xoutput',
                          'save/restore_all', 'save/Const:0', 'output_model/pb_model/frozen_model.pb',
                          True, "")

if cfg.isQuant:
    converter = tf.lite.TFLiteConverter.from_session(sess, input_tensors=[tfInput], output_tensors=[tfoutput])
    if is_quantized_uint8:
        converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
        input_arrays = converter.get_input_arrays()
        converter.quantized_input_stats = {input_arrays[0]: (cfg.input_mean, cfg.input_std_dev)}
    tflite_model = converter.convert()
    open("./output_model/converted_model.tflite", "wb").write(tflite_model)
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    num_sample = 1000
    x_np, y_np = cfg.gen_data(num_sample, noise=None)
    acc = 0
    total_time = 0
    if is_quantized_uint8:
        interpreter.set_tensor(input_details[0]['index'], cfg.float2int(x_np[0:1, :, :, :]))
        interpreter.set_tensor(input_details[0]['index'], cfg.float2int(x_np[0:1, :, :, :]))
        interpreter.set_tensor(input_details[0]['index'], cfg.float2int(x_np[0:1, :, :, :]))
    else:
        interpreter.set_tensor(input_details[0]['index'], np.array(x_np[0:1, :, :, :], dtype=np.float32))
        interpreter.set_tensor(input_details[0]['index'], np.array(x_np[0:1, :, :, :], dtype=np.float32))
        interpreter.set_tensor(input_details[0]['index'], np.array(x_np[0:1, :, :, :], dtype=np.float32))
    for i in range(num_sample):
        tic = time.time()
        if is_quantized_uint8:
            interpreter.set_tensor(input_details[0]['index'], cfg.float2int(x_np[i:i + 1, :, :, :]))
        else:
            interpreter.set_tensor(input_details[0]['index'], np.array(x_np[i:i+1,:,:,:],dtype=np.float32))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        total_time += (time.time() - tic)
        acc += np.abs((np.argmax(output_data) - y_np[i]))
    print('uint8 inference in tflite. acc={}, time={}'.format(1 - (acc / num_sample), total_time / num_sample))

    """ darw the interface"""
    # plt_x, plt_y, plt_o = [], [], []
    # for int_x in range(0,256,8):
    #     for int_y in range(0,256,8):
    #         plt_x.append(int_x)
    #         plt_y.append(int_y)
    #         int_input = np.array([[[[int_x,int_y]]]],dtype=np.uint8)
    #         interpreter.set_tensor(input_details[0]['index'], int_input)
    #         interpreter.invoke()
    #         output_data = interpreter.get_tensor(output_details[0]['index'])
    #         plt_o.append(np.argmax(output_data))
    # plt.scatter(plt_x, plt_y, marker='o', c=plt_o); plt.show()
print('done....')
