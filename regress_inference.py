import os
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import regress_config as cfg

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf_input, _, tf_output, _ = cfg.model(is_train=False, is_train_bn=False)

tf.contrib.quantize.create_eval_graph(input_graph=tf.get_default_graph())

with tf.Session() as sess:
    tf.global_variables_initializer().run(session=sess)
    writer = tf.summary.FileWriter('./board', sess.graph)

    saver = tf.compat.v1.train.Saver()
    saver.restore(sess=sess, save_path='./model/hxh')

    # save graph
    tf.train.write_graph(sess.graph_def, 'output_model/pb_model', 'model.pb')
    # 把图和参数结构一起
    freeze_graph.freeze_graph('output_model/pb_model/model.pb', '', False, './model/hxh', 'Xoutput',
                              'save/restore_all', 'save/Const:0', 'output_model/pb_model/frozen_model.pb',
                              True, "")

    converter = tf.lite.TFLiteConverter.from_session(sess, input_tensors=[tf_input], output_tensors=[tf_output])
    converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
    input_arrays = converter.get_input_arrays()
    converter.quantized_input_stats = {input_arrays[0]: (cfg.input_mean, cfg.input_std_dev)}
    tflite_model = converter.convert()
    open("./output_model/converted_model.tflite", "wb").write(tflite_model)
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for int_x in range(-2,257):
        float_x = cfg.int2float(int_x)
        interpreter.set_tensor(input_details[0]['index'], [[[[np.array(int_x, dtype=np.uint8)]]]])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print('%2.2f\t%2.2f\t%d\t%d' % (float_x, cfg.f(float_x), int_x, output_data[0][0]))

print('done....')
