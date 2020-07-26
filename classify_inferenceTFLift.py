
import os, time
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

import classify_config as cfg

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

interpreter = tf.lite.Interpreter(model_path="./output_model/converted_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


num_sample = 1000
x_np, y_np = cfg.gen_data(num_sample, noise=None)
acc = 0
total_time = 0
for i in range(num_sample):
    tic = time.time()
    interpreter.set_tensor(input_details[0]['index'], cfg.float2int(x_np[i:i+1,:,:,:]))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    total_time += (time.time() - tic)
    acc += np.abs((np.argmax(output_data) - y_np[i]))
print('uint8 inference in tflite. acc={}, time={}'.format(1 - (acc / num_sample), total_time / num_sample))
