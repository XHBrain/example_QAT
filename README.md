# example_QAT
the example of Quant Aware Training with tensorflow1.5

#### 使用记录
主要有用的是min_train2.py, min_inference.py, min_config.py。
其他都是验证可靠性与测试是否存在bug。

#### 经验
通常conv+BN+activate,但tflite有bug，不同组合会有不同的bug
1. conv之后加batchnorm，量化训练会缺少fake note。
原因是此时的batchnorm会自动变成fused batchnorm和前面卷积合并在一起，所以却少了卷积运算后的fake note。
解决方法是设置batchnorm里面一个参数fused=False。

2. BN的training参数是确定的，tf.placeholder出bug

3. 激活函数缺fake比较好办，通常都知道激活函数的输出范围。因此遇到bug，只能选择这个然后手动插入fake note。

4. 量化训练，设置tf.contrib.quantize.create_training_graph中的quant_delay时。
如果设置为0，graph就会缺少fake_quantization_step节点，目前不清楚缺少这个note会不会带来bug。
按照逻辑是不会影响的，但谁知道呢，先记录下来。

#### reference
Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference, Benoit-Jacob, 2017
https://stackoverflow.com/questions/56856245/how-to-properly-inject-fake-quant-operations-in-a-graph
https://stackoverflow.com/questions/52807152/tf-fake-quant-with-min-max-vars-is-a-differentiable-function
https://stackoverflow.com/questions/50524897/what-are-the-differences-between-tf-fake-quant-with-min-max-args-and-tf-fake-qua
https://github.com/tensorflow/tensorflow/issues/15685
https://www.tensorflow.org/api_docs/python/tf/quantization

##### 后话
基本在tf1上把QAT该趟的坑都躺过了，成功用QAT跑通人脸特征提取，精度损失不到2%。。tf2的QAT方法已经放出来了，后面跟进之后再更新吧。