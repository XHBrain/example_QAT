
import numpy as np
import matplotlib.pyplot as plt

num_data = 1000
net_node = 16
momentum = 0.9

#########################################################

data = np.random.random([2,num_data]).astype(np.float32)
label = (data[0,:]-0.5)*(data[1,:]-0.5)>0
label_int = label.astype(np.int8)

#########################################################

w1 = (np.random.random([net_node, 2]).astype(np.float32) - 0.5)/4
b1 = np.zeros([net_node,1], dtype=np.float32)
w2 = (np.random.random([1, net_node]).astype(np.float32) - 0.5)/4
b2 = np.zeros([1,1], dtype=np.float32)
a1, z1, a2, z2 = None, None, None, None
m_w1, m_b1, m_w2, m_b2 = 0.0, 0.0, 0.0, 0.0

#########################################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def g_sigmoid(x):
    return x*(1-x)
def relu(x):
    x[x<0] = 0
    return x
def g_relu(x):
    x[x>0] = 1
    return x

def pridect_float(x):
    global a1, z1, a2, z2
    a1 = np.dot(w1, x) + b1
    # z1 = sigmoid(a1)
    z1 = relu(a1)
    a2 = np.dot(w2,z1) + b2
    z2 = sigmoid(a2)

def gradien(lr, x,y):
    global w1, b1, w2, b2, a1, z1, a2, z2, m_w1, m_b1, m_w2, m_b2
    batchsize = len(y)
    g_a2 = -(y-z2)
    g_w2 = np.dot(g_a2,a1.T)/batchsize
    g_b2 = np.sum(g_a2, axis=1, keepdims=True)/batchsize
    # g_a1 = w2.T*g_a2*g_sigmoid(z1)
    g_a1 = w2.T*g_a2*g_relu(z1)
    g_w1 = np.dot(g_a1,x.T)/batchsize
    g_b1 = np.sum(g_a1, axis=1, keepdims=True)/batchsize

    m_w1 = momentum*m_w1 + g_w1
    m_b1 = momentum*m_b1 + g_b1
    m_w2 = momentum*m_w2 + g_w2
    m_b2 = momentum*m_b2 + g_b2
    w2 = w2 - lr * m_w2
    b2 = b2 - lr * m_b2
    w1 = w1 - lr * m_w1
    b1 = b1 - lr * m_b1

def loss(y):
    batchsize = len(y)
    return -np.sum(y*np.log(z2+1e-5)+(1-y)*np.log(1-z2+1e-5))/batchsize

#########################################################

for i in range(500):
    pridect_float(data)
    print(loss(label_int), np.sum((z2>0.5)==label)/len(label_int))
    gradien(0.9, data, label_int)

