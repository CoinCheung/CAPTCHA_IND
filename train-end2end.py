#!/usr/bin/python
# -*- encoding: utf8 -*-



import mxnet as mx
import numpy as np


batch_size = 16

## data iterator
# TODO: add a random seed generator
train_iter = mx.io.ImageRecordIter(
    path_imgrec='./datasets/train_list.rec',
    data_shape=(3,30,100),
    label_width=4,
    shuffle=True,
    seed = 0,
    batch_size=batch_size
)

train_iter.reset()
batch = train_iter.next()

#  print(dir(train_iter))

#  print(mx.nd.one_hot(batch.label[0],36))
#  print(batch.data[0])
#

## symbol LeNet-5
# TODO: check net structure if it does not work well
is_test=False
img = mx.sym.var('img')
label = mx.sym.var('label')
# 3x30x100
conv1 = mx.sym.Convolution(img, num_filter=32, kernel=(3,3), stride=(1,1), pad=(1,1), no_bias=False, name='conv1')
relu1 = mx.sym.Activation(conv1, act_type='relu', name='relu1')
pool1 = mx.sym.Pooling(relu1, kernel=(2,2),stride=(2,2), pad=(0,0), pool_type='max', name='pool1')
# 32x15x50
conv2 = mx.sym.Convolution(pool1, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1,1), no_bias=False, name='conv2')
relu2 = mx.sym.Activation(conv2, act_type='relu', name='relu2')
pool2 = mx.sym.Pooling(relu2, kernel=(2,2),stride=(2,2), pad=(1,0), pool_type='max', name='pool2')
# 64x8x25
conv3 = mx.sym.Convolution(pool2, num_filter=128, kernel=(3,3), stride=(1,1), pad=(1,1), no_bias=False, name='conv3')
relu3 = mx.sym.Activation(conv3, act_type='relu', name='relu3')
pool3 = mx.sym.Pooling(relu3, kernel=(2,2),stride=(2,2), pad=(0,1), pool_type='max', name='pool3')
# 128x4x13
fc1 = mx.sym.FullyConnected(pool3, num_hidden=1024, no_bias=False, flatten=True, name='fc1')
bn4 = mx.sym.BatchNorm(fc1, fix_gamma=False, use_global_stats=is_test, name='bn4')
relu4 = mx.sym.Activation(bn4, act_type='relu', name='relu4')
# batch_sizex1024
fc1 = mx.sym.FullyConnected(relu4, num_hidden=36*4, no_bias=False, flatten=True, name='fc2')
# batch_sizex(36x4)

softmax = mx.sym.log_softmax(fc1, axis=1)
label_one_hot = mx.sym.one_hot(label, 36)
label_2d = mx.sym.reshape(label_one_hot, shape=(-1,36*4))
cross_entropy = -mx.sym.sum(softmax * label_2d)

loss = mx.sym.MakeLoss(cross_entropy)


## module
#  print(train_iter.provide_data)
mod = mx.mod.Module(loss, context=mx.gpu(),data_names=['img'],label_names=['label'])
mod.bind(data_shapes=[('img',(batch_size,3,30,100))], label_shapes=[('label',(batch_size,4))])
mod.init_params(mx.init.Xavier())
mod.init_optimizer(
    optimizer='adam',
    optimizer_params=(('learning_rate',1e-3),('beta1',0.9))
    )









