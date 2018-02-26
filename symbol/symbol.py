#!/usr/bin/python
# -*- encoding: utf8 -*-


import mxnet as mx
import core.config as config



def lenet5_symbol():
    '''
    This method returns the symbol of the lenet-5 network.
    The network structure is:
        1. convolutional layer: 32, 3x3 filters, padding (1,1), stride 1, relu
           max pooling: 2x2, stride 2, pading (0,0)
           drop out: p=0.2
        2. convolutional layer: 64, 3x3 filters, padding (1,1), stride 1, relu
           max pooling: 2x2, stride 2, padding (1,0)
        3. convolutional layer: 128, 3x3 filters, padding (1,1) stride 1, relu
           max pooling: 2x2, stride 2, padding (0,1)
        4. Fully Connected: 1024 hidden nodes, batch norm, relu
        5. Fully Connected: 36*4 hidden nodes

    return:
        a symbol group of softmax cross entropy, softmax scores and input label
    '''
    img = mx.sym.var('img')
    label = mx.sym.var('label')
    # 3x30x100
    conv1 = mx.sym.Convolution(img, num_filter=32, kernel=(3,3), stride=(1,1), pad=(1,1), no_bias=False, name='conv1')
    relu1 = mx.sym.Activation(conv1, act_type='relu', name='relu1')
    pool1 = mx.sym.Pooling(relu1, kernel=(2,2),stride=(2,2), pad=(0,0), pool_type='max', name='pool1')
    dp1 = mx.sym.Dropout(pool1, 0.2, 'training', name='dropout1')
    # 32x15x50
    conv2 = mx.sym.Convolution(dp1, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1,1), no_bias=False, name='conv2')
    relu2 = mx.sym.Activation(conv2, act_type='relu', name='relu2')
    pool2 = mx.sym.Pooling(relu2, kernel=(2,2),stride=(2,2), pad=(1,0), pool_type='max', name='pool2')
    # 64x8x25
    conv3 = mx.sym.Convolution(pool2, num_filter=128, kernel=(3,3), stride=(1,1), pad=(1,1), no_bias=False, name='conv3')
    relu3 = mx.sym.Activation(conv3, act_type='relu', name='relu3')
    pool3 = mx.sym.Pooling(relu3, kernel=(2,2),stride=(2,2), pad=(0,1), pool_type='max', name='pool3')
    # 128x4x13
    fc1 = mx.sym.FullyConnected(pool3, num_hidden=1024, no_bias=False, flatten=True, name='fc1')
    bn4 = mx.sym.BatchNorm(fc1, fix_gamma=False, name='bn4')
    relu4 = mx.sym.Activation(bn4, act_type='relu', name='relu4')
    # batch_sizex1024
    fc2 = mx.sym.FullyConnected(relu4, num_hidden=36*4, no_bias=False, flatten=True, name='fc2')
    # batch_sizex(36x4)

    # loss
    scores = mx.sym.reshape(fc2,shape=(-1,4*36))
    softmax = mx.sym.softmax(scores, axis=1)
    softmax_log = mx.sym.log(softmax)
    label_one_hot = mx.sym.one_hot(label, 36)
    label_2d = mx.sym.reshape(label_one_hot, shape=(-1,36*4))
    product = softmax_log * label_2d
    cross_entropy = -mx.sym.mean(mx.sym.sum(product, axis=1))
    loss = mx.sym.MakeLoss(cross_entropy)
    # predicted score
    score_pred = mx.sym.BlockGrad(mx.sym.reshape(softmax, shape=(-1,4,36)))

    out = mx.sym.Group([loss, score_pred, label])

    return out

