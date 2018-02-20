#!/usr/bin/python
# -*- encoding: utf8 -*-



import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import random


batch_size = 90



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
conv3 = mx.sym.Convolution(pool2, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1,1), no_bias=False, name='conv3')
relu3 = mx.sym.Activation(conv3, act_type='relu', name='relu3')
pool3 = mx.sym.Pooling(relu3, kernel=(2,2),stride=(2,2), pad=(0,1), pool_type='max', name='pool3')
# 128x4x13
fc1 = mx.sym.FullyConnected(pool3, num_hidden=1024, no_bias=False, flatten=True, name='fc1')
bn4 = mx.sym.BatchNorm(fc1, fix_gamma=False, use_global_stats=is_test, name='bn4')
relu4 = mx.sym.Activation(bn4, act_type='relu', name='relu4')
# batch_sizex1024
fc1 = mx.sym.FullyConnected(relu4, num_hidden=36*4, no_bias=False, flatten=True, name='fc2')
# batch_sizex(36x4)
# loss
scores = mx.sym.reshape(fc1,shape=(-1,4*36))
softmax = mx.sym.softmax(scores, axis=1)
softmax_log = mx.sym.log(softmax)
#  softmax_log = mx.sym.log_softmax(scores, axis=1)
label_one_hot = mx.sym.one_hot(label, 36)
label_2d = mx.sym.reshape(label_one_hot, shape=(-1,36*4))
#  product = softmax_log * label_one_hot
product = softmax_log * label_2d
cross_entropy = -mx.sym.mean(mx.sym.sum(product, axis=1))
loss = mx.sym.MakeLoss(cross_entropy)
# pred
score_pred = mx.sym.BlockGrad(mx.sym.reshape(softmax, shape=(-1,4,36)))
out = mx.sym.Group([loss, score_pred])


## module
#  print(train_iter.provide_data)
#  lr_scheduler = mx.lr_scheduler.FactorScheduler(step=500,factor=0.8,stop_factor_lr=1e-6)
lr_scheduler = mx.lr_scheduler.FactorScheduler(step=500,factor=0.8,stop_factor_lr=1e-6)
mod = mx.mod.Module(out, context=mx.gpu(),data_names=['img'],label_names=['label'])
mod.bind(data_shapes=[('img',(batch_size,3,30,100))], label_shapes=[('label',(batch_size,4))])
mod.init_params(mx.init.Xavier())
mod.init_optimizer(
    optimizer='adam',
    optimizer_params=(('learning_rate',1e-3),
                      ('beta1',0.9),
                      ('wd',5e-4),
                      ('lr_scheduler', lr_scheduler))
    )


## data iterator
# the datasets is shuffled after each reset(), thus after each reset() the data
# in each batch should be different though given the same seed, the behavior of
# the program at each iter and epoch should be the same
# TODO: wrap the iterator with a function so as to shuffle the dataset given
# seed
def get_train_iter(data_record, shape, label_width):
    '''
    A wrapper of the ImageRecordIter, to assign the seed randomly
    '''
    seed = random.randint(0, 5000)
    return mx.io.ImageRecordIter(
        path_imgrec='./datasets/train_list.rec',
        data_shape=(3,30,100),
        label_width=4,
        shuffle=True,
        seed = seed,
        batch_size=batch_size
    )

seed = random.randint(0, 5000)
test_iter = mx.io.ImageRecordIter(
    path_imgrec='./datasets/test_list.rec',
    data_shape=(3,30,100),
    label_width=4,
    shuffle=True,
    seed = seed,
    batch_size=8*batch_size
)



## accuracy
def accuracy(scores, label_real):
    label_pred = np.argmax(scores, axis=2)
    same = np.sum(label_pred == label_real, axis=1)
    #  print(label_pred == label_real)
    acc0 = np.sum(same==0)/same.size
    acc1 = np.sum(same==1)/same.size
    acc2 = np.sum(same==2)/same.size
    acc3 = np.sum(same==3)/same.size
    acc4 = np.sum(same==4)/same.size
    return [acc0, acc1, acc2, acc3, acc4]


## draw
def draw_loss(train_loss, fig_num):
    '''
    params:
        train_loss: a list containing train loss in the training process
    '''
    f, ax = plt.subplots(fig_num)
    plt.ion()
    ax.plot(np.array(train_loss))
    ax.set_title('train_loss')
    plt.show()

def draw_acc(acc, fig_num):
    '''
    params:
        acc: accuracy list. Should be of shape (valid times, 5) which indicates
        the accuracy of successful prediction of 0-4 characters
    '''
    acc_array = np.array(acc)

    fig = plt.figure()
    plt.ioff()
    _, ax = plt.subplots(5,1, sharex=True, sharey=False, num=fig_num)
    ax[0].plot(acc_array[:,0])
    ax[0].set_title("0 character successfully predicted")
    ax[1].plot(acc_array[:,1])
    ax[1].set_title("1 character successfully predicted")
    ax[2].plot(acc_array[:,2])
    ax[2].set_title("2 character successfully predicted")
    ax[3].plot(acc_array[:,3])
    ax[3].set_title("3 character successfully predicted")
    ax[4].plot(acc_array[:,4])
    ax[4].set_title("4 character successfully predicted")
    plt.tight_layout()
    plt.show()



## training
epoch = 80
train_loss_list = []
test_acc_list = []
end_epoch = False
train_iter = get_train_iter('./datasets/train_list.rec',(3,30,100),4)
for e in range(epoch):
    train_iter.reset()
    i = 0
    for batch in train_iter:
        mod.forward(batch)
        mod.backward()
        mod.update()

        # keep track of training loss
        output = mod.get_outputs()
        train_loss = output[0].asnumpy()
        train_loss_list.append(train_loss[0])

        # keep track of validation accuracy
        if i % 10 == 0:
            test_iter.reset()
            test_batch = test_iter.next()
            mod.forward(test_batch)
            test_output = mod.get_outputs()
            test_loss = test_output[0].asnumpy()
            test_scores = test_output[1].asnumpy()
            test_acc = accuracy(test_scores, test_batch.label[0].asnumpy())
            test_acc_list.append(test_acc)
            if test_acc[-1] > 0.85:
                print("accuracy {}, better than 85% at epoch {} iter {}".format(test_acc[-1],e,i))
                end_epoch = True
                break

        i += 1

        if i % 50 == 0:
            #  print("epoch: {} iter: {} train_loss: {} train_acc: {}".format(e,i,train_loss[-1],train_acc))
            print("epoch: {} iter: {} train_loss: {}".format(e,i,train_loss[-1]))

    print("epoch: {} validation_loss: {} validation_acc: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}"\
          .format(e,test_loss,test_acc_list[-1][0],test_acc_list[-1][1],test_acc_list[-1][2],test_acc_list[-1][3],test_acc_list[-1][4]))
    if(end_epoch == True):
        break

draw_loss(train_loss_list, 1)
draw_acc(test_acc_list, 2)






