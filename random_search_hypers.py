#!/usr/bin/python
# -*- encoding: utf8 -*-



import mxnet as mx
import numpy as np
import random
import os
import symbol.symbol as symbol
import core.config as config
import core.meric as meric
import core.io as io




# get hyper parameters
def get_params(lr_arr, lr_sch_arr, wd_arr, hypers):
    '''
    This method returns a tuple containing hyper-parameters randomly chosen
    from the given lists of their scope.
    params:
        lr_arr: a tuple or list or any structure that can be randomly choiced,
                indicating the scope from which learning rate is chosen
        lr_sch_arr: same structure as lr_arr, but indicates the ratio by which
                    the learning rate will be multiplied each 500 iters
        wd_arr: same structure as lr_arr, but suggesting the weight decay ratio
                which has the same effect as L2 regularization for anti-overfitting
        hypers: tuple or list or set of tuples which contain the hyper-parameters
                that have been assigned to the model for random search. Any tried
                hyper-parameters will be tupled and add to this structure to
                avoid duplicated search
    return:
        a tuple of the randomly chosen hyper parameter of learning rate,
        learning rate factor and weight decay ratio
    '''
    lr = random.choice(lr_arr)
    lr_sch = random.choice(lr_sch_arr)
    wd = random.choice(wd_arr)
    tp = lr, lr_sch, wd
    while tp in hypers:
        print("used hyper parameters, try another set")
        lr = random.choice(lr_arr)
        lr_sch = random.choice(lr_sch_arr)
        wd = random.choice(wd_arr)
        tp = lr, lr_sch, wd

    return tp


def tune_hyperparameters(batch_size, epoch, lr, lr_factor, wd):
    '''
    This method receives some hyper parameters and train the model with them to
    valid the accuracy of the model under these hyper parameters.
    params:
        batch_size: batch_size
        epoch: epoch
        lr: the initial learning rate used to train the model
        lr_factor: the current learning rate factor used in the training process
        wd: the current weight decay ratio used in this training process
    '''
    # get network symbol
    out = symbol.lenet5_symbol()

    ## module
    lr_scheduler = mx.lr_scheduler.FactorScheduler(step=500,factor=lr_factor,stop_factor_lr=1e-7)
    mod = mx.mod.Module(out, context=mx.gpu(),data_names=['img'],label_names=['label'])
    mod.bind(data_shapes=[('img',(batch_size,3,30,100))], label_shapes=[('label',(batch_size,4))])
    mod.init_params(mx.init.Xavier())
    mod.init_optimizer(
        optimizer='adam',
        optimizer_params=(('learning_rate',lr),
                        ('beta1',0.9),
                        ('wd',wd),
                        ('lr_scheduler', lr_scheduler))
        )


    ## training
    train_iter = io.get_record_iter('./datasets/train_list.rec',(3,30,100),4,batch_size)
    test_iter = io.get_record_iter('./datasets/test_list.rec',(3,30,100),4,batch_size)
    test_acc_list = []
    track_epoch = max(0, epoch - 10)
    for e in range(epoch):
        train_iter.reset()
        i = 0
        for batch in train_iter:
            mod.forward(batch)
            mod.backward()
            mod.update()

            # synchronize
            output = mod.get_outputs()
            train_loss = output[0].asnumpy()

            # keep track of validation accuracy
            if (e > track_epoch) and (i % 10 == 0):
                valid = mod.predict(test_iter, 16, always_output_list=True)

                test_score = valid[1].asnumpy()
                label = valid[2].asnumpy()
                test_acc = meric.accuracy(label, test_score)
                test_acc_list.append(test_acc)
            i += 1

        if e % 20 == 0:
            print("epoch {}".format(e))


    # average accuracy
    acc_chunk = np.array(test_acc_list[-20:])
    acc_chunk_avg = np.mean(acc_chunk, axis=0)
    print("average validation accuracy is: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(acc_chunk_avg[0],acc_chunk_avg[1],acc_chunk_avg[2],acc_chunk_avg[3],acc_chunk_avg[4]))

    return acc_chunk_avg


if __name__ == '__main__':

    batch_size = config.batch_size
    epoch = config.epoch
    wd_arr = tuple([n*2e-7 for n in range(1,5001)])
    lr_arr = tuple([n*1e-4 for n in range(1,101)])
    lr_factor_arr = tuple([n*0.05+0.2 for n in range(16)])


    # create log file and write headers if no existing log file
    logname = 'hyper-params.log'
    if not os.path.exists(logname):
        with open(logname, 'a+') as w:
            head = 'lr \t|\t lr_factor \t|\t wd \t|\t acc_0 \t|\t acc_1 \t|\t acc_2 \t|\t acc_3 \t|\t acc_4\n'
            w.write(head+'\n')


    # read all the tried hypers
    with open(logname, 'r') as r:
        read_hypers = lambda line: tuple([float(token) for token in line.split(', \t')[:3]])
        lines = r.readlines()[2:]
        hypers = set(map(read_hypers, lines))


    # try new hypers and write results
    with open(logname, 'a+') as w:
        for i in range(20):
            print('round {}:'.format(i))
            lr,lr_factor,wd = get_params(lr_arr, lr_factor_arr, wd_arr, hypers)
            tp = (lr,lr_factor,wd)
            hypers.add(tp)

            acc = tune_hyperparameters(batch_size, epoch, lr, lr_factor, wd)

            print('lr: {:.6f}, lr_facot: {:.2f}, wd: {:.8f}'.format(lr, lr_factor, wd))
            params = '{}, \t {}, \t {}, \t|\t {:.5f}, \t|\t {:.5f}, \t|\t {:.5f}, \t|\t {:.5f}, \t|\t {:.5f}\n'.format(lr,lr_factor,wd,acc[0],acc[1],acc[2],acc[3],acc[4])
            w.write(params)



