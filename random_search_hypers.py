#!/usr/bin/python
# -*- encoding: utf8 -*-



import mxnet as mx
import numpy as np
import random
import os
import symbol.symbol as symbol
import core.config as config



## data iterator
# the datasets is shuffled after each reset(), thus after each reset() the data
# in each batch should be different though given the same seed, the behavior of
# the program at each iter and epoch should be the same
def get_train_iter():
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



def get_test_iter():
    seed = random.randint(0, 5000)
    return mx.io.ImageRecordIter(
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



# get hyper parameters
def get_params(lr_arr, lr_sch_arr, wd_arr, hypers):
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


def tune_hyperparameters(batch_size, epoch, is_test, lr, lr_factor, wd):
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
    train_loss_list = []
    test_acc_list = []
    end_epoch = False
    track_epoch = max(0, epoch - 10)
    train_iter = get_train_iter()
    test_iter = get_test_iter()
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
            #  train_loss_list.append(train_loss[0])

            # keep track of validation accuracy
            if (e > track_epoch) and (i % 10 == 0):
                test_iter.reset()
                test_batch = test_iter.next()
                mod.forward(test_batch)
                test_output = mod.get_outputs()
                test_scores = test_output[1].asnumpy()
                test_acc = accuracy(test_scores, test_batch.label[0].asnumpy())
                test_acc_list.append(test_acc)
                if test_acc[-1] > 0.9:
                    print("accuracy {}, better than 85% at epoch {} iter {}".format(test_acc[-1],e,i))
                    end_epoch = True
                    break
            i += 1

        if(end_epoch == True):
            break

    # average accuracy
    acc_chunk = np.array(test_acc_list[-20:])
    acc_chunk_avg = np.mean(acc_chunk, axis=0)
    print("average validation accuracy is: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(acc_chunk_avg[0],acc_chunk_avg[1],acc_chunk_avg[2],acc_chunk_avg[3]))

    return acc_chunk_avg


if __name__ == '__main__':

    batch_size = config.batch_size
    epoch = config.epoch
    wd_arr = tuple([n*2e-7 for n in range(1,5001)])
    lr_arr = tuple([n*1e-4 for n in range(1,101)])
    lr_factor_arr = tuple([n*0.05+0.6 for n in range(9)])


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
        for i in range(35):
            print(i)
            lr,lr_factor,wd = get_params(lr_arr, lr_factor_arr, wd_arr, hypers)
            tp = (lr,lr_factor,wd)
            hypers.add(tp)

            acc = tune_hyperparameters(batch_size, epoch, False, lr, lr_factor, wd)

            print('lr: {:.6f}, lr_facot: {:.2f}, wd: {:.8f}'.format(lr, lr_factor, wd))
            params = '{}, \t {}, \t {}, \t|\t {:.5f}, \t|\t {:.5f}, \t|\t {:.5f}, \t|\t {:.5f}, \t|\t {:.5f}\n'.format(lr,lr_factor,wd,acc[0],acc[1],acc[2],acc[3],acc[4])
            w.write(params)



