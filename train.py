#!/usr/bin/python
# -*- encoding: utf8 -*-



import mxnet as mx
import numpy as np
import core.config as config
import core.io as io
import core.module as module
import core.meric as meric
import core.visualize as visualize




def train_model(batch_size, epoch, verbose):

    mod = module.get_module()

    ## training
    train_loss_list = []
    test_acc_list = []
    end_epoch = False
    train_iter = io.get_record_iter('./datasets/train_list.rec',(3,30,100),4,batch_size)
    test_iter = io.get_record_iter('./datasets/test_list.rec',(3,30,100),4,batch_size*8)
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

            # keep track of validation accuracy each 10 iters
            if i % 10 == 0:
                test_iter.reset()
                test_batch = test_iter.next()
                mod.forward(test_batch)
                test_output = mod.get_outputs()
                test_loss = test_output[0].asnumpy()
                test_scores = test_output[1].asnumpy()
                test_acc = meric.accuracy(test_scores, test_batch.label[0].asnumpy())
                test_acc_list.append(test_acc)
                if test_acc[-1] > 0.9:
                    print("accuracy {}, better than 90% at epoch {} iter {}".format(test_acc[-1],e,i))
                    end_epoch = True
                    break

            i += 1
            if verbose and i % 50 == 0:
                print("epoch: {} iter: {} train_loss: {}".format(e,i,train_loss[-1]))

        # print acc after each epoch
        if verbose:
            print("epoch: {} validation_loss: {} validation_acc: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}"\
            .format(e,test_loss,test_acc_list[-1][0],test_acc_list[-1][1],test_acc_list[-1][2],test_acc_list[-1][3],test_acc_list[-1][4]))
        if(end_epoch == True):
            break

    # average accuracy
    acc_chunk = np.array(test_acc_list[-20:])
    acc_chunk_avg = np.mean(acc_chunk, axis=0)

    # save model and parameters
    mod.save_checkpoint("./model_export/lenet5", epoch, True)

    # draw the training loss and validation accuracy
    if verbose:
        visualize.draw_loss(train_loss_list, 1)
        visualize.draw_acc(test_acc_list, 2)

    return acc_chunk_avg


if __name__ == '__main__':

    batch_size = config.batch_size
    epoch = config.epoch
    verbose = config.verbose

    acc = train_model(batch_size, epoch, verbose)

    print('the final average accuracy over the last twenty validation is: ')
    print('[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(acc[0],acc[1],acc[2],acc[3],acc[4]))


