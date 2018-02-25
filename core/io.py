#!/usr/bin/python
# -*- encoding: utf8 -*-


import mxnet as mx
import random


## data iterator
# the datasets is shuffled after each reset(), thus after each reset() the data
# in each batch should be different though given the same seed, the behavior of
# the program at each iter and epoch should be the same
def get_record_iter(data_record='./datasets/train_list.rec', shape=(3,30,100), label_width=4, batch_size=128):
    '''
    A wrapper of the ImageRecordIter, to assign the seed randomly
    '''
    seed = random.randint(0, 5000)
    return mx.io.ImageRecordIter(
        path_imgrec=data_record,
        data_shape=shape,
        label_width=label_width,
        shuffle=True,
        seed = seed,
        batch_size=batch_size
    )
