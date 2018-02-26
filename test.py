#!/usr/bin/python
# -*- encoding: utf8 -*-


import mxnet as mx
import numpy as np
import core.module as module
import core.io as io
import tools.vec2str as vec2str



def test(small_batch_size=10):
    '''
    test the trained model. Certain amount of pictures will be chosen from the
    test dataset and fed into the trained model. Then the predicted captcha label
    and the true captcha label will be printed.
    Note: this script should only run after the train.py is executed. The variable epoch in the file
    config.py should remain unchanged when the two scripts are executed.

    params:
        small_batch_size: number of samples to be chosen from the test dataset to valid the model
    '''

    # data iterator
    test_iter = io.get_record_iter('./datasets/test_list.rec',(3,30,100),4,small_batch_size)

    # get module
    mod = module.get_test_module()

    # predict
    valid = mod.predict(test_iter, 1, always_output_list=True)

    scores = valid[1].asnumpy()
    ind = np.argmax(scores, axis=2)
    label_code = valid[2].asnumpy().astype(np.int32)

    # switch to captcha
    pred = list(map(vec2str.vec2str,ind))
    label = list(map(vec2str.vec2str,label_code))

    for i in range(small_batch_size):
        print('correct label: {}, \t predicted label: {}'.format(pred[i], label[i]))



if __name__ == '__main__':
    small_batch_size = 10
    test(small_batch_size)
