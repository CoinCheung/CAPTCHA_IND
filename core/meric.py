#!/usr/bin/python
# -*- encoding: utf8 -*-


import numpy as np


def accuracy(label, score):
    '''
    This method accepts a true label and its associating predicted score to
    computer the prediction accuracy
    params:
        label: the true labels of the sample
        score: the predicted scores that is used to compute the predicted label
    return:
        the accuracy of the predicted label computed with its true label counterparts
        the list [acc0, acc1, acc2, acc3, acc4] are the accuracy of correctly
        predicting 0-4 letters or digits in the picture.
    '''
    pred = np.argmax(score, axis=2)

    same = np.sum(pred==label, axis=1)
    acc0 = np.sum(same==0)/same.size
    acc1 = np.sum(same==1)/same.size
    acc2 = np.sum(same==2)/same.size
    acc3 = np.sum(same==3)/same.size
    acc4 = np.sum(same==4)/same.size
    return [acc0, acc1, acc2, acc3, acc4]

