#!/usr/bin/python
# -*- encoding: utf8 -*-


import numpy as np



## accuracy
def accuracy(scores, label_real):
    label_pred = np.argmax(scores, axis=2)
    same = np.sum(label_pred == label_real, axis=1)

    acc0 = np.sum(same==0)/same.size
    acc1 = np.sum(same==1)/same.size
    acc2 = np.sum(same==2)/same.size
    acc3 = np.sum(same==3)/same.size
    acc4 = np.sum(same==4)/same.size
    return [acc0, acc1, acc2, acc3, acc4]


