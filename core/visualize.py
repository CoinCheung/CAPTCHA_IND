#!/usr/bin/python
# -*- encoding: utf8 -*-


import numpy as np
import matplotlib.pyplot as plt



## draw
def draw_loss(train_loss, fig_num):
    '''
    Draw the loss curve with a loss array in form of numpy array
    params:
        train_loss: a numpy array containing train loss in the training process
        fig_num: the number of the figure
    '''
    f, ax = plt.subplots(fig_num)
    plt.ion()
    ax.plot(np.array(train_loss))
    ax.set_title('train_loss')
    plt.show()



def draw_acc(acc, fig_num):
    '''
    Draw the curve with a list of 4 numpy array data
    params:
        acc: accuracy list. Should be of shape (valid times, 5) which indicates
             the accuracy of successful prediction of 0-4 characters
        fig_num: number of the figure
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

