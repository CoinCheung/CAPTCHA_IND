#!/usr/bin/python
# -*- encoding: utf8 -*-


import mxnet as mx
import symbol.symbol as symbol
import core.config as config


def get_module():

    # parameters
    lr = config.learning_rate
    lr_factor = config.lr_factor
    wd = config.weight_decay
    batch_size = config.batch_size

    # get model symbol
    out = symbol.lenet5_symbol()

    ## get model module
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

    return mod

