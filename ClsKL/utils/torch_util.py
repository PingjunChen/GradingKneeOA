# -*- coding: utf-8 -*-

import os, sys, pdb

class LRScheduler():
    def __init__(self, init_lr=1.0e-4, lr_decay_epoch=10):
        self.init_lr = init_lr
        self.lr_decay_epoch = lr_decay_epoch

    def __call__(self, optimizer, epoch):
        '''Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.'''
        lr = self.init_lr * (0.8 ** (epoch // self.lr_decay_epoch))
        lr = max(lr, 1e-8)
        if epoch % self.lr_decay_epoch == 0:
            print ('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        return optimizer
