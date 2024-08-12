import os

from data import common
from data import srdata

import numpy as np

import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(Benchmark, self).__init__(
            args, name=name, train=train, benchmark=True
        )
    def _set_filesystem(self, dir_data):
        if self.name in ['Set5', 'Set14']:
            self.apath = os.path.join(dir_data, self.name, 'original')
            self.dir_hr = self.apath
            self.dir_lr = os.path.join(dir_data, self.name, 'LRbicx' + str(self.scale[0]))
            # print('self.dir_lr:', self.dir_lr)
            self.ext = ('.png', '.png')
        elif self.name in ['B100']:
            self.apath = os.path.join(dir_data, 'B100', 'HR')
            self.dir_hr = self.apath
            self.dir_lr = os.path.join(dir_data, 'B100', 'bicubic_x' + str(self.scale[0]))
            # self.dir_lr = os.path.join(dir_data, 'B100', 'x' + str(self.scale[0]))
            print('self.dir_lr:', self.dir_lr)
            self.ext = ('.png', '.png')
        elif self.name in ['Urban100']:
            # self.apath = os.path.join(dir_data, 'U100', 'HR')  # xian 4 3090
            self.apath = os.path.join(dir_data, 'Urban100', 'HR')  # taicang 4 3090
            self.dir_hr = self.apath
            # self.dir_lr = os.path.join(dir_data, 'U100', 'bicubic_x' + str(self.scale[0]))  # xian 4 3090
            self.dir_lr = os.path.join(dir_data, 'Urban100', 'bicubic_x' + str(self.scale[0]))  # taicang 4 3090
            # print('self.dir_lr:', self.dir_lr)
            self.ext = ('.png', '.png')
        else:
            print('Not benchmark!')
            return
'''
        #self.apath = os.path.join(dir_data, self.name)
        self.apath = os.path.join(dir_data, 'test')
        #self.dir_hr = os.path.join(self.apath, 'Images_HR')
        self.dir_hr = self.apath
        self.dir_lr = os.path.join(self.apath, 'bicubic_x4')
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.png', '.png')


    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        if self.input_large:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubicL')
        else:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('', '.png')
'''
