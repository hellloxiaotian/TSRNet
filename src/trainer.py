import os
import math
from decimal import Decimal

import utility

import torch
from torch import tensor
import torch.nn.utils as utils
from tqdm import tqdm
import numpy as np
import cv2

from thop import profile
import time
from ptflops import get_model_complexity_info

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        # print('self.scale:', self.scale)

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()  # learning rate

        self.ckp.write_log(
            # '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
            '[Epoch {}]\tLearning rate: {}'.format(epoch, lr)
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        # self.loader_train.dataset.set_scale(0)
        self.loader_train.dataset.set_scale(self.scale)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            # print('--lr', lr.shape)
            # print('trainer-lr1', lr)
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            # print('--sr', sr.shape[2])  # 126
            # print('--sr', sr.shape)  # torch.Size([64, 3, 126, 126])
            # if sr.shape!=hr.shape:
            #     hr = hr[:, :, :sr.shape[2], :sr.shape[3]]
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

        learning_rate = self.decay_learning_rate(epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate

        # if self.args.model.lower().startswith('ddy'):
        #     self.model.model.update_temperature()

        self.optimizer.schedule()

    def decay_learning_rate(self, epoch):
        lr = self.args.lr * (0.5 ** (epoch // 300))
        return lr

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()

        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    # print('lr.shape', lr.shape)
                    # starttime = time.time()
                    sr = self.model(lr, idx_scale)
                    # endtime = time.time()
                    # dtime = endtime - starttime
                    # print('RunTime: ', dtime)

                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    # self.ckp.log[-1, idx_data, idx_scale] += utility.calc_ssim(
                    #     sr, hr, scale, self.args.rgb_range, dataset=d
                    # )
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )

                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.5f} (Best: {:.5f} @epoch {})'.format(
                        # '[{} x{}]\tSSIM: {:.5f} (Best: {:.5f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1,
                    )
                )
        # test = torch.rand(1, 3, 1024, 1024).cuda(3)  # 0.0141658;0.0261087;0.5226306
        # print('1024')
        # test = torch.zeros(1, 3, 1024, 1024).cuda(0)  # 0.021426;0.028827;0.4102299
        # print('512')
        # test = torch.zeros(1, 3, 512, 512).cuda(3)  # 0.021426;0.028827;0.4102299
        # print('256')
        # test = torch.zeros(1, 3, 256, 256).cuda(3)  # 0.021426;0.028827;0.4102299
        # starttime = time.time()
        # sr = self.model(test, 4)
        # endtime = time.time()
        # dtime = endtime - starttime
        # print('RunTime: ', dtime)
        # 计算模型总参数量count total number of parameters
        # flops, params = get_model_complexity_info(self.model.DDYCNNSR((1, 3, 256, 256), 4), (3, 256, 256), as_strings=True,
        #                                           print_per_layer_stat=True)  # 不用写batch_size大小，默认batch_size=1
        # print('Flops:  ' + flops)
        # print('Params: ' + params)
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # print("src-trainer-prepare-cuda:0")
        device = torch.device('cpu' if self.args.cpu else 'cuda:0')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch()
            return epoch >= self.args.epochs
