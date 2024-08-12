import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            # print('--------------------------')
            # print(args)
            # print('--------------------------')
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None

            # 计算模型总参数量count total number of parameters
            total_num = sum(p.numel() for p in _model.parameters())  #1818520
            trainable_num = sum(p.numel() for p in _model.parameters() if p.requires_grad)  #1818496
            print('Total_parameters: ', total_num)
            print('trainable_parameter_num: ', trainable_num)

            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            # flops, params = get_model_complexity_info(_model, (3, 256, 256), as_strings=True,
            #                                           print_per_layer_stat=True)  # 不用写batch_size大小，默认batch_size=1
            # print('Flops:  ' + flops)
            # print('Params: ' + params)
            checkpoint.done()

if __name__ == '__main__':
    main()
