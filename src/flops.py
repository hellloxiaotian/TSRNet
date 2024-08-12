import torch

import utility
from option import args
from ptflops import get_model_complexity_info

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

# 建议连服务器在服务器上改代码，本地的代码不是最新版的
def flops():
    global model

    # loader = data.Data(args)
    print(args)
    _model = model.Model(args, checkpoint)
    print(_model)
    # model1 = model.DDYCNNSR(args, checkpoint)
    # _loss = loss.Loss(args, checkpoint) if not args.test_only else None

    # 计算模型总参数量count total number of parameters
    total_num = sum(p.numel() for p in _model.parameters())  #1818520
    trainable_num = sum(p.numel() for p in _model.parameters() if p.requires_grad)  #1818496
    print('Total_parameters: ', total_num)
    print('trainable_parameter_num: ', trainable_num)

    flops, params = get_model_complexity_info(_model.model, (3, 256, 256), as_strings=True,
    # flops, params = get_model_complexity_info(model1, (3, 256, 256), as_strings=True,
                                              print_per_layer_stat=True)  # 不用写batch_size大小，默认batch_size=1
    print('flops:', flops)
    print('params:', params)
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')



if __name__ == '__main__':
    flops()
