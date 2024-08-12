#运行命令：
# python main.py --model tsrnet --scale 4  --data_test Urban100 --patch_size 128 --save tsrnet --epochs 1200 --batch_size 64 --data_range 1-900 --gclip 10.0
#测试命令：
# python main.py --model tsrnet --scale 4  --data_test Set14  --pre_train ../experiment/tsrnet_x4/model/model_best.pt --test_only --save_results
# python main.py --model tsrnet --scale 3  --data_test Set14  --pre_train ../experiment/tsrnet_x3/model/model_best.pt --test_only --save_results
# python main.py --model tsrnet --scale 2  --data_test Set14  --pre_train ../experiment/tsrnet_x2/model/model_best.pt --test_only --save_results
