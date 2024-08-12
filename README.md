# TSRNet
## A Tree-guided CNN for image super-resolution
## This paper is conducted by Chunwei Tian, Mingjian Song. 


## Absract
### Deep convolutional neural networks can extract more accurate structural information via deep architectures to obtain good performance in image super-resolution. However, a single network architecture is not easy to find effects of key layers to decrease denoising effects. In this paper, we design a tree-guided CNN for image super-resolution (TSRNet). It uses a tree architecture to guide a deep network to enhance effects of key nodes to enhance relation of hierarchical information for improving ability of recovering images. To prevent insufficiency of obtained structural information, cosine transform techniques in the TSRNet used to extract cross-domain information further enhances performance of image super-resolution.  Adaptive Nesterov momentum algorithm (Adan) optimizer is used to optimize parameters to improve effectiveness of training a super-resolution model. Extended experiments can verify superiority of the proposed TSRNet for restoring high-quality images.

## Requirements (Pytorch)  
#### Pytorch 1.13.1

#### Python 3.8

#### torchvision

#### openCv for Python


## Datasets
### Training dataset

#### The training dataset is downloaded at https://data.vision.ee.ethz.ch/cvl/DIV2K/

### Test datasets

#### The test dataset of Set5 is downloaded at 链接：https://pan.baidu.com/s/1YqoDHEb-03f-AhPIpEHDPQ (secret code：atwu) (baiduyun) or https://drive.google.com/file/d/1hlwSX0KSbj-V841eESlttoe9Ew7r-Iih/view?usp=sharing (google drive)

#### The test dataset of Set14 is downloaded at 链接：https://pan.baidu.com/s/1GnGD9elL0pxakS6XJmj4tA (secret code：vsks) (baiduyun) or https://drive.google.com/file/d/1us_0sLBFxFZe92wzIN-r79QZ9LINrxPf/view?usp=sharing (google drive)

#### The test dataset of B100 is downloaded at 链接：https://pan.baidu.com/s/1GV99jmj2wrEEAQFHSi8jWw （secret code：fhs2) (baiduyun) or https://drive.google.com/file/d/1G8FCPxPEVzaBcZ6B-w-7Mk8re2WwUZKl/view?usp=sharing (google drive)

#### The test dataset of Urban100 is downloaded at 链接：https://pan.baidu.com/s/15k55SkO6H6A7zHofgHk9fw (secret code：2hny) (baiduyun) or https://drive.google.com/file/d/1yArL2Wh79Hy2i7_YZ8y5mcdAkFTK5HOU/view?usp=sharing (google drive)

## Commands
### Training a model for single scale

#### x2
#### python main.py --model tsrnet --scale 2  --data_test Urban100 --patch_size 128 --save tsrnet --epochs 1200 --batch_size 64 --data_range 1-900 --gclip 10.0
#### x3
#### python main.py --model tsrnet --scale 3  --data_test Urban100 --patch_size 128 --save tsrnet --epochs 1200 --batch_size 64 --data_range 1-900 --gclip 10.0
#### x4
#### python main.py --model tsrnet --scale 4  --data_test Urban100 --patch_size 128 --save tsrnet --epochs 1200 --batch_size 64 --data_range 1-900 --gclip 10.0

### Test with your own parameter setting in the option.py.
#### x2
#### python main.py --model tsrnet --scale 2  --data_test Set14  --pre_train ../experiment/tsrnet_x2/model/model_best.pt --test_only --save_results
#### x3
#### python main.py --model tsrnet --scale 3  --data_test Set14  --pre_train ../experiment/tsrnet_x3/model/model_best.pt --test_only --save_results
#### x4
#### python main.py --model tsrnet --scale 4  --data_test Set14  --pre_train ../experiment/tsrnet_x4/model/model_best.pt --test_only --save_results


## 1. Network architecture of TSRNet



## 2. HDSRNet for x2，x3 and x4 on Set5



## 3. HDSRNet for x2，x3 and x4 on Set14



## 4. HDSRNet for x2，x3 and x4  on B100



## 5. HDSRNet for x2，x3 and x4  on U100



## 6. Running time of different methods on hr images of size 256x256 and 512x512 for x4.



## 7. Complexities of different methods for x4.



## 8. Visual results of Set14 for x2.



## 9. Visual results of B100 for x3.



## 10. Visual results of B100 for x3.



## 11. Visual results of U100 for x4.



## 12. Visual results of U100 for x4.



## 13. Visual results of U100 for x4.



## If you cite this paper, please use the following format:




