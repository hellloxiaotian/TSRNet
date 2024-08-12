import os
import shutil


def alter_fileName(path):  # 获取目录下所有文件的文件名
    n = 0
    filelist = os.listdir(path)
    for i in filelist:  # i是path目录下的文件名
        # oldname = targe_path + os.sep + filelist[n]  # oldname是原文件路径
        oldname = path + os.sep + i  # oldname是原文件路径
        newname = os.listdir(path)[n].replace(".png", "_LR.png")  # newname是新文件名，路径+新文件名
        os.rename(oldname, newname)  # 改名字
        print(oldname, '======>', newname)
        n += 1


def file_copy(path, targe_path):  # 将path目录下所有jpg文件复制到targe_path
    '''
    root 所指的是当前正在遍历的这个文件夹的本身的地址
    dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
    files 同样是 list , 内容是该文件夹中所有的文件名(不包括子目录)
    '''
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith('.png'):  # 若文件名结尾是以png结尾，则复制到新文件夹
                list = (os.path.join(root, name))  # list是文件的全路径
                shutil.copy(list, targe_path)  # 将文件复制到新文件夹


def copyandalter_HR(path):  # 获取目录下所有文件的文件名
    file_copy(path, path_HR)
    n = 0
    filelist = os.listdir(path_HR)
    for i in filelist:  # i是path目录下的文件名
        # oldname = targe_path + os.sep + filelist[n]  # oldname是原文件路径
        oldname = path + os.sep + i  # oldname是原文件路径
        newname = os.listdir(path_HR)[n].replace(".png", "_HR.png")  # newname是新文件名，路径+新文件名
        os.rename(oldname, newname)  # 改名字
        print(oldname, '======>', newname)
        n += 1


path = r'/data/zhangxuanyu/datasets/bottle/Images/'
path_HR = '/data/zhangxuanyu/datasets/bottle/Images_HR/'
targe_path2 = '/data/zhangxuanyu/datasets/bottle/Images_bicubic_x2/'
targe_path3 = '/data/zhangxuanyu/datasets/bottle/Images_bicubic_x3/'
targe_path4 = '/data/zhangxuanyu/datasets/bottle/Images_bicubic_x4/'
copyandalter_HR(path)
alter_fileName(targe_path4)
file_copy(path_HR, targe_path4)

'''
path1 = r'/data/zhangxuanyu/datasets/bottle/Images/'
path2 = '/data/zhangxuanyu/datasets/bottle/Images_bicubic_x4/'

i = 0
j = 0
for file in os.listdir(path1):
    # newname = name.split('.')[0] + '.mkv'  # 切割出正确标题部分添加到列表
    newname = os.listdir(path1)[i].replace(".png", "_HR.png")
    i += 1
    os.rename(os.path.join(path1, file), os.path.join(path1, newname))

for file in os.listdir(path2):
    if os.path.isfile(os.path.join(path2, file)) == True:
        # newname = listnew[n]  # 获取新文件名
        newname = os.listdir(path1)[j].replace("_HR.png", "_LR.png")
        j += 1  # 下标加一
        os.rename(os.path.join(path2, file), os.path.join(path2, newname))  # 重命名完成
'''
