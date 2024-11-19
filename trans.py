import os
import re
import glob
import numpy as np
from PIL import Image
import scipy.io as io

src_dir = r'E:\pidinet-master\wget\NYUD\test\contour\test\\'
save_dir = r'E:\Eevee\newnet\model\model-2\test\mat\gt/'

all_file = os.walk(src_dir)
fileNum = 0

for root, dirs, files in all_file:
    # print(root, end=',')
    for file in files:
        fileNum = fileNum + 1
        print(src_dir + file)
        file_ext = os.path.splitext(file)
        front, ext = file_ext
        # print(front)  #获取图片名字前缀
        # print(ext)  # 获取图片后缀

        img = Image.open(src_dir + file)
        # 保存为.npy
        res = np.array(img, dtype='uint16')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        # np.save(save_dir + front + '.npy', res)
        # 保存为.mat
        # numpy_file = np.load(save_dir + front + '.npy')
        # io.savemat(save_dir+front+'.mat', {'data': numpy_file})
        io.savemat(save_dir + front + '.mat', {'groundTruth': [{'Boundaries': res}]})

# 删除中间文件mat
delete_command = 'rm -rf ' + save_dir + '*.npy'
print(delete_command)
os.system(delete_command)
print('共转化了' + str(fileNum) + '张')
