# -*- coding: utf-8 -*-
import os
import glob
import random
import numpy as np

import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb

#Size of images
IMAGE_WIDTH = 224	#chage old file 227 to 224
IMAGE_HEIGHT = 224

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    # 直方图均衡，提高图像对比度
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    #interpolation-插值方法，inter_cubic三次插值，用于放大图像

    return img

#获取图像及其标签，并返回包含图像及其标签的Datum对象
def make_datum(img, label):

    # 图像是一个np格式，RGB，要转换成caffe的BGR格式
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())    #把轴2放到轴0的位置，即宽/高/颜色通道->颜色通道/宽/高
        #再把数组形式转化为string形式
train_lmdb = '/home/gp/zk/ck+/mytest/input/train_lmdb'
validation_lmdb = '/home/gp/zk/ck+/mytest/input/validation_lmdb'

# #如果存在了这个文件夹，先删除
# os.system('rm -rf  ' + train_lmdb)  #相当于在命令行中输入的命令,rm -rf train_lmdb，移除train_lmdb这个文件
# os.system('rm -rf  ' + validation_lmdb)

#读取图像
train_data = [img for img in glob.glob("/home/gp/zk/ck+/mytest/train/*png")]  #图片放在"/home/gp/zk/ck+/mytest/train"下
test_data = [img for img in glob.glob("/home/gp/zk/ck+/mytest/test/*png")]
#glob.glob("../input/test1/*jpg") ：获取上级目录的所有.jpg文件的路径名

#打乱数据顺序
random.shuffle(train_data)

print 'Creating train_lmdb'

#打开lmdb环境
#创建数据集
in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:         #因为是lmdb格式的文件，所以要用他的方式打开，
    for in_idx, img_path in enumerate(train_data):
        if in_idx %  6 == 0:       #只处理 5/6 的数据作为训练集
            continue               #留下 1/6 的数据用作验证集
        # 读取图像，做直方图均衡化、裁剪操作
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        if img_path[29] == "0":
            label = 0
        elif img_path[29] == "1":
            label = 1
        elif img_path[29] == "2":
            label = 2
        elif img_path[29] == "3":
            label = 3
        elif img_path[29] == "4":
            label = 4
        elif img_path[29] == "5":
            label = 5
        elif img_path[29] == "6":
            label = 6
        datum = make_datum(img, label)  #caffe定义的数据类型
        #序列化datum对象到in_txn文件中
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString()) #'{:0>5d}'一种格式
        print '{:0>5d}'.format(in_idx) + ':' + img_path
#结束后必须要释放资源，否则下次用打不开
in_db.close()


print '\nCreating validation_lmdb'

#创建验证集
in_db = lmdb.open(validation_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
        if in_idx % 6 != 0:
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        if img_path[0] == "0":
            label = 0
        elif img_path[0] == "1":
            label = 1
        elif img_path[0] == "2":
            label = 2
        elif img_path[0] == "3":
            label = 3
        elif img_path[0] == "4":
            label = 4
        elif img_path[0] == "5":
            label = 5
        elif img_path[0] == "6":
            label = 6
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()

print '\nFinished processing all images'

#/home/gp/zk/cnn/deeplearning-cats-dogs-tutorial/code
