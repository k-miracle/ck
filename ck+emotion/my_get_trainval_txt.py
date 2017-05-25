# -*- coding: utf-8 -*-
from skimage import io
import os

path="/home/gp/zk/ck+/haarck/"
f=open("/home/gp/zk/ck+/haarck/fileharrck.txt","r")

train="/home/gp/zk/ck+/mytest/train"
test="/home/gp/zk/ck+/mytest/test"

countall=0

lines=f.readlines()
for line in lines:
	if line:
		line=line.strip('\n')	#去除读取一行时末尾的换行符
		content=line.split(' ')
		tmp=content[0]
		#scrimg=io.imread(path+tmp[35:58])
		imgpath=path+tmp[35:58]
		if countall % 10 == 0:
			os.system("cp " + imgpath + " " + train +"/"+content[1]+tmp[37:58]) #重命名图片：在每张图片名字前加上他的标签号
		else:
			os.system("cp " + imgpath + " " + test +"/"+content[1]+tmp[37:58])
		countall +=1
		print "this is " + str(countall) + "pic" 

print "all pic copy already"

