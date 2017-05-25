#-*- coding: utf-8 -*-
import glob
train="/home/gp/zk/ck+/mytest/train"
test="/home/gp/zk/ck+/mytest/test"

trainimg_ls=[img for img in glob.glob(train+"/*.png")]
testimg_ls=[img for img in glob.glob(test+"/*.png")]

print len(trainimg_ls)
print len(testimg_ls)

print trainimg_ls[0]
print testimg_ls[0]

f=open("./trainlabel.txt","w")
for img in trainimg_ls:
	tmp=img.split("/")
#	print tmp[7]
#	print tmp[7][0]
	f.write(img+" "+tmp[7][0])
	f.write('\n')
	print img+" "+tmp[7][0]
f.close()

f=open("./testlabel.txt","w")
for img in testimg_ls:
        tmp=img.split("/")
#       print tmp[7]
#       print tmp[7][0]
        f.write(img+" "+tmp[7][0])
        f.write('\n')
        print img+" "+tmp[7][0]
f.close()

