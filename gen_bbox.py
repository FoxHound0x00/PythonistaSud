## Python Script to generate yolo format bboxes for Market1501

import cv2
import os
str = "0 0.5 0.5 1 1"
for file in os.listdir():
        if file.endswith(".jpg"):
                f1 = file.rsplit('.',1)[0] + '.txt'
                print(f1)
                print(str)
                f2 = open(f1,'w')
                f2.write(str)




## testing stuff
#import cv2
#img = cv2.imread('dataset/0227_c6s1_047701_00.jpg')
#print(img.shape)
#w,h,c = img.shape
#print(w)
#print(h)
#nxmin = (0+(w/2))/w
#nymin = (0+(h/2))/h
#nw = 1
#nh = 1
#class_num = 0 # provide the class number of person
#str = f'{class_num} {nxmin} {nymin} {nw} {nh}'
#print(str_final)
