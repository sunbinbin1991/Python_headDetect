#!/usr/bin/env python2.7
# coding: utf-8
# so I change the detail of this 
# what maybe find?

import cv2
import numpy as np
import matplotlib.pyplot as plt
'''
def CannyThreshold(lowThreshold):
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
    detected_edges = cv2.Canny(detected_edges, lowThreshold,
                               lowThreshold * ratio, apertureSize=kernel_size)
    dst = cv2.bitwise_and(img, img, mask=detected_edges)
    cv2.imshow('canny demo', dst)
    cv2.imwrite("CannyThreshold.jpg", dst)

lowThreshold = 0
max_lowThreshold = 255
ratio = 100
kernel_size = 3

firstImg = cv2.imread('cctv.jpg')
img = cv2.imread('cctv.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img2 = img.resize(50,50)

cv2.namedWindow('canny demo')

cv2.createTrackbar('Min threshold', 'Canny demo',
                   lowThreshold, max_lowThreshold, CannyThreshold)

plt.subplot(121)
plt.imshow(firstImg)

plt.subplot(122)
plt.imshow(img)


'''

img = cv2.imread("cctv.jpg", 0)
shape = img.shape
img = cv2.GaussianBlur(img, (3, 3), 0)
canny = cv2.Canny(img, 10, 250)

cv2.imshow('Canny', canny)
cv2.imwrite("cctv.jpg", canny)

image = cv2.imread('cctv.jpg')
height = image.shape[0]#图像的高
width = image.shape[1]#图像的宽
print height
print width

res = cv2.resize(image,(200,100), interpolation=cv2.INTER_LINEAR)
cv2.imshow('iker', res)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destoryAllWindows()
'''
CV_INTER_NN - 最近邻插值,

CV_INTER_LINEAR - 双线性插值 (缺省使用)

CV_INTER_AREA - 使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现。当图像放大时，类似于 CV_INTER_NN 方法..

CV_INTER_CUBIC - 立方插值.
'''