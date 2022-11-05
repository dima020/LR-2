import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('input/image_1.jpg')
# Инициировать детектор ORB
orb = cv.ORB_create()
# найти ключевые точки с помощью ORB
kp = orb.detect(img,None)
print(kp)
# вычисляем дескрипторы с помощью ORB
kp, des = orb.compute(img, kp)
#print(kp,des)
# draw only keypoints location,not size and orientation
img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

#plt.imshow(img2), plt.show()

cv.imshow('Result', img2)
cv.imwrite('image_result.jpg', img2)
cv.waitKey(0)