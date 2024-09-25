import os
import cv2

from PIL import Image
# for i in range(1,201):
#   if i==169 or i==189:
#     i = i+1
# pth = "C:\\Users\\Desktop\\asd\\"+str(i)+".bmp"
# pth = "ISTFormer/output/cut/016/coal_016_test_test.png"
pth = "ISTFormer/output/cutted/016/coal_016_test_test.png"

image = cv2.imread(pth)  
# print(image)    
# cropImg = image[600:1000,2800:3400]
cropImg = image[50:170,400:580] 
# cv2.imwrite("C:\\Users\\Desktop\\qwe\\"+str(i)+".bmp",cropImg)
r = 2000.0/cropImg.shape[1]
dim = (2000, int(cropImg.shape[0]*r))

# 执行图片缩放，并显示
resized = cv2.resize(cropImg, dim, interpolation=cv2.INTER_AREA)

# cropImg = resized[5000:30000,0:12000] 
# cv2.imwrite("ISTFormer/output/cutted/016/coal_016_test_test.png", cropImg)

cv2.imwrite("ISTFormer/output/cutted2/016/coal_016_test_test.png", cropImg)