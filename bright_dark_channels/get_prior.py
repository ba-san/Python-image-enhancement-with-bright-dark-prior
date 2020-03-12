import cv2
import numpy as np

src = cv2.imread('../images/desk.png')

print(src.shape)

blank = np.zeros((src.shape[0],src.shape[1],3), np.uint8)

for i in range(3): # channel
	for x in range(src.shape[1]):
		for y in range(src.shape[0]):
			

cv2.imshow('test', blank)
cv2.waitKey(100)
cv2.imshow('test', src)
cv2.waitKey(0)
