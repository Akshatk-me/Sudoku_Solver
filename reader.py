
import cv2 as cv 
import numpy as np
import recognize

def sqrt(a, b):
    return np.sqrt(a*a + b*b)

img = cv.imread('puzzle_images/puzzle.jpg', cv.IMREAD_GRAYSCALE)

img_blur = cv.bilateralFilter(src=img, d=9, sigmaColor=75, sigmaSpace=75)
#cv.imshow('puzzle', img)
#img2 will be blurred for edge detection

#blur = cv.GaussianBlur()


th, dst = cv.threshold(img, 230, 255, cv.THRESH_BINARY)
th_blur, dst_blur = cv.threshold(img_blur, 230, 255, cv.THRESH_BINARY)
edges = cv.Canny(image=dst_blur, threshold1=50, threshold2=100)
#edges = cv.bitwise_not(edges)
contours, hierarchy = cv.findContours(image=dst_blur, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

ls = []




for contour in contours:
    x, y, w, h = cv.boundingRect(contour)
    roi = dst[y: y+h, x:x+w]
    check = (h>20)&(w>20) 
    if check: 
        roi = cv.resize(roi, (28, 28), interpolation=cv.INTER_LINEAR)
        ls.append(roi)
        print(roi.shape)
        #print(x, y, w, h)
    
ls.pop(0)

print(len(ls))


#recognize.predictor(ls[2])

#at this point ls[1:] contains all the required images to detect number
#import a function to convert reading images to int.

cv.imshow('contour', ls[2])
cv.imshow('puzzle', dst)
cv.imshow('dst_blur', dst_blur)
#cv.imshow('edges', edges)


while True:
    key = cv.waitKey(1) & 0xFF 
    if key == ord('q'):
        break






