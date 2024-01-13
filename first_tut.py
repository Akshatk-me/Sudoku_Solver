import cv2 as cv
import numpy as np

#opening an image

img = cv.imread('MemeArchive/yo_dawg.jpg')
(h, w, d) = img.shape

print(h,w,d)

#cv.imshow("Image", img)
#cv.waitKey(0) #waitkey waits for a keypress and '0' specifies indefinite loop

# to read image as grayscale
#img = cv.imread('MemeArchive/yo_dawg.jpg', 0)

#cv.destroyAllWindows()

#cv.imwrite('grayscale.jpg', img) 
#opencv uses BRG format to read images, compared to others, that use RGB

#image filtering using convolution

#Convolution kernel is a 2D matrix that is used to filter images.
#Convolution matrix is typically a square odd order matrix.
#These kernels perform mathematical operations on each pixel of an image

#Blurring reduces certain types of noises in image. Hence called smoothing.

#Blurring also used to remove distracting background (example potrait mode)

#Convolution of an image is a mathematical operations 
# Center of kernel is postioned over a specific pixel.
# Multiply value of each element in the kernel with the corresponding pixel element (pixel intensity).
# Add the result of multiplications and compute the average.
# Replace value of pixel, with the average value just computed.

# filter2D() function performs the linear filtering operation.

kernel1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) #identity kernel

identity = cv.filter2D(src=img, ddepth=-1, kernel=kernel1)
#ddepth indicates teh depth of the resulting image. -1 indicates final image will also have the same depth as teh source.
kernel2 = np.ones((10, 10), np.float32)/100
# we had to divide by 25 (number of elements) so all the values stay within (0,1).
# Like when calculating average adding all elements must < 1 

blur = cv.filter2D(src=img, ddepth=-1, kernel=kernel2)

cv.imshow("Original", img)
cv.imshow("identity", identity)
cv.imshow("blur", blur)


while True:
    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cv.destroyAllWindows()

