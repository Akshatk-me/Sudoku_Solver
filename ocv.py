import cv2 as cv
import numpy as np

#opening an image

img = cv.imread('MemeArchive/yo_dawg.jpg')
img2 = cv.imread('MemeArchive/numbers.jpg', cv.IMREAD_GRAYSCALE)
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


cv.destroyAllWindows()

#opencv default fucntion to blur
img_blur = cv.blur(src=img, ksize=(5,5))

# Gaussian Blurring 
# Basically has weighted average instead of uniform average.

gaussian_blur = cv.GaussianBlur(img, ksize=(5,5), sigmaX=1, sigmaY=9)

#Sharpening imagesg 

kernel3 = np.array([[0, -1,  0],
                   [-1,  5, -1],
                    [0, -1,  0]])

sharp_img = cv.filter2D(img, ddepth=-1, kernel=kernel3)



# sigmaX is the standard devication in x (horizontal) direction


#There is also median blur function. 
#Bilateral filtering 

# Only blurs similar intensity picture in a neighborhood. Sharp edges are preserved, wherever possible.
# This is Basically 2D Gaussian(weighted) blur where weights are controlled by color intensity

noise_red = cv.bilateralFilter(src=img, d=9, sigmaColor=75, sigmaSpace=75)

# sigmaColor defines 1D Gaussian distruibution which specifies degree to which differences in pixel intensity can be tolerated.
# sigmaSpace defines the spatial extent of the kernel (just like GaussianBlur)

#Thresholding in opencv
# All the pixels having intensity (grayscale value) below arbitrary set value will be reduced further in their intensity. 

#Global Thresholding.
# takes source image(src) and threshold value. produces an output image(dst).
# if src(x, y) > thresh then dst(x,y) is assigned some value. 


# Binary Thresholding. dst(x,y) is either maxvalue or 0

th, dst = cv.threshold(img2, 0, 85, cv.THRESH_BINARY)
print(cv.THRESH_BINARY)
print(th)


#edge detection
# sudden changes in pixel intesity characterize edges. 
# Sobel edge detection and canny edge detection 

#convert to gray scale
# img_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

img_blur = cv.GaussianBlur(img2,(5,5), 0)

#sobel edge detection 
sobelx = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5)
#sobel detection on x axis


#canny edge detection
edges = cv.Canny(image=img_blur, threshold1=50, threshold2=100)

#contour detection opencv
# contour refers to boundary pixels that have same color and intensity.

#convert to grayscale, apply Binary Thresholding

ret, thresh = cv.threshold(img_blur, 230, 255, cv.THRESH_BINARY)
#threshold range min (0 corresponds to darkest) 

contours, hierarchy = cv.findContours(image=thresh, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

img2_copy = img2.copy()
cv.drawContours(image=img2_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=3, lineType=cv.LINE_AA)

#cv.imshow("canny", edges)
cv.imshow("contours", img2_copy)
cv.imshow("Threshed", thresh)
cv.imshow("img_blur", img_blur)
#cv.imshow("sobel", sobelx)
#cv.imshow("identity", identity)
#cv.imshow("blur", blur)
#cv.imshow('gaussian_blur', gaussian_blur)
#cv.imshow('sharp_img',sharp_img) 
#cv.imshow('bilateralFilter',noise_red) 
#cv.imshow('threshold',dst) 
#cv.imshow("Keypoints", im_with_keypoints)

while True:
    key = cv.waitKey(1) & 0xFF #detects q keypress
    # 1 indicates delay of 1 millisecond 
    # & 0xFF reutrns the last 8 bits (Ascii value) of the keycode 
    if key == ord('q'):
        break















