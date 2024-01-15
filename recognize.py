from cv2.typing import MatLike
import tensorflow as tf 
import numpy as np 
import keras
import os
import cv2 as cv 

model = keras.models.load_model('digit_identifier.model')

#img = cv.imread('puzzle_images/sample2.jpg')[:,:,0]


def predictor(img: MatLike):
    prediction = model.predict(img)
    print("This digit could be a", np.argmax(prediction))



#cv.imshow("image", img)
#
#while True:
#    key = cv.waitKey(1) & 0xFF 
#    if key == ord('q'):
#        break
#
