import numpy as np 
import keras

model = keras.models.load_model('drecv2')

#img = cv.imread('puzzle_images/sample2.jpg')[:,:,0]


def predictor(img):
    print("shape of image is", img.shape)
    prediction = model.predict(img)
    print("This digit could be a", np.argmax(prediction))
    print("Other things could be:", prediction)



#cv.imshow("image", img)
#
#while True:
#    key = cv.waitKey(1) & 0xFF 
#    if key == ord('q'):
#        break
#
