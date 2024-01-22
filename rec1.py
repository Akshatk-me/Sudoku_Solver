import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained MNIST digit recognition model
model = load_model('drecv2')  # Replace 'mnist_model.h5' with the path to your model file

def preprocess_image(image):
    # Preprocess the input image for the model
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    reshaped = normalized.reshape((1, 28, 28, 1))
    return reshaped

def predict_digit(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make a prediction
    predictions = model.predict(processed_image)
    #print(predictions)
    
    # Get the predicted digit
    digit = np.argmax(predictions)
    
    return (predictions, digit)

# Capture video from the default camera (you can change the parameter to use a different camera)

# Read a frame from the camera
#frame = cv2.imread('six.jpg')

# Flip the frame horizontally for a later selfie-view display
#frame = cv2.flip(frame, 1)

#frame = cv2.imread("example.jpg")
# Display the frame
#cv2.imshow("Digit Recognition", frame)

#predicted_digit = predict_digit(frame)
#print(f"Predicted Digit: {predicted_digit}")
#
# Release the camera and close all OpenCV windows



#while True:
#    key = cv2.waitKey(1) & 0xFF 
#    if key == ord('q'):
#        break
#

