import cv2 as cv
import numpy as np
from keras.models import load_model
import solver

# Load the pre-trained MNIST digit recognition model
model = load_model('drecv2')  # Replace 'mnist_model.h5' with the path to your model file

def preprocess_image(image):
    # Preprocess the input image for the model
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    resized = cv.resize(gray, (28, 28), interpolation=cv.INTER_AREA)
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
    
    return int(digit)

def extractBoxes(img): #Gives out list, 0: sudoku, 1: A11 2: A12...
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    img_blur = cv.bilateralFilter(src=img_gray, d=9, sigmaColor=75, sigmaSpace=75)
    #cv.imshow('puzzle', img)
    #img2 will be blurred for edge detection

    _, dst_blur = cv.threshold(img_blur, 230, 255, cv.THRESH_BINARY)

    contours, hierarchy = cv.findContours(image=dst_blur, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
    box_data = []

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        roi = img[y: y+h, x:x+w]
        check = (h>20)&(w>20)
        if check:
            roi = cv.resize(roi, (28, 28), interpolation=cv.INTER_LINEAR)
            roi = cv.bitwise_not(roi)
            #roi = roi.astype(np.float32) / np.float32(255)
            #roi = roi.reshape((1, 28, 28, 1))
            box_data.append(roi)
            #print(roi.shape)
            #print(x, y, w, h)

    box_data.pop(0)

    return box_data


def ConvertToGrid(box_data):
    #remove 0th entry which is the full puzzle
    box_data.pop(0)

    puzzle =[[0 for x in range(9)]for y in range(9)]

    for rowindex in range(9): # [0..8]
        for columnindex in range(9):
            index = rowindex*9 + columnindex
            puzzle[rowindex][columnindex] = predict_digit(box_data[index])

    return puzzle

frame = cv.imread('puzzle_images/sample2.jpg')

boxes = extractBoxes(frame)
puzzle = ConvertToGrid(boxes)

print(boxes.__len__())

solver.print_grid(puzzle)
solver.solve_sudoku(puzzle)

