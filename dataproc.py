import tensorflow as tf 
import keras


mnist = keras.datasets.mnist

# x: pixel data, y: classification
(x_train, y_train), (x_test, y_test) = mnist.load_data() 

# scale the grayscale values 0-255 to 0-1
x_train = keras.utils.normalize(x_train, axis = 1)
x_test = keras.utils.normalize(x_test, axis = 1)

# done preprocessing, start with model

model = keras.models.Sequential()

# now we add layers to the model
# convert 28*28 to one single line of 784
model.add(keras.layers.Flatten(input_shape=(28, 28))) 

# Just regular densely connected NN layers.
# It performs various matrix-vector multiplications. 
# Outputs 'm' dimensional vector. Changes dimensions and also
# operations like rotation, scaling, translation are done on vector.

#relu is rectified linear unit. An activation function
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
#softmax makes it so that all neuron outputs add upto 1.
#Basically gives out the probabilities of object being of certain class.
model.add(keras.layers.Dense(10, activation='softmax'))

#compiling the model. 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

model.save('digit_identifier.model')


#model = keras.models.load_model('digit_identifier.model')
#
#loss, accuracy = model.evaluate(x_test, y_test)
#
#print(loss)
#print(accuracy)
