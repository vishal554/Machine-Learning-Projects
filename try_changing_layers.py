# compare 27 models by tweaking the layer sizes and number of layers

import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, MaxPooling2D, Conv2D, Flatten, Dropout

dense_layers = [0,1,2]
conv_layers = [1,2,3]
layer_sizes = [32,64]

X = np.load("malaria_X_normalized.npy")
y = np.load("malaria_y_normalized.npy")

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for  conv_layer in conv_layers:
            NAME = f"{conv_layer}-conv-{dense_layer}-dense-{layer_size}-layers-{int(time.time())}"
            tensorboard = TensorBoard(log_dir="bunch_of_logs/{}".format(NAME))

            model = Sequential()

        
            model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer-1):
    
                model.add(Conv2D(layer_size,(3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))
            

            model.add(Flatten())

            for l in range(dense_layer):
                    
                model.add(Dense(layer_size))
                model.add(Activation("relu"))


            model.add(Dense(1))
            model.add(Activation("sigmoid"))
                    
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
                    
            model.fit(X,y,batch_size=32, validation_split=0.3, epochs=7, callbacks=[tensorboard])