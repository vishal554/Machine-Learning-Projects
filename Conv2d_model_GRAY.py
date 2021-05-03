import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, MaxPooling2D, Conv2D, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU

# definig variables

NAME = f"malaria_cell_classifier_CNN_2x64-{int(time.time())}"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

# loading the dataset

X = np.load("malaria_X_normalized.npy")
y = np.load("malaria_y_normalized.npy")

#Creating our Convulation model and training it

model = Sequential()

model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
          
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))
          
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        
model.fit(X,y,batch_size=1, validation_split=0.3, epochs=7, callbacks=[tensorboard])

model.save("64_CNN.model") 





