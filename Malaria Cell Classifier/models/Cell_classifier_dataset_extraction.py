# Malaria cell classifier


#importing the libraries

import numpy as np
import pandas as pd
import os
import cv2 as cv
import matplotlib.pyplot as plt
import random

# Defining variables

DATADIR = "./cell_images/"
CATEGORIES = ["Uninfected", "Parasitized"]
IMG_SIZE = 50
training_data = []


# Creating the data set from images 

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_arr = cv.imread(os.path.join(path, img), cv.IMREAD_GRAYSCALE)
                new_img_arr = cv.resize(img_arr, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_img_arr, class_num])
            except Exception as e:
                print(e)

create_training_data()

# Shuffle the data

random.shuffle(training_data)

# seperate Features and Labels and convert it to NumPy Array so that we can run it to tensorflow

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

#normalize the array/image
X = X / 255.0

# check the shuffled data
print(y[0:10])

# saving the dataset

np.save("malaria_X_normalized", X)
np.save("malaria_y_normalized", y)


