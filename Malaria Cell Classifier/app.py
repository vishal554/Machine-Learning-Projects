import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2 as cv
import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from models.gradcam import GradCAM
import imutils

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

IMG_SIZE = 50
CATEGORIES = ["Parasitized", "Uninfected"]

#######################################################################################

def teachable_machine_classification(image_array, model_file):
    # Load the model
    model = keras.models.load_model(model_file)

    # Create the array of the right shape to feed into the keras model
    img_arr = cv.resize(image_array, (IMG_SIZE, IMG_SIZE))
    new_arr = img_arr.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    # normalize the array
    new_arr = new_arr/255.0

    # run the inference
    pred = model.predict_classes(new_arr)
    return pred

######################################################################################################

st.title("Malaria parasite detection")
st.header("Malaria infected cell Classification Example")
st.text("Upload a cell Image for image classification as infected/Parasitized or uninfected")

uploaded_file = st.file_uploader("Choose a cell ...", type="png")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded cell.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    st.write(uploaded_file.name)
    
    image1 = image.resize((50,50))
    gray = image1.convert("L")
    img_array = np.array(gray)


    label = teachable_machine_classification(img_array, "models/64_CNN.model")

    if label > 0.50:
        st.write("The cell is Infected with malaria")

        model = keras.models.load_model("models/64_CNN.model")
        image2 = image.resize((100,100))

        doc = keras.preprocessing.image.img_to_array(gray) # -> numpy array
        doc = np.expand_dims(doc, axis=0)

        doc = doc/255.0

        cam = GradCAM(model, 0, "conv2d_2")
        heatmap = cam.compute_heatmap(doc)
        
        # resize the resulting heatmap to the original input image dimensions
        # and then overlay heatmap on top of the image
        heatmap = cv.resize(heatmap, (100,100))

        (heatmap, output) = cam.overlay_heatmap(heatmap, np.array(image2) , alpha=0.5)

        # # draw the predicted label on the output image

        # cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
        # cv2.putText(output, "label", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,``
        # 	0.8, (255, 255, 255), 2)
        # display the original image and resulting heatmap and output image
        # to our screen
        output = np.hstack([image2, output])
        output = imutils.resize(output, height=500, width=1000)
        st.image(output, caption="Hotspots that influence the prediction")
            
    else:
        st.write("The cell is not infected with malaria")




####################################################################################
    

    

    
        



    