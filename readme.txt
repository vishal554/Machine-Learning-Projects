
Hello,
please see below points for details of each files.

1] gradcam.py is the class from which heatmap will be generated
2] Conv2d is the keras model that has been trained to construck 64_CNN.model file
3] malaria_X_normalized and malaria_y_normalized are the numpy arrays generated from the Cell_classifier_dataset_extraction file which converts the images to grayscale and resizes them to 50x50.
4] try_changing_layers is the file that has been used to train the model for 27 times by permuting over the number of layers and layer numbers. the optimum model is chosen by plotting the graph in TensorBoard
____________________________________________________________________________________________________________________________

How to run the project: 

Open cmd from this directory and run the command :> streamlit run app.py

____________________________________________________________________________________________________________________________

Developed by Vishal panchal 
contact: 9987190959 / vishalpanchal338@gmail.com


