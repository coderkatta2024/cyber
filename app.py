import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model # pyright: ignore[reportMissingImports]
import base64
import warnings
import pandas as pd
from streamlit_option_menu import option_menu
from PIL import Image
import cv2
import os
import re

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set the background of the Streamlit app
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/jpeg;base64,%s");
    background-position: center;
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown('<style>h1 {  font-family: "Times New Roman", Times, serif; font-size: 30px; text-align: left; color: black; margin-left: 20px; font-weight: bold; }</style>', unsafe_allow_html=True)

    st.markdown('<style>p { color: Black; }</style>', unsafe_allow_html=True)
    st.markdown(page_bg_img, unsafe_allow_html=True)



# Setting Page Configuration
st.set_page_config(page_title="An Unsupervised Adversarial Autoencoder for Cyber Attack Detection in Power Distribution Grids",
                   layout="wide",
                   initial_sidebar_state="expanded",
                   menu_items={'About': """Hybrid CNN and LSTM Algorithm Autoencoder for Cyber Attack Detection in Power Distribution Grids"""})

st.markdown("<h1>An Unsupervised Adversarial Autoencoder for Cyber Attack Detection in Power Distribution Grids</h1>",
            unsafe_allow_html=True)

# Creating Option Menu
selected = option_menu(None, ["Home", "Abstract", "Prediction"],
                       icons=["house", "cloud-upload", "pencil-square"],
                       default_index=0,
                       orientation="horizontal",
                       styles={"nav-link": {"font-size": "25px", "text-align": "center", "margin": "-2px", "--hover-color": "#a7b300"},
                               "icon": {"font-size": "25px"},
                               "container": {"max-width": "6000px"},
                               "nav-link-selected": {"background-color": "#a7b300"}})

set_background('background/1.jfif')



Abstract_Cont = """
    <h1 >
       PROJECT ABSTRACT 

        1.To address these challenges, this paper proposes an unsupervised adversarial autoencoder (AAE)

        2.Model to detect FDIAs in unbalanced power distribution grids integrated with DERs

        3.PV systems and wind generation. 

        4.The proposed method utilizes long short-term memory (LSTM) in the structure of the autoencoder 

        5. The advantage of the proposed data-driven model or mathematical models.

    </h1>

"""

Abstract_Cont1 = """
    <h1 >
       PROJECT OBJECTIVE 

        1.Develop a hybrid deep learning approach to classify encoded power Grid Cyberattack .

        2.The power Grid Fault or Non Fault can be Detected . 

        3.Attack Detection using hybrid Deep Learning Algorithm. 

        4.Model prediction on Long short-term memory (LSTM).

        5.Improved Performance Analysis .

    </h1>

"""
prediction = """
    <h1 >
       PROJECT PREDICTION 
       1.HYBRID CNN & LSTM ALGORITHM   
    </h1>

"""
if selected == "Home":
    st.markdown(Abstract_Cont1, unsafe_allow_html=True)

if selected == "Abstract":
    st.markdown(Abstract_Cont, unsafe_allow_html=True)

if selected == "Prediction":
    st.markdown(prediction, unsafe_allow_html=True)
    # Load the model
    model = load_model('Model.h5')
    
    # Function to make predictions
    def make_prediction(test_sample):
        test_sample = np.expand_dims(test_sample, axis=0)
        predictions = model.predict(test_sample)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_class = int(predicted_class[0])
        class_labels = ['Deductive (Non-Fault)', 'Additive (Fault)']
        predicted_label = class_labels[predicted_class]
        return predicted_label, predictions
    
    st.write("Enter your input data for the features:")
    
    # Define feature labels
    feature_labels = ['current in phase A (Ia)', 'current in phase B (Ib)', 'current in phase C (Ic)', 
                      'Voltage in phase A (Va)', 'Voltage in phase B (Vb)', 'Voltage in phase C (Vc)']
    
    # Create input fields for each feature
    feature_values = []
    for label in feature_labels:
        value = st.text_input(f"Enter value for {label}", value='0.0')
        feature_values.append(value)
    
    if st.button('Prediction'):
        try:
            # Convert feature values to numpy array
            test_sample = np.array([float(value) for value in feature_values])
            
            # Display the entered test sample
            st.write("Entered Test Sample: ", test_sample)
            
            # Make predictions
            predicted_label, predictions = make_prediction(test_sample)
            
            # Display the results
            st.write("Predicted Class: ", predicted_label)
            st.write("Model Output: ", predictions)
        except ValueError:
            st.error("Please enter valid numerical values for all features.")
    
    if st.button('Dataset Test Sample'):
        # Example test sample
        test_sample = np.array([-0.0291654 ,  0.22419893, -0.25310159, -1.74378807,  1.1811018 , 0.51398549])
        st.write("Test Sample: ", test_sample)
        
        # Make predictions
        predicted_label, predictions = make_prediction(test_sample)
        
        # Display the results
        st.write("Predicted Class: ", predicted_label)
        st.write("Model Output: ", predictions)
