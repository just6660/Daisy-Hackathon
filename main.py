import streamlit as st
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from PIL import Image
import sklearn

st.set_page_config(layout="wide")

# Create a title and a sub title
st.write("""
# Apartment Price Machine Learning Predictive Model
""")

st.write("""# Introduction
This webapp was created for the 2022 Daisy Intelligence Hackathon. It is a machine inteligence web application project with a machine learning predictive component. Scroll down to explore further features such as booking a meeting with a Waterloo advisor. 
""")

Image = Image.open('logo.png')
st.image(Image, use_column_width= True)

# Getting User Input
def get_user_input():

    latitude = st.number_input('Insert a latitude')
    longitude = st.number_input('Insert a longitude')
    bedrooms = st.number_input('Insert number of bedrooms')
    bathrooms = st.number_input('Insert number of bathrooms')
    den = st.number_input('Insert number of dens')




    user_data = {
        'Lat': latitude,
        'Long': longitude,
        'Den': den,
        'bathroom': bathrooms,
        'bedroom': bedrooms
    }
    return user_data



user_input = get_user_input()
st.write(user_input.get("Lat"))


