import streamlit as st
import pandas as pd
import numpy as np
from hate.exception import CustomException
from hate.logger import logging
import requests

st.title('Hate Speech Detection')

# User input for text
text = st.text_input('Enter text')

# get request from api 
if st.button('Predict'):
    try:
        response = requests.post('http://localhost:8000/predict', json={'text': text})
        if response.status_code == 200:
            prediction = response.json()['prediction']
            st.success(f'The prediction is: {prediction}')
        else:
            st.error('API request failed')
    except requests.exceptions.RequestException as e:
        st.error(f'An error occurred: {e}')