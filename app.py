import streamlit as st
import requests

# Title of the Streamlit app
st.title("Text Harm Prediction App")

# Input for the user to enter text
user_input = st.text_area("Enter Text for Harm Prediction:")

# Make an API request if the user has entered text
if st.button("Predict"):
    if user_input:
        # Call the Render API with the text parameter in the body
        url = "https://text-harm-prediction.onrender.com/predict"
        
        # Send the POST request with the text as JSON data
        response = requests.post(url, json={"text": user_input})
        
        # Check if the response was successful
        if response.status_code == 200:
            prediction = response.json()  # Assuming the API returns a JSON response
            st.write("Prediction:", prediction)
        else:
            st.error(f"Error in prediction: {response.status_code}. {response.text}")
    else:
        st.warning("Please enter some text.")
