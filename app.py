import streamlit as st
from hate.pipeline.predict_pipeline import PredictionPipeline
from hate.exception import CustomException
import sys

# Streamlit App
def main():
    st.title("Hate Speech Detection")
    st.subheader("Enter text to analyze if it contains hate speech.")

    # Input text
    user_input = st.text_area("Input Text", placeholder="Type or paste your text here...")

    if st.button("Predict"):
        try:
            # Create PredictionPipeline object
            obj = PredictionPipeline()
            
            # Predict using input data
            result = obj.predict(text=user_input)
            
            # Display the result
            st.success(f"Prediction: {result}")
        
        except Exception as e:
            # Handle and display exceptions
            st.error("An error occurred during prediction.")
            st.error(CustomException(sys, e))

if __name__ == "__main__":
    main()
