import streamlit as st
import pickle
import os 
import sys
import tensorflow as tf
from hate.logger import logging
from hate.exception import CustomException
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from hate.pipeline.predict_pipeline import PredictionPipeline
import uvicorn

app = FastAPI()

@app.get("/",tags=['authentication'])
async def index():
    return RedirectResponse(url="/docs")

@app.post("/predict")
async def predict(text: str):
    try:
        obj = PredictionPipeline()
        result = obj.predict(text)
        return {"prediction": result}
    except Exception as e:
        raise CustomException(sys,e)
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)