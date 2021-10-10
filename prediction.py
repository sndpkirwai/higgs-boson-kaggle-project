import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


model = tf.keras.models.load_model('Higgs')

def prediction(model,input):
    prediction = model.predict_classes(input)
    return prediction

def proba(model,input):
    proba = model.predict(input)
    return proba