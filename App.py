import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

model = tf.keras.models.load_model('Higgs.h5')


def prediction(model, input):
    prediction = model.predict(input)
    print('prediction successful')
    return 's' if prediction[0][0] >= 0.5 else 'b'


def proba(model, input):
    proba = model.predict(input)
    print('probability successful')
    return proba


col = [['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis',
        'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
        'DER_prodeta_jet_jet']]


def main():
    st.header('Higgs Boson Event Detection')

    st.write('This is a simple demo of the Streamlit framework')
    st.write('It demonstrates how to load a model, make predictions, and display the results')
    st.write('The model was trained on the Higgs Boson dataset')

    st.subheader('Input the Data')
    st.write('Please input the data below')

    i = st.number_input('DER_mass_MMC', )
    j = st.number_input('DER_mass_transverse_met_lep', )
    k = st.number_input('DER_mass_vis', )
    l = st.number_input('DER_pt_h', )
    m = st.number_input('DER_deltaeta_jet_jet', )
    n = st.number_input('DER_mass_jet_jet', )
    o = st.number_input('DER_prodeta_jet_jet', )

    input = np.array([[i, j, k, l, m, n, o]])
    print(type(i))
    print(input)

    if st.button('Detect Event'):
        pred = prediction(model, input)
        st.success('The event is predicted is ' + pred)

    if st.button('Show Probability'):
        prob = proba(model, input)
        st.success('The probability of the event is {}'.format(prob[0][0]))


if __name__ == '__main__':
    main()
