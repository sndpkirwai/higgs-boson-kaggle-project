import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

df = pd.read_csv('training.csv',nrows=30000)

df = df[['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis',
       'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
       'DER_prodeta_jet_jet','Label']]

X = df.drop(columns=['Label'])
y = df['Label']

y = y.replace('s',0)
y = y.replace('b',1)
X = X.values
y = y.values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(30, input_dim=30, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

model = create_baseline()

history = model.fit(X_train,y_train,validation_data =(X_test,y_test), epochs=80)

model.save('Higgs')