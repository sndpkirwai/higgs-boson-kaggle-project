# IMPORTING LIBRARIES
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.impute import SimpleImputer
from collections import Counter
import seaborn as sb
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt

st.title('HIGGS BOSON - DEEP LEARNING PROJECT')
st.subheader('Upload the Higgs Boson dataset: (.csv)')
# creating a side bar
st.sidebar.info("Created By : Sandeep Kirwai")
# Adding an image to the side bar
st.sidebar.subheader("Contact Information : ")
col1, mid, col2 = st.sidebar.beta_columns([1, 1, 20])
with col1:
    st.sidebar.subheader("LinkedIn : ")
with col2:
    st.sidebar.markdown(
        "[![Linkedin](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTLsu_X_ZxDhuVzjTHvk4eZOmUDklreUExhlw&usqp=CAU)](https://www.linkedin.com/in/sandeep-kirwai-2888bb72/)")

col3, mid, col4 = st.sidebar.beta_columns([1, 1, 20])
with col3:
    st.sidebar.subheader("Github : ")
with col4:
    st.sidebar.markdown(
        "[![Github](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQJGtP-Pq0P67Ptyv3tB7Zn2ZYPIT-lPGI7AA&usqp=CAU)](https://github.com/sndpkirwai)")

# if the user chooses to upload the data
file = st.file_uploader('Dataset')
# browsing and uploading the dataset (strictly in csv format)
# dataset = pd.DataFrame()
flag = False

if file is not None:

    dataset = pd.read_csv(file)
    # flag is set to true as data has been successfully read
    flag = "True"
    st.header('**HIGGS BOSON DATA**')
    st.write(dataset.head())
    # dataset.drop("EventId", axis=1, inplace=True)

    st.write("PERFORM EXPLORATORY DATA ANALYSIS")

    st.subheader("Labels distribution")
    st.bar_chart(dataset["Label"].value_counts())
    st.subheader("Finding no. of null values per column in the dataset")
    st.write('Getting dummy variable for target column')
    dataset['Label'] = dataset['Label'].replace('s',0)
    dataset['Label'] = dataset['Label'].replace('b',1)
    st.write('converted label counts')
    st.write(dataset['Label'].value_counts())
    st.write(dataset.isna().sum())
    st.subheader("Statistical information about the dataset")
    st.write(dataset.describe())
    st.subheader("Shape of dataset")
    st.write(dataset.shape)

    st.subheader("Correlation matrix of the features")
    corr = dataset.corr()
    # Get the absolute value of the correlation
    cor_target = abs(corr['Label'])
    fig, ax = plt.subplots()
    sb.heatmap(corr, annot=True, ax=ax)
    st.write(fig)

    y = None
    x = None

    st.subheader('After looking at correlation plot, which threhold point you have to set for feature selection :')
    k = st.number_input('', step=0.1, min_value=0.1, value=0.3)
    if k > 0:
            # Select highly correlated features (thresold = 0.2)
            relevant_features = cor_target[cor_target > k]

            # Collect the names of the features
            names = [index for index, value in relevant_features.iteritems()]

            # Drop the target variable from the results
            names.remove('Label')

            st.subheader('Based on provided threshold, best features are:')
            # Display the results
            st.write(names)

            y = dataset['Label']
            x = dataset[names]

    st.subheader('As you can see from the dataset. It is having negative values as well. :')
    st.subheader('Also dataset contains -999 values which is less relevent so which technique you want to use to imputing')
    mopt = st.selectbox("Select :", ("convert to zero", "convert to mean"))
    # "Click to select",
        if (st.button("START imputing")):
            if "convert to zero" in mopt:

                imp_zero = SimpleImputer(missing_values=-999.0, strategy='constant', fill_value=0)
                # Imputation transformer for completing missing values.
                imp_zero.fit(x)
                dataset = imp_zero.transform(x)
            if "convert to mean" in mopt:
                imp_mean = SimpleImputer(missing_values=-999.0, strategy='mean')
                        # Imputation transformer for completing missing values.
                imp_mean.fit(x)
                dataset = imp_mean.transform(x)
                    
            st.write(dataset)

    st.subheader('As we know that data has skewness so need to scale numeric columns ')
    st.subheader('Which technique you want to use ')

    scalopt = st.selectbox("Select :", ("StandardScaler", "Normalize"))
    # "Click to select",
    if (st.button("START scalling")):
        if "StandardScaler" in scalopt:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            x = scaler.fit_transform(dataset)
        if "Normalize" in scalopt:
            from sklearn.preprocessing import normalize
            x = normalize(dataset)
                    
        st.write(x)
    st.subheader('Test size split of users choice:')
    st.text('Default is set to 20%')
    splt = st.number_input('', step=5, min_value=10, value=20)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=splt * 0.01, random_state=0)
    st.write("Data is being split into testing and training data!")
                    # Splitting the data into 20% test and 80% training data
                    # Outlier detection and removal
    st.success("Data cleaned!")

