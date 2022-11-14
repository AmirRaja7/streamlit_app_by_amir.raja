# importing libs
import  pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
import seaborn as sns




# make container
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()


with header:
    st.title("This is a Data webapp")
    st.text(" Made in Streamlit ")

with dataset:
    st.title("We will be using pre-existed dataset")
    st.text(" We will import data using seaborn ")
    
    # importing data using seaborn
    df = sns.load_dataset("titanic")
    df= df.dropna()
    st.write(df.head())
    
    st.subheader("How many Males and Females board the Titanic")
    st.bar_chart(df['sex'].value_counts())
    
    # other plot
    st.subheader("Passenger Catagories")
    st.subheader("Class")
    # another plot
    st.bar_chart(df['age'].sample(10)) # takes random 10



with features:
    st.header("These are the App features")
    st.text(" What are the Variables ")
    st.markdown("1. **Feature 1:** First feature of the app")
    st.markdown("2. **Feature 2:** Second feature of the app")





with model_training:
    st.header("In this we will train our Model")
    st.text(" We will train our model using input variables ")
    
    # making column
    input, display = st.columns(2)
    
    # In first column we will provide selection points
    max_depth = input.slider("How many people did board on Titanic: ", min_value=10, max_value=100, value=25, step=5)

# n_estimators
n_estimators = input.selectbox("How many tree should be in a RF?", options=[50,100,200,300,'No Limit'])

# adding list of features
input.write(df.columns)

# input features from user
input_features = input.text_input("Put feature to get relevant data")


# machine learning model

model = RandomForestRegressor(max_depth= max_depth, n_estimators=n_estimators)
# defining the meaning of "No limit to the code"
if n_estimators == 'No limit':
    random_r = RandomForestRegressor(max_depth =  max_depth)
else:
    random_r = RandomForestRegressor(max_depth = max_depth, n_estimators = n_estimators)

# NO LIMIT is still giving error. find the way to solve it.



# define X and y
X=df[[input_features]]
y=df[['fare']]

# fit our model
model.fit(X,y)
pred = model.predict(y)

# Display metrics
display.subheader("Mean absolute error of model is: ")
display.write(mean_absolute_error(y, pred))
display.subheader("Mean squared error of model is: ")
display.write(mean_squared_error(y, pred))
display.subheader("r2_score (r-square score) of model is: ")
display.write(r2_score(y, pred))