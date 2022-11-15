# PIMA indians diabetic data analysis and ploting with Streamlit

# import libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# loading dataset
df = pd.read_csv("D:\Python ka Chilla Dobara\Day_28_Dashboards_and_webapps_with_streamlit\diabetes.csv")

# Heading
st.title("Diabetes Prediction App")
st.sidebar.header('Patient Data')
st.subheader('Description Stats of Data')
st.write(df.describe())

# Data split into X, y and Train test split
X = df.drop(['Outcome'], axis=1)
y = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=0)

# function
def user_report():
    Pregnancies = st.sidebar.slider("Pregnencies", 0, 17, 2 )   #Description ("col_name", min_value, max_value, default_slider_value)
    Glucose = st.sidebar.slider("Glucose", 0, 199, 50 )
    BloodPressure = st.sidebar.slider("BloodPressure", 0, 122, 60 )
    SkinThickness = st.sidebar.slider("SkinThickness", 0, 99, 10 )
    Insulin = st.sidebar.slider("Insulin", 0, 846, 100 )
    BMI = st.sidebar.slider("BMI", 0.0, 67.1, 5.0)
    DiabetesPedigreeFunction = st.sidebar.slider("DiabetesPedigreeFunction", 0.078, 2.420, .2 )
    Age = st.sidebar.slider("Age", 21, 81, 24 )
    
    user_report_data= {
        "Pregnancies":Pregnancies,
        "Glucose" : Glucose,
        "BloodPressure" : BloodPressure,
        "SkinThickness" : SkinThickness,
        "Insulin" : Insulin,
        "BMI" : BMI,
        "DiabetesPedigreeFunction" : DiabetesPedigreeFunction,
        "Age" : Age}
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Patient data
user_data = user_report()
st.subheader("Patient Data")
st.write('user_data')

# Model 
rc = RandomForestClassifier()
rc.fit(X_train,y_train)
user_result = rc.predict(user_data)


# y_pred = rc.predict(y)
# # y_pred = rc.predict(y)

# Visualization
st.title("Visualized Patient Data")

# Color function
if user_result[0] == 0:
    color = "blue"
else: 
    color = 'red'

# Age Vs Pregnancies plot 
st.header("Pregnancy Count Graph (Others vs Yours)")
fig_preg = plt.figure()
ax1 = sns.scatterplot(x = 'Age', y= "Pregnancies", data = df, hue = 'Outcome', palette = "Greens")
ax2 = sns.scatterplot(x = user_data['Age'], y = user_data['Pregnancies'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title("0 = Healthy & 1 = Diabetic")
st.pyplot(fig_preg)



# Bolow code running OK

# Assignment 01
# Age Vs BloodPressure plot 
st.header("Effect of Age on Human Blood Pressure (Others vs Yours)")
fig_bppreg = plt.figure()
axbp1 = sns.scatterplot(x = 'Age', y= "BloodPressure", data = df, hue = 'Outcome', palette = "Greens")
axbp2 = sns.scatterplot(x = user_data['Age'], y = user_data['BloodPressure'], s = 150, color = color)     # s = graph scale
plt.xticks(np.arange(21,81,24))
plt.yticks(np.arange(0,122,60))
plt.title("0 = Healthy & 1 = Diabetic")
st.pyplot(fig_bppreg)

# Output
st.header("Your Report: ")
output = ''
if user_result[0] ==0:
    output = 'You are Healthy 😍'
    st.balloons()
else:
    output = "Take less Sugar and get medical Care"
    st.warning("Avoid Sugar")
st.title(output)




# # Display metrics
st.subheader("Mean absolute error of model is: ")
st.write(np.square(np.subtract(y,user_result).mean()))
# MAR= (np.square(np.subtract(y,user_result).mean()))
# st.subheader("Mean squared error of model is: ")
# st.write(MAR(y, user_result))
# st.subheader("r2_score (r-square score) of model is: ")
# st.write(r2_score(y, y_pred))



# st.subheader("Accuracy: ")
# st.write(str(accuracy_score(y_test, rc.predict(X_test)*100 + "%")))