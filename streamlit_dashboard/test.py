# importing libs
import streamlit as st
import seaborn as sns


st.header("This is a tribute to the saviour of mankind Prophet Muhammad")
st.text("Peace be Upon him")

st.header("The month of the King of Sufis and Scholars")
st.header("Al Shaikh Abdul Qadir Jilanni")
st.text("May Allah be pleased with him")

df = sns.load_dataset('iris')
st.write(df[['species', 'sepal_length', 'petal_length']].head(10))

# we can also make bar plots of individual characteristics
st.bar_chart(df['sepal_length'])

# we can also make bar plots of multiple characteristics
# st.bar_chart(df[['sepal_length','petal_length']])

# making line chart based on single variable
# st.line_chart(df['sepal_length'])

# making line chart based on multiple variables
st.line_chart(df[['sepal_length','petal_length']])









# to run this file
# streamlit run "filename".py
