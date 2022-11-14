# import libs
import streamlit as st
import plotly.express as px
import pandas as pd




st.title('Making interactive graphs using plotly with "gapminder" Dataset')
st.text('Dataset display')
df = px.data.gapminder()
st.write(df)

st.text('Here 5 random samples from Dataset are displayed ')
df = px.data.gapminder()
st.write(df.sample(5))

#to display the columns only
st.write(df.columns)

# summary stats
st.write(df.describe())


# Data management using plotly
year_option = df['year'].unique().tolist()
year = st.selectbox("which year should we plot?", year_option, 0)
# df = df[df['year']== year]

# plotly
fig = px.scatter(df, x= 'gdpPercap', y= 'lifeExp', size= 'pop', color= 'country', hover_name= 'country',
                log_x=True, size_max= 55, range_x= [100,100000], range_y= [20,90],
                animation_frame= 'year', animation_group='country')

fig.update_layout(width=700, height=700)
st.write(fig)