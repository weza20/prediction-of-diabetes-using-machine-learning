import streamlit as st
from PIL import Image

page_bg_img = """
<style>
[data-testId ="stAppViewContainer"]{
background-color: #000405;
opacity: 1;
background-image: radial-gradient(#398040 0.5px, #000405 0.5px);
background-size: 10px 10px;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
image =Image.open('pages/,.jpg')
st.title(" rate us  ")
st.image(image, caption='', width=700 )





st.write( "we care about your opinion ")

import streamlit.components.v1 as com

x = st.slider('choose a number', 1, 5)
col1, col2, col3 = st.columns([2, 0.2, 2])
with col1: st.empty()
with col2:
    if x == 1: st.write(":star2:")
    if x == 2: st.write(":star2:",":star2:")
    if x == 3: st.write(":star2:",":star2:",":star2:")
    if x == 4: st.write(":star2:",":star2:",":star2:",":star2:")
    if x == 5: st.write(":star2:",":star2:",":star2:",":star2:",":star2:")
with col3: st.empty()