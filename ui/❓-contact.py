import streamlit as st

st.header(":mailbox:contact us")
import streamlit.components.v1 as com

import requests

import streamlit as st
from streamlit_lottie import st_lottie

url = requests.get(
    "https://assets6.lottiefiles.com/packages/lf20_eroqjb7w.json")
# Creating a blank dictionary to store JSON file,
# as their structure is similar to Python Dictionary
url_json = dict()

if url.status_code == 200:
    url_json = url.json()
else:
    print("Error in the URL")

st_lottie(url_json)
# com.iframe("https://embed.lottiefiles.com/animation/112608")
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
page_icon = "🧊"
##############################################################3

with st.container():
    st.write("---")

st.write("##")  # line        st.header(":mailbox:contact us")
contact_form = """
<form action="https://formsubmit.co/diateam23@gmail.com" method="POST">
    <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name"placeholder=" your name " required>
     <input type="email" name="email"placeholder="your email "  required>
     <textarea name="message" placeholder="Details of your problem"></textarea>
         <button type="submit">Submit</button>
         <button type="cancel">cancel</button>

</form>
"""
st.markdown(contact_form, unsafe_allow_html=True)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("pages/style/style.css")