import time
import  streamlit as st
from plyer import notification
import requests
import streamlit as st

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


st.image("https://media.istockphoto.com/id/1470115399/vector/medication-calendar-treatment-schedule-cartoon-medical-healthcare-concept-doctor-and-patient.jpg?s=612x612&w=0&k=20&c=Ijp125dVh0Lw5m2kaUxRqjX79FHm-QO9-lg-IjrzcYA=",

         width=500,use_column_width=False)

s=st.text_input("name of the medicine")
l= st.time_input("time of medicine daily")
p=st.text_input(" any medication notes u want to remind ")
if st.button("confirm") :
    notification_title = 'massege from diaAPP'
    notification_message = 'please take your medicine at time '
    notification.notify(
        title=notification_title,
        message=notification_message,
        app_icon="Paomedia-Small-N-Flat-Bell.ico",

        timeout=10,
        toast=False
    )
    time.sleep(60 * 60 * 24)

else :
    st.warning("please enter your medicine name and time to remind u")



