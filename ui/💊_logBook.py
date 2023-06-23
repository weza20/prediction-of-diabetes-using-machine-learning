import time
import  streamlit as st
from plyer import notification
import requests

import streamlit as st
st.image('https://static.vecteezy.com/system/resources/thumbnails/006/959/727/small/medicine-schedule-or-medical-reminder-planner-flat-style-design-illustration-vector.jpg',
         width=500,use_column_width=False)

s=st.text_input("name of the medicine")
l= st.time_input("time of medicine daily")
if st.button("confirm") :
    notification_title = 'massege from diaAPP'
    notification_message = 'please take your medicine at time '
    notification.notify(
        title=notification_title,
        message=notification_message,
        timeout=10,
        toast=False
    )
    time.sleep(60 * 60 * 60)

else :
    st.warning("please enter your medicine name and time to remind u")



