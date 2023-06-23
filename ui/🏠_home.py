import streamlit as st
from PIL import Image

#  🏠_home.py
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

image = Image.open('pages/c.jpeg')

st.image(image, caption='diabetes', width=600 )

# header
with st.container():
    st.title("the ultimate way to detect  diabetes ")
    st.sidebar.success("select a page above :")
st.write("+ making the detection  of diabetes easier ")
st.write("+ helping of treatment journey of diabetic people")
st.write("+ allow more understanding of the diabetes ")

# -----------------------------
# ----------------------------------------------------------------------------


with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)  # build a line
    with left_column:
        st.title("bio data page")
        image = Image.  open('pages/aa.jpg')

        st.image(image, caption='bio data')

    # st.write("##") # line
st.write(
    "* bioDatapage is the page where u drop your bio information  to help us to detect diabetes")
st.write("* Early detection is key in diabetes because early treatment can prevent serious complications. When a "
         "problem with blood sugar is found, doctors and patients can take steps to prevent permanent damage to the "
         "heart, kidneys, eyes, nerves, blood vessels, and other vital organs , whith simple process in our website ")
st.write(
    "* Diabetic care often focuses on treatment of the condition. While treatment is important, early detection increases the potential for effective changes early in the disease process ")
######################streamlit run ##########################################

with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)  # build a line
        with left_column:

                     st.title(" log book page ")
image = Image.open('pages/d.png')
st.image(image, caption='reminder' ,width=300)




st.write(
    " *A logbook for glucose, insulin, nutrition, medications, injection sites, notes to remind ")
st.write(" We hope to get a controlled blood sugar level through this section ")
############################################


with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)  # build a line
    with left_column:
        st.header("about us : ")
        st.write("##")  # line

        image = Image.open('pages/aa.jfif')
        st.image(image, caption='about us', width=300)

st.write("we looking for :")
st.write("_ creating a Machine Learning methodology that can detect diabetes in easily with high accuracy ")
st.write(
    "_ Exploiting MLA  Algorithms is essential if healthcare professionals are able to identify diseases more effectively")
st.write(
    "_improve the medical diagnosis of diabetes this research explored and contrasts various MLA that can identify diabetes risk early")
st.write("_Finally, it can help control diabetes and reduce its spread and fatal effects")


