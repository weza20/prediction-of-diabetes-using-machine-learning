#import st as st
import streamlit as st
import streamlit.components.v1 as com
import sqlite3
#import u

menu=["choose","log in", "sign up" ]
choise= st.sidebar.selectbox("menu",menu)

if choise=="log in":
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
    
    connection = sqlite3.connect('pages/database/database.db')
    cursor = connection.cursor()
    

    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.subheader("log in section ")
    email = st.text_input(" your email ")
    password = st.text_input("password", type='password')
    # choose date and time
    st.date_input(' date')
    st.time_input(' time')
    if st.button("log in "):
        query= "SELECT id,password FROM diabetes_user WHERE mail=?"
        cursor.execute(query, (email,))
        result=cursor.fetchone()
        if result:
            if result[1]==password:
                st.success(" u logged in successfully")
                #u.getFeatures(result[0])
            else:
                st.warning("Not valid email or password ")
        else:
            st.warning("you donot have accont,please sign up ")
    else:
        st.warning("please enter your password ")
    cursor.close()
    connection.close()
        
elif choise=="choose":

    import json
    import requests

    import streamlit as st
    from streamlit_lottie import st_lottie

    url = requests.get(
        "https://assets7.lottiefiles.com/packages/lf20_aynureu3.json")
    url_json = dict()
    if url.status_code == 200:
        url_json = url.json()
    else:
        print("Error in URL")


    st_lottie(url_json,
              # change the direction of our animation
              reverse=True,
              # height and width of animation
              height=400,
              width=400,
              # speed of animation
              speed=1,
              # means the animation will run forever like a gif, and not as a still image
              loop=True,
              # quality of elements used in the animation, other values are "low" and "medium"
              quality='high',
              # THis is just to uniquely identify the animation
              key='Car'
              )
    st.subheader(" please choose sign up or log in ")

else:
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
    st.header("sign up section ")
    user_name = st.text_input("user name")
    password = st.text_input("password", type='password')
    password2= st.text_input("password2", type='password')
    if password2==password==0:
        st.warning("enter your password and confirm it")
    elif password==password2:
        st.success(" your password confirmed")


    else:
        st.warning(" your password didn't confirmed ")
    email = st.text_input(" your email ")

    age = st.number_input("Enter a your age", min_value=0, max_value=120)
    st.write("You entered:", age)
    
    gender = st.radio(
        "Select your Gender",        ('Male', 'Female', 'Others'))
    if gender == 'Male':
        st.write('You have selected Male.')
    elif gender == 'Female':
        st.write("You have selected Female.")
    else:
        st.write("You have selected Others.")
    
    smoking = st.radio("Are you smoking?",('Yes', 'No'))
    if smoking =='Yes':
        somking_f=True
        st.write("You have selected Yes.")
    else:
        somking_f=False
        st.write("You have selected NO.")


   # if st.button("sign up"):
       # st.success("signed up as{}".format( user_name))
    
    if st.button("sign up"):
        
        ########connect with database
        connection = sqlite3.connect('pages/database/database.db')
        cursor = connection.cursor()
        
        query= "SELECT id,password FROM diabetes_user WHERE mail=?"
        cursor.execute(query, (email,))
        result=cursor.fetchone()
        if result:
            st.warning(" you already have an account!")
        else:
            st.success("signed up as{}".format(user_name))
        
            query = "SELECT id, user_id FROM diabetes_user"
            cursor.execute(query)
            results = cursor.fetchall()
            id1=(results[-1][0])+1
            user_id=(results[-1][1])+1

            query = "INSERT INTO diabetes_user (id, user_id, name, mail, smoking, age, gender, password) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"



            cursor.execute(query, (id1, user_id, user_name, email, somking_f, age, gender, password))
            connection.commit()
        
        cursor.close()
        connection.close()
        ###
        #u.getFeatures(id1)

      
    ###############################end  bio data           #####3 #####################################################################
