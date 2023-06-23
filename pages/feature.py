import streamlit as st
import pickle
import sqlite3

    
model = pickle.load(open('model.pkl', 'rb'))


def predict_result(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
    input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]
    result = model.predict(input_data)
    return result[0]

#def getFeatures(user_id):
user_id = 7
pregnancies = st.number_input('Enter the number of pregnancies')
glucose = st.number_input('Enter your glucose level')
blood_pressure = st.number_input('Enter your blood pressure')
skin_thickness = st.number_input('Enter your skin thickness')
insulin = st.number_input('Enter your insulin level')
BMI = st.number_input('Enter your BMI')
diabetes_pedigree_function = st.number_input('Enter your Diabetes Pedigree Function')
age = st.number_input('Enter your age')

# Validate the input values
if pregnancies < 0 or pregnancies > 20:
    st.error('Number of pregnancies should be between 0 and 20.')

if glucose < 0 or glucose > 200:
    st.error('Glucose level should be between 0 and 200.')

if blood_pressure < 0 or blood_pressure > 200:
    st.error('Blood pressure should be between 0 and 200.')

if skin_thickness < 0 or skin_thickness > 120:
    st.error('Skin thickness should be between 0 and 120.')

if insulin < 0 or insulin > 1000:
    st.error('Insulin level should be between 0 and 1000.')

if BMI < 0 or BMI > 80:
    st.error('BMI should be between 0 and 80.')

if diabetes_pedigree_function < 0 or diabetes_pedigree_function > 3:
    st.error('Diabetes Pedigree Function should be between 0 and 3.')

if age < 1 or age > 100:
    st.error('Age should be between 1 and 100.')

# Continue with the rest of your code
# ...


if st.button('Check the Result'):
    result = predict_result(pregnancies, glucose, blood_pressure, skin_thickness, insulin, BMI, diabetes_pedigree_function, age)
    if result==1:
        finalresult ="Postive"
    else:
        finalresult ="Negative"
    st.write('The predicted result is:', finalresult)

    ## connect with the database
    connection = sqlite3.connect('pages/database/database.db')
    cursor = connection.cursor()

    query = "SELECT diagnoses_id FROM diabetes_diagnoses_feature"
    cursor.execute(query)
    results2 = cursor.fetchall()

    id2=(results2[-1][0])+1


    query = "INSERT INTO diabetes_diagnoses_feature (diagnoses_id, Pregnancies, glucose, BloodPressure, skin_thickness, insulin, Bmi, diabetespedegreefunction, age, user_diag_id,result,outcome) VALUES (?, ?, ?, ?, ?, ?, ?, ?,?, ?,?,?)"

    cursor.execute(query, (id2 ,pregnancies, glucose, blood_pressure, skin_thickness, insulin, BMI, diabetes_pedigree_function, age, user_id,finalresult,int(result)))

    connection.commit()

    cursor.close()
    connection.close()

