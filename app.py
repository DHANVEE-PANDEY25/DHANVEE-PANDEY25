import streamlit as st
import numpy as np
import pandas as pd
import pickle


rfr = pickle.load(open('rfr.pkl','rb'))


X_train = pd.read_csv('X_train.csv')

def pred(Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp):
    features = np.array([[Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp]])
    prediction = rfr.predict(features).reshape(1,-1)
    return prediction[0]

st.title("Calories Burn Prediction")


Gender = st.selectbox('Gender',X_train['Gender'])
Age = st.selectbox('Age',X_train['Age'])
Height = st.selectbox('Height',X_train['Height'])
Weight= st.selectbox('Weight',X_train['Weight'])
Duration = st.selectbox('Duration(minutes)',X_train['Duration'])
Heart_Rate= st.selectbox('Heart Rate (bpm)',X_train['Heart_Rate'])
Body_Temp = st.selectbox('Body Temperature',X_train['Body_Temp'])

result = pred(Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp)

if st.button('predict'):
    if result:
        st.write("YOU HAVE CONSUMED THIS CALORIES :",result)






