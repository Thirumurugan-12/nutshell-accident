# import all the app dependencies
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from IPython import get_ipython
from PIL import Image


# load the encoder and model object
model = joblib.load("rta_model_deploy3.joblib")
encoder = joblib.load("ordinal_encoder2.joblib")

st.set_option('deprecation.showPyplotGlobalUse', False)

# 1: serious injury, 2: Slight injury, 0: Fatal Injury

st.set_page_config(page_title="Accident Severity Prediction App",
                page_icon="🚧", layout="wide")

#creating option list for dropdown menu
op_Accident_Classification = ["Road Accidents","Rail Road Accidents","Other Railway Accidents","Not Applicable"]

op_Accident_Spot = ['Bottleneck', 'Bridge', 'Cross roads', 'Not Applicable',
       'More than four arms', 'Road hump or Rumble strips', 'Junction',
       'Curves', 'Other', 'T Junction', 'Offset', 'Narrow road',
       'Culvert', 'Staggered junction', 'Y Junction', 'Circle',
       'Railway crossing', 'Rail Crossing manned',
       'Round about or Circle', 'Rail Crossing Unmanned']

op_Accident_Location = ['Rural Areas', 'Villages settlement', 'City/Town',
       'Not Applicable']

features = ["Accident_Classification","Accident_Spot","Accident_Location"]

# Give a title to web app using html syntax
st.title("Accident Severity Prediction App 🚧")
st.sidebar.title("Accident Detection")
# define a main() function to take inputs from user in form based approch
def main():
    with st.form("road_traffic_severity_form"):
        st.subheader("Please enter the following inputs:")
        
        Noofvehicle_involved = st.slider("Noofvehicle_involved:",1,10, value=0, format="%d")
        Accident_Classification = st.selectbox("Accident_Classification:",options=op_Accident_Classification)
        Accident_Spot = st.selectbox("Accident_Spot:", options=op_Accident_Spot)
        Accident_Location = st.selectbox("Accident_Location:", options=op_Accident_Location)
        
        submit = st.form_submit_button("Predict")

    # encode using ordinal encoder and predict
    if submit:
        input_array = np.array([Accident_Classification,Accident_Spot,Accident_Location], ndmin=2)

        encoded_arr = list(encoder.transform(input_array).ravel())

        num_arr = [Noofvehicle_involved]
        pred_arr = np.array(num_arr + encoded_arr).reshape(1,-1)     

        ans = ['Damage Only', 'Fatal' ,'Grievous Injury' ,'Simple Injury' ,'Unknown']

        prediction = model.predict(pred_arr)
        print(prediction) 
        
        st.write(ans[prediction[0]])
            


if __name__ == '__main__':
   main()


