import streamlit as st
from dummies import *
import joblib
import pandas as pd
import requests
import joblib
import io

# Function to download and load model from a raw GitHub URL
def load_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the download failed
    return io.BytesIO(response.content)

# Define the raw URLs for your model, scaler, and feature files
model_url = "https://github.com/AhmedSalem225/Final-Project_Epsilon-AI/raw/80b6387b2bcc4dd71d4bebc38cbda1c921752c6c/models/model.h5"
scaler_url = "https://github.com/AhmedSalem225/Final-Project_Epsilon-AI/raw/80b6387b2bcc4dd71d4bebc38cbda1c921752c6c/models/scaler.h5"
input_data_url = "https://github.com/AhmedSalem225/Final-Project_Epsilon-AI/raw/80b6387b2bcc4dd71d4bebc38cbda1c921752c6c/models/feature.h5"

# Load the model and scaler
model = joblib.load(load_from_url(model_url))
scaler = joblib.load(load_from_url(scaler_url))
input_data = joblib.load(load_from_url(input_data_url))







st.title("ENHANCING TRAFFIC ACCIDENT CLASSIFICATION USING MACHINE LEARNING")
st.info('The primary goal of this project is to classify traffic accidents into four severity levels, using machine learning techniques')
st.info('Severity Level 1: Represents accidents of less severity. These incidents are minor, causing no serious damage or injury.')
st.info('Severity Level 2: Represents incidents of moderate severity. These accidents are more serious than Level 1 and may cause moderate injuries.')
st.info('Severity Level 3: Represents incidents with a high degree of severity. They are considered serious and can cause significant damage and serious injuries.')
st.info('Severity Level 4: Often indicates incidents with the highest degree of severity. These accidents are very serious and can cause significant damage and serious injuries.')
Start_Lat = st.number_input('"Please, enter the rudimentary latitude (Start_Lat) for the location you want to specify.": ')
Start_Lng=st.number_input('"Please, enter the primitive longitude (Start_Lng) for the location you want to specify.": ')
Des=st.number_input('Enter the distance affected by the accident in miles: ')
Temperature_F = st.number_input("Please enter the temperature in Fahrenheit Temperature(F) at the relevant location.")
Wind_Chill_F = st.number_input("Please enter wind temperature in Fahrenheit (Wind_Chill(F)) if available at the specified location or time:")
Humidity = st.number_input("Please enter humidity (%) in the respective location.(نسبة الرطوبه)")
Pressure = st.number_input("Please enter the value of atmospheric pressure in inches (Pressure(in)) at the specified location:")
Visibility_mi = st.number_input("Please enter visibility in mileage (mi) at the specified location.(مدى الرؤية بوحدة الأميال)")
Wind_Speed_mph = st.number_input("Please enter Wind_Speed(mph) : ")
Precipitation_in = st.number_input("Please enter precipitation in inches (precipitation(in)) in the specified location.")
Amenity = st.selectbox('Is there a safety permit signal on the road? If yes choose 1(اشارة إذن أمان)', ['1', '0'])
Bump = st.selectbox('Are there any obstacles on the way if yes choose 1(هل توجد اي عقبات على الطريق )', ['1', '0'])
Crossing = st.selectbox('Is there pedestrian crossing if yes choose 1', ['1', '0'])
Give_Way = st.selectbox('Is there a sign "Give priority if yes choose 1', ['1', '0'])
Junction = st.selectbox('Is there a crossover. If yes choose 1(هل يوجود تقاطع)', ['1', '0'])
No_Exit = st.selectbox('The presence of the sign No_Exit. If yes choose 1', ['1', '0'])
Railway = st.selectbox('Having a railway line if yes Choose 1(خطه سكه حديد)', ['1', '0'])
Roundabout = st.selectbox('The presence of a ring road if yes choose 1', ['1', '0'])
Station = st.selectbox('The presence of a station. If yes choose 1', ['1', '0'])
Stop = st.selectbox('Presence of the sign "Stop if Yes Select 1', ['1', '0'])
Traffic_Calming = st.selectbox('The presence of calming traffic. If yes choose 1(وجود تهدئة حركة المرور)', ['1', '0'])
Traffic_Signal = st.selectbox('Traffic_Signal? if yes choose 1', ['1', '0'])
Turning_Loop = st.selectbox('Having a ring to rotate if yes choose 1', ['1', '0'])
Day_of_accident = st.slider('Day_of_accident', min_value=1, max_value=30)
Month_of_accident = st.slider('Month_of_accident', min_value=1, max_value=12)
Year_of_accedent = st.number_input('Year_of_accedent: ')
Hour_of_starting = st.slider('Hour_of_starting', min_value=0, max_value=23)
Minutes_of_starting = st.slider('Minutes_of_starting', min_value=0, max_value=60)
Seconds_of_starting = st.slider('Seconds_of_starting', min_value=0, max_value=60)
Hour_of_end_accident = st.slider('Hour_of_end_accident', min_value=0, max_value=23)
Minutes_of_end_accident = st.slider('Minutes_of_end_accident', min_value=0, max_value=60)
Seconds_of_end_accident = st.slider('Seconds_of_end_accident', min_value=0, max_value=60)
if st.button('Classify Accident'):
    input_data = [Start_Lat, Start_Lng, Des, Temperature_F, Wind_Chill_F, Humidity, Pressure, Visibility_mi, Wind_Speed_mph, Precipitation_in, Amenity, Bump, Crossing, Give_Way, Junction, No_Exit, Railway, Roundabout, Station, Stop]
    input_data += [Traffic_Calming, Traffic_Signal, Turning_Loop, Day_of_accident, Month_of_accident, Year_of_accedent, Hour_of_starting, Minutes_of_starting, Seconds_of_starting, Hour_of_end_accident, Minutes_of_end_accident, Seconds_of_end_accident]
    pred = model.predict(scaler.transform([input_data]))
    result = pred[0]
    st.success("Predicted Accident Severity: {}".format(result))
