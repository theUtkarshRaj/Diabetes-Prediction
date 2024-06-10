import numpy as np
import pickle
import streamlit as st


# Loading the model
loaded_model = pickle.load(open("trained_model.sav", 'rb'))


# Creating a function for prediction
def diabetic_prediction(input_data):
    input_as_numpy_array = np.asarray(input_data, dtype=float)

    # Reshape the input as we are passing only one input
    input_data_reshaped = input_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 1:
        return "Person is Diabetic"
    else:
        return "Person is not Diabetic"
  
    
def main():
    # Giving a title
    st.title("Diabetes Prediction")
    
    # Getting input data from user
    Pregnancies = st.text_input("Number of Pregnancies", value='0')
    Glucose = st.text_input("Number of Glucose", value='0')
    BloodPressure = st.text_input("Number of BloodPressure", value='0')
    SkinThickness = st.text_input("Number of SkinThickness", value='0')
    Insulin = st.text_input("Number of Insulin", value='0')
    BMI = st.text_input("Number of BMI", value='0')
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value", value='0')
    Age = st.text_input("Number of Age", value='0')
    
    # Code for prediction
    diagnosis = ''
    
    # Creating a button for prediction
    if st.button("Diabetes test Result"):
        try:
            input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            input_data = [float(x) for x in input_data]  # Convert input to float
            diagnosis = diabetic_prediction(input_data)
        except ValueError:
            st.error("Please enter valid numerical values for all fields.")
    
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
