
import numpy as np
import pickle
import streamlit as st


#loading the model
loaded_model = pickle.load(open("trained_model.sav", 'rb'))


#creating a function for prediction
#  input_data =(15,136,70,32,110,37.1,0.153,43)

def diabetic_prediction(input_data):
  
    input_as_numpy_array = np.asarray(input_data)

    #reshape batata hai ki hum sirf 1 input de rahe hai
    input_data_reshaped =input_as_numpy_array.reshape(1,-1)

    prediction =loaded_model.predict(input_data_reshaped)
    print(prediction)
    if(prediction[0]==1):
      return "Person is Diabetec"
    else:
      return "Person is not Diabetec"
  
    
def main():
    
    #giving a title
    
    st.title("Diabetes Prediction")
    
    #getting input data from user
    Pregnancies = st.text_input("Number of Pregnencies")
    Glucose =  st.text_input("Number of Glucose")
    BloodPressure =  st.text_input("Number of BloodPressure")
    SkinThickness =  st.text_input("Number of SkinThickness")
    Insulin =  st.text_input("Number of Insulin")
    BMI =  st.text_input("Number of BMI")
    DiabetesPedigreeFunction =  st.text_input(" Diabetes Pedigree Function value ")
    Age =  st.text_input("Number of Age")
    
    
    #code for prediction
    diagnosis =''
    
    #creating a button for prediction
    if st.button("Diabetes test Result"):
        diagnosis = diabetic_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    