# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 17:28:34 2023

@author: Anuruddha
"""
import numpy as np
import pickle 
import  streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler

# loading saved models
diabetes_model = pickle.load(open("diabetes_model.sav",'rb'))
hart_disease_model = pickle.load(open("hart_disease_model.sav",'rb'))
parkinsons_model = pickle.load(open("parkinsons_disease_model.sav",'rb'))

# create side bar or navigation

with st.sidebar:
    
    selected = option_menu('Multiple Diasese Prediction System',
                           ['Diabetes Prediction', 'Hart Disease Prediction', 'Parkinsons Prediction'],
                           
                           icons = ['activity','heart','person'], # icons from bootstrap website
                           
                           default_index = 0) # default_index is defalut selected page

scaler = StandardScaler() #data normalization function



    
# Diadetis prediction page
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction Using ML')
    
    
    # layout of the page
    # make three columns in one line and place three textboxes
    # these order must be our training data set column order
    # these are page text boxes
   
    col1,col2,col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPresuare = st.text_input('Blood Presuare Value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness Value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI Value')
    with col1:
         DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    with col2:
         Age = st.text_input('Age of the Person')
        
        
    # code for Prediction 
    diab_dignosis = '' # final result
    
    # creating a button for prediction
    if st.button('Diabetes Test Result'):
        input_raw_data = np.array([Pregnancies,Glucose,BloodPresuare,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        input_raw_data_reshaped = input_raw_data.reshape(1,-1)  # reshape data
        input_std_data = scaler.fit_transform(input_raw_data_reshaped) # normalize the data
        diab_prediction = diabetes_model.predict(input_std_data)
        
        #result
        if (diab_prediction[0]==0):
            diab_dignosis = 'The person is not Diabetes'
        else: 
            diab_dignosis = 'The person is Diabetes'
            
    st.success(diab_dignosis) # display result
 
    
 
    
 
    
 
    
 
    
# Hart Disease Prediction page
if (selected == 'Hart Disease Prediction'):
    
    # page title
    st.title('Hart Disease Prediction Using ML')
    
    col1,col2,col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex')
    with col3:
        cp = st.text_input('Chest Pain Types')
    with col1:
        tresbps = st.text_input('Resting Blood Presure')
    with col2:
        chol = st.text_input('Serum Cholsetoral in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Suger > 120 mg/dl')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
    with col2:
         thalach = st.text_input('Maximum Hart Rate Achived')
    with col3:
         exang = st.text_input('Exercise Induced Angina')
    with col1:
         oldpeak = st.text_input('ST depression induced by exercises')
    with col2:
         slope = st.text_input('Slop of the peak exercise ST segment')
    with col3:
         ca = st.text_input('Major vessels colored by flourosopy')
    with col1:
         thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')                 
        
    # code for Prediction 
    hart_dignosis = '' # final result
    
    # creating a button for prediction
    if st.button('Hart Disease Test Result'):
        input_raw_data = np.array([age,sex,cp,tresbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
        input_raw_data_reshaped = input_raw_data.reshape(1,-1)  # reshape data
        hart_prediction = hart_disease_model.predict(input_raw_data_reshaped)
        
        #result
        if (hart_prediction[0]==0):
            hart_dignosis = 'The person has not Hart Disease'
        else: 
            hart_dignosis = 'The person has Hart Disease'
            
    st.success(hart_dignosis) # display result
    








# Parkinsons Prediction page
if (selected == 'Parkinsons Prediction'):
    
    # page title
    st.title('Parkinsons Prediction Using ML')
    
    
    col1,col2,col3,col4,col5 = st.columns(5)
    
    with col1:
        fo = st.text_input('MDVP [Fo(Hz)]')
    with col2:
        fhi = st.text_input('MDVP [Fhi(Hz)]')
    with col3:
        flo = st.text_input('MDVP [Flo(Hz)]')
    with col4:
        Jitter_precent = st.text_input('MDVP [Jitter(H%)]')
    with col5:
        Jitter_abs = st.text_input('MDVP [Jitter(Abs)]')
    with col1:
        RAP = st.text_input('MDVP [RAP]')
    with col2:
        PRQ = st.text_input('MDVP [PRQ]')
    with col3:
        DDP = st.text_input('Jitter [DDP]')
    with col4:
        Shimmer = st.text_input('MDVP [Shimmer]')
    with col5:
        Shimmer_dB = st.text_input('MDVP [Shimmer(dB)]')
    with col1:
        APQ3 = st.text_input('Shimmer [APQ3]')
    with col2:
        APQ5 = st.text_input('Shimmer [APQ5]')
    with col3:
        APQ = st.text_input('MDVP [APQ]')
    with col4:
        DDA = st.text_input('Shimmer [DDA]') 
    with col5:
        NHR = st.text_input('NHR')
    with col1:
        HNR = st.text_input('HNR')
    with col2:
        RPDE = st.text_input('RPDE')
    with col3:
        DFA = st.text_input('DFA')
    with col4:
        spread1 = st.text_input('spread1')
    with col5:
        spread2 = st.text_input('spread2') 
    with col1:
        D2 = st.text_input('D2')
    with col2:
        PPE = st.text_input('PPE')                  
       
    # code for Prediction 
    park_dignosis = '' # final result
    
    # creating a button for prediction
    if st.button('Parkinsons Test Result'):
        input_raw_data = np.array([fo,fhi,flo,Jitter_precent,Jitter_abs,RAP,PRQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE])
        input_raw_data_reshaped = input_raw_data.reshape(1,-1)  # reshape data
        input_std_data = scaler.fit_transform(input_raw_data_reshaped) # normalize the data
        park_prediction = parkinsons_model.predict(input_std_data)
        
        #result
        if (park_prediction[0]==0):
            park_dignosis = 'The person has not Parkinsons'
        else: 
            park_dignosis = 'The person has Parkinsons'
            
    st.success(park_dignosis) # display result


    
    
