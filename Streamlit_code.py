#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib 
import numpy as np

model = joblib.load('ModelDep.pkl')

def main():
    st.title('Machine Learning Model Deployment')
    
    #Add user input components for 5 features
    sepal_length = st.slider('sepal_length', min_value = 0.0,max_value =10.0,value =0.1)
    sepal_width = st.slider('sepal_width', min_value = 0.0,max_value =10.0,value =0.1)
    petal_length = st.slider('patal_length', min_value = 0.0,max_value =10.0,value =0.1)
    petal_width = st.slider('patal_width', min_value = 0.0,max_value =10.0,value =0.1)
    
    if st.button('Make Prediction'):
        features = [sepal_length,sepal_width,petal_length,petal_width]
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')
        
    def make_prediction(features):
        #use the loaded model to make predictions
        input_array = np.array(features).reshape(1,-1)
        prediction = model.predict(input_array)
        return prediction[0]
    
    if _name_ == '_main_':
        main()


# In[ ]:




