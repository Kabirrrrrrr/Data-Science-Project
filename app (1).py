# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 08:08:23 2023

@author: AKSHAY
"""
import pickle
import streamlit as st

st.title('Fake Bills Detection')

load = open('fake_bills_pickle.pkl','rb')
model = pickle.load(load)


def predict(b,c,d,e,f,g):
    prediction = model.predict([[b,c,d,e,f,g]])
    return prediction

def main():
    
    st.markdown('This is a very simple webapp for prediction weather the bill is fake or genuine :chart:')
    import pandas as pd
    entries=[[
    (st.number_input('Diagonal', min_value= 0.00 , max_value=1000.00)),
    (st.number_input('Height Left', min_value= 0.00 , max_value=1000.00)),
    (st.number_input('Height Right', min_value= 0.00 , max_value=1000.00)),
    (st.number_input('Margin Up', min_value= 0.00 , max_value=1000.00)),
    (st.number_input('Length', min_value= 0.00 , max_value=1000.00)),
    (st.number_input('Margin Low', min_value= 0.00 , max_value=1000.00))
    ]]
    df = pd.DataFrame(entries,
                      columns=['b','c','d','e','f','g'])
    if st.button('Predict'):
        result = model.predict(df)
        st.success('The Bill is Genuine : {}'.format(result))   
if __name__ == '__main__':
    main()