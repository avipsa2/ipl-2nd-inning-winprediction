import streamlit as st
import pandas as pd
import numpy as np
import pickle

teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('pipe.pkl', 'rb'))
st.title('IPL WIN PREDICTOR')
col1, col2 = st.columns(2)

with col1:
    battingteam = st.selectbox('Select the batting team', sorted(teams))

with col2:
    bowlingteam = st.selectbox('Select the bowling team', sorted(teams))

city = st.selectbox('Select the city where the match is being played', sorted(cities))

target = st.number_input('Target')
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score')

with col4:
    overs = st.number_input('Overs Completed')

with col5:
    wickets = st.number_input('Wickets Fallen')

if st.button('Probability'):
    runs_left = target - score
    wickets_left = 10 - wickets
    balls_left = 120 - (overs * 6)
    currentrunrate = score / overs if overs > 0 else 0
    requiredrunrate = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [battingteam],
        'bowling_team': [bowlingteam],
        'city': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'total_runs_x': [target],
        'current_runrate': [currentrunrate],
        'req_runrate': [requiredrunrate]
    })

    # Verify that all required columns are present
    required_columns = ['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left',
                        'wickets_left', 'total_runs_x', 'current_runrate', 'req_runrate']
    # for col in required_columns:
    #     if col not in input_df.columns:
    #         input_df[col] = 0  # Or any default value

    result = pipe.predict_proba(input_df)
    lossprob = result[0][0]
    winprob = result[0][1]

    st.header(f"{battingteam} - {round(winprob * 100, 2)}%")
    st.header(f"{bowlingteam} - {round(lossprob * 100, 2)}%")
