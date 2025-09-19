import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('Social_Media_Addiction_Classifier.pkl')

def map_addicted_score_to_grade(score):
    bins = [1, 4.5, 7.8, 10]
    labels = ['Low', 'Medium', 'High']
    return pd.cut([score], bins=bins, labels=labels, include_lowest=True)[0]

#-----------------------------------------------------------------------------#

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("You are here:", ["Machine Learning Prediction", "Bergen Scale", "ReadMe"])

if page == "Machine Learning Prediction":
    st.title("Predict Social Media Addiction Using Machine Learning")
    st.markdown('''
    This application predicts your level of social media addiction based on your answers to a set of questions. Fill out the form below to get a risk prediction: | <span style="color:green">Low</span> | <span style="color:orange">Medium</span> | <span style="color:red">High</span> |

    <br>If you are curious, and want to read more the data used to train the model: [Click Here](https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships)
    ''', unsafe_allow_html=True)

    st.markdown("---")

    st.subheader('Basic Information:')
    age = st.number_input('What is your age?', min_value=8, max_value=100, value=18)
    gender = st.radio('Gender', options=['Male', 'Female'], index=0, horizontal=True)

    st.subheader("Take a moment to reflect on your habits:")
    avg_daily_usage = st.slider('Average daily usage of social media platforms (hours)', min_value=0.0, max_value=12.0, value=0.0, step=0.5)
    sleep_hours = st.slider('Average sleep hours per night', min_value=0.0, max_value=12.0, value=0.0, step=0.5)
    mental_health = st.slider('Mental health score (1 = Poor, 10 = Excellent)', 1, 10, 0, step=1)
    conflicts = st.slider('Average number of conflicts over social media', min_value=0, max_value=10, value=0, step=1)
    affects_academic = st.radio('Does your social media usage affect your academic results?', options=['Yes', 'No'], index=1, horizontal=True)
    relationship_status = st.radio('What is your relationship status?', options=['Single', 'In Relationship', 'Complicated'], index=0, horizontal=True)


    # Input_dict for model to predict:
    with st.form("ml_form"):
        # Prepare input as DataFrame (order must match training data)
        input_dict = {
            'Age': [age], #TODO: Map to sleep recommendation by age
            'Avg_Daily_Usage_Hours': [avg_daily_usage],
            'Sleep_Hours_Per_Night': [sleep_hours],
            'Mental_Health_Score': [mental_health],
            'Conflicts_Over_Social_Media': [conflicts],
            'Gender_Female': [1 if gender == 'Female' else 0], # Drop?
            'Gender_Male': [1 if gender == 'Male' else 0], # Drop?
            'Affects_Academic_Performance_No': [1 if affects_academic == 'No' else 0],
            'Affects_Academic_Performance_Yes': [1 if affects_academic == 'Yes' else 0],
            'Relationship_Status_Complicated': [1 if relationship_status == 'Complicated' else 0],
            'Relationship_Status_In Relationship': [1 if relationship_status == 'In Relationship' else 0],
            'Relationship_Status_Single': [1 if relationship_status == 'Single' else 0],
        }
        X_input = pd.DataFrame(input_dict)
        pred = model.predict(X_input)[0]
        label_map = {1: 'Low', 2: 'Medium', 3: 'High'}
        submitted = st.form_submit_button("Predict ML Grade")
        if submitted:
            st.success(f'Your predicted addiction grade is: {label_map.get(pred, pred)}')

    # Bergen Social Media Addiction Scale in a collapsible area
    with st.expander("Bergen Social Media Addiction Scale Questionnaire"):
        st.subheader('Bergen Social Media Addiction Scale:')
        st.markdown('''
        <span style="color:lightgreen">
        Here are six statements to consider. For each, indicate how often it applies to you:
        <br><strong>(1) Very rarely, (2) Rarely, (3) Sometimes, (4) Often, (5) Very often<strong>
        </span>
        ''', unsafe_allow_html=True)

        st.markdown('---')

        bergen_qs = [
            "You spend a lot of time thinking about social media or planning how to use it.",
            "You feel an urge to use social media more and more.",
            "You use social media in order to forget about personal problems.",
            "You have tried to cut down on the use of social media without success.",
            "You become restless or troubled if you are prohibited from using social media.",
            "You use social media so much that it has had a negative impact on your job/studies."
        ]
        bergen_answers = []
        for i, q in enumerate(bergen_qs, 1):
          ans = st.slider(f'{i}. {q}', min_value=1, max_value=5, value=1, key=f'bergen_{i}')
          bergen_answers.append(ans)

        # Calculate Bergen score and scale
        addicted_score = sum(bergen_answers) / 3

        # After calculating addicted_score from the questionnaire:
        addicted_grade = map_addicted_score_to_grade(addicted_score)
        st.write(f"Your Bergen Social Media Addiction Scale score: {addicted_score}")
        st.write(f"Addiction grade (Bergen scale): {addicted_grade}")

        if st.button("Predict Bergen Grade"):
            addicted_grade = map_addicted_score_to_grade(addicted_score)
            st.write(f"Your Bergen Social Media Addiction Scale score: {addicted_score}")
            st.write(f"Addiction grade (Bergen scale): {addicted_grade}")

elif page == "Bergen Scale":
    st.title("Bergen Social Media Addiction Scale")
    st.write("Answer the Bergen scale questions below:")
    bergen_q1 = st.slider("How often do you spend a lot of time thinking about social media or planning how to use it?", 1, 5, 1)
    bergen_q2 = st.slider("How often do you feel an urge to use social media more and more?", 1, 5, 1)
    bergen_q3 = st.slider("How often do you use social media in order to forget about personal problems?", 1, 5, 1)
    bergen_q4 = st.slider("How often have you tried to cut down on the use of social media without success?", 1, 5, 1)
    bergen_q5 = st.slider("How often do you become restless or troubled if you are prohibited from using social media?", 1, 5, 1)
    bergen_q6 = st.slider("How often do you use social media so much that it has had a negative impact on your job/studies?", 1, 5, 1)

    bergen_responses = [bergen_q1, bergen_q2, bergen_q3, bergen_q4, bergen_q5, bergen_q6]
    addicted_score = sum(bergen_responses)
    if st.button("Predict Bergen Grade"):
        addicted_grade = map_addicted_score_to_grade(addicted_score)
        st.write(f"Your Bergen Social Media Addiction Scale score: {addicted_score}")
        st.write(f"Addiction grade (Bergen scale): {addicted_grade}")

elif page == "ReadMe":
    st.title("ReadMe")
    st.markdown("""
    # Social Media Addiction App
    
    - **Main:** Predict addiction grade using the ML model (excluding Bergen scale score).
    - **Bergen Scale:** Self-assessment using the Bergen Social Media Addiction Scale.
    - **Readme:** Project info and instructions.
    
    *Add your own documentation here!*
    """)