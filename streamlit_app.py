import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = joblib.load('Social_Media_Addiction_Classifier2.pkl')

def map_addicted_score_to_grade(score):
    bins = [1.0, 4.5, 7.8, 10.0]
    labels = ['Low', 'Medium', 'High']
    
    score = max(min(score, 10.0), 1.0)
    return pd.cut([score], bins=bins, labels=labels, include_lowest=True)[0]

#-----------------------------------------------------------------------------#

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Machine Learning", "Bergen Scale", "ReadMe"])

if page == "Machine Learning":
    st.title("Predict Social Media Addiction Using Machine Learning")
    st.markdown("""
    <div style="font-size:16px;">
      <p>
        This application predicts your level of social media addiction based on your responses to a set of questions.
        The prediction is made using a machine learning model trained on real-world data.
      </p>
      <p>
        <b>Note:</b> Scoring is based on <a href="https://hub.salford.ac.uk/psytech/2021/08/10/bergen-social-media-addiction-scale/" target="_blank">The Bergen Social Media Addiction Scale</a>, which is available on a separate page for further self-assessment.
      </p>
      <p>
        <b>Curious about the data used to train the model? </b>
        <a href="https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships" target="_blank">
           [ View the dataset on Kaggle ] 
        </a>
      </p>
      <hr>
      <h4>
        <b>Fill out the form to get a risk prediction</b> | <span style="color:green;">Low</span> | <span style="color:orange;">Medium</span> | <span style="color:red;">High</span> |
      </h4>
    </div>
    """, unsafe_allow_html=True)

    #st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)
    #st.markdown("##### Basic Information:")
    age = st.number_input('What is your age?', min_value=8, max_value=100, value=18)
    #gender = st.radio('Gender', options=['Male', 'Female'], index=0, horizontal=True)

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
            #'Age': [age], #TODO: Map to sleep recommendation by age
            'Avg_Daily_Usage_Hours': [avg_daily_usage],
            'Sleep_Hours_Per_Night': [sleep_hours],
            'Mental_Health_Score': [mental_health],
            'Conflicts_Over_Social_Media': [conflicts],
            #'Gender_Female': [1 if gender == 'Female' else 0], # Drop?
            #'Gender_Male': [1 if gender == 'Male' else 0], # Drop?
            'Affects_Academic_Performance_No': [1 if affects_academic == 'No' else 0],
            'Affects_Academic_Performance_Yes': [1 if affects_academic == 'Yes' else 0],
            'Relationship_Status_Complicated': [1 if relationship_status == 'Complicated' else 0],
            'Relationship_Status_In Relationship': [1 if relationship_status == 'In Relationship' else 0],
            'Relationship_Status_Single': [1 if relationship_status == 'Single' else 0],
        }
        X_input = pd.DataFrame(input_dict)
        pred = model.predict(X_input)[0]
        label_map = {1: 'Low', 2: 'Medium', 3: 'High'}
        submitted = st.form_submit_button("Predict Addiction Grade")
        if submitted:
            ml_grade = label_map.get(pred, pred)
            color_map = {"Low": "#6a9c6a", "Medium": "#c2b546", "High": "#c91818"}
            color = color_map.get(str(ml_grade), "#f0f0f0")
            st.markdown(f"""
                <div style='background-color: {color}; padding: 1.2em; margin-bottom: 15px; border-radius: 10px; text-align: center; font-size: 1.3em; font-weight: bold;'>
                    {ml_grade}
                </div>
            """, unsafe_allow_html=True)

            # --- Visualization Section ---
            # Load your dataset (adjust path as needed)
            df = pd.read_csv("Students_Social_Media_Addiction.csv")
            # Calculate means
            mean_usage = df['Avg_Daily_Usage_Hours'].mean()
            mean_sleep = df['Sleep_Hours_Per_Night'].mean()
            mean_mental = df['Mental_Health_Score'].mean()
            mean_conflicts = df['Conflicts_Over_Social_Media'].mean()

            st.subheader("How You Compare to the Dataset:")
            # Horizontal grouped bar plot for user vs mean
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            metrics = ['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score', 'Conflicts_Over_Social_Media']
            user_vals = [avg_daily_usage, sleep_hours, mental_health, conflicts]
            mean_vals = [mean_usage, mean_sleep, mean_mental, mean_conflicts]
            bar_width = 0.35
            y_pos = np.arange(len(metrics))
            ax2.barh(y_pos - bar_width/2, user_vals, bar_width, label='You', color='#4e79a7')
            ax2.barh(y_pos + bar_width/2, mean_vals, bar_width, label='Dataset', alpha=0.6, color="#e05555")
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(metrics)
            ax2.invert_yaxis()
            ax2.set_xlabel('Value')
            ax2.set_title('Your Input vs Dataset Mean')
            ax2.legend()
            st.pyplot(fig2)

            # Sleep Hours Per Night - detailed plot
            st.subheader("Sleep Hours Per Night vs. Dataset & Recommendations")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(
                df[(df['Age'] >= 18) & (df['Age'] <= 25)]['Sleep_Hours_Per_Night'],
                color='lightgrey', kde=True, alpha=0.4, ax=ax
            )
            ax.axvline(7, color='green', linestyle='dashed')
            ax.axvline(9, color='green', linestyle='dashed')
            ax.axvspan(7, 9, color='green', alpha=0.15, label='Recommended Range (7-9h)')
            ax.axvline(mean_sleep, color='red', linestyle='dashed', label=f"Dataset: {mean_sleep:.2f}")
            ax.axvline(sleep_hours, color='blue', linestyle='dashed', linewidth=2, label=f"You: {sleep_hours}")
            ax.set_xlabel('Sleep Hours Per Night')
            ax.set_title('Sleep Hours Distribution & NSF Recommendation')
            ax.legend()
            st.pyplot(fig)
            st.markdown("""
            <div style="font-size:15px;">
              <b>Sleep Analytics:</b><br>
              To put The National Sleep Foundation’s (NSF’s) mission is to improve health and well-being through sleep health education and advocacy. The NSF provides the public with the most up-to-date, scientifically rigorous sleep health recommendations. 
              <br>
              <a href="https://www.sleephealthjournal.org/article/s2352-7218(15)00015-7/fulltext" target="_blank">
                Read more about NSF sleep recommendations
              </a>
            </div>
            """, unsafe_allow_html=True)

elif page == "Bergen Scale":
    st.header("Bergen Social Media Addiction Scale")
    st.markdown("##### Here are six statements to consider. For each, answer:")
    st.markdown("""
    | <span style="color:green;">(1) very rarely</span> 
    | <span style="color:#FFFFE0;">(2) rarely</span> 
    | <span style="color:yellow;">(3) sometimes</span> 
    | <span style="color:orange;">(4) often</span> 
    | <span style="color:red;">(5) very often</span> |
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    bergen_q1 = st.slider("You spend a lot of time thinking about social media or planning how to use it.", 1, 5, 1)
    bergen_q2 = st.slider("You feel an urge to use social media more and more.", 1, 5, 1)
    bergen_q3 = st.slider("You use social media in order to forget about personal problems.", 1, 5, 1)
    bergen_q4 = st.slider("You have tried to cut down on the use of social media without success.", 1, 5, 1)
    bergen_q5 = st.slider("You become restless or troubled if you are prohibited from using social media.", 1, 5, 1)
    bergen_q6 = st.slider("You use social media so much that it has had a negative impact on your job/studies.", 1, 5, 1)

    bergen_responses = [bergen_q1, bergen_q2, bergen_q3, bergen_q4, bergen_q5, bergen_q6]
    addicted_score = sum(bergen_responses) / 3
    bergen_score = sum(bergen_responses)
    if st.button("Predict Bergen Grade"):
        addicted_grade = map_addicted_score_to_grade(addicted_score)
        st.write(f"Your Bergen Social Media Addiction Scale score: {bergen_score}")
        st.write(f"Addiction grade (Bergen scale): {addicted_grade if pd.notna(addicted_grade) else 'Out of range'}")

elif page == "ReadMe":
    st.title("ReadMe")
    st.markdown("""
    # Social Media Addiction App
    
    - **Main:** Predict addiction grade using the ML model (excluding Bergen scale score).
    - **Bergen Scale:** Self-assessment using the Bergen Social Media Addiction Scale.
    - **Readme:** Project info and instructions.
    
    *Add your own documentation here!*
    """)