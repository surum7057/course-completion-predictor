import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Course Completion Predictor", page_icon="🎓", layout="centered")

MODEL_PATH = "final_model_pipeline.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🎓 Online Course Completion Predictor")
st.write(
    "Predicts whether a learner will **complete** their course, based on engagement "
    "and profile data. Fill in the fields below and click **Predict**."
)

with st.form("learner_form"):
    st.subheader("Learner profile")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=15, max_value=75, value=28)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
        employment = st.selectbox("Employment Status", ["Student", "Unemployed", "Freelancer", "Employed"])
    with col2:
        course_category = st.selectbox(
            "Course Category", ["Programming", "Finance", "Digital Marketing", "Data Science", "Graphic Design"]
        )
        device = st.selectbox("Device Used", ["Laptop", "Mobile", "Tablet"])
        discount = st.selectbox("Discount Availed", ["No", "Yes"])

    st.subheader("Engagement metrics")
    col3, col4 = st.columns(2)
    with col3:
        weekly_hours = st.slider("Weekly Study Hours", 0.0, 40.0, 6.0, 0.5)
        videos_pct = st.slider("Videos Watched (%)", 0.0, 100.0, 50.0, 1.0)
        quiz_score = st.slider("Quiz Average Score", 0.0, 100.0, 60.0, 1.0)
        assignments = st.slider("Assignments Submitted", 0, 10, 5)
    with col4:
        forum_count = st.number_input("Forum Participation Count", min_value=0, max_value=50, value=2)
        days_enrolled = st.number_input("Days Since Enrollment", min_value=0, max_value=1000, value=45)
        prev_courses = st.number_input("Previous Courses Completed", min_value=0, max_value=20, value=1)
        reminders = st.number_input("Reminder Emails Opened", min_value=0, max_value=30, value=3)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Education_Level": education,
        "Employment_Status": employment,
        "Course_Category": course_category,
        "Device_Used": device,
        "Weekly_Study_Hours": weekly_hours,
        "Videos_Watched_Percentage": videos_pct,
        "Quiz_Avg_Score": quiz_score,
        "Assignments_Submitted": assignments,
        "Forum_Participation_Count": forum_count,
        "Days_Since_Enrollment": days_enrolled,
        "Previous_Courses_Completed": prev_courses,
        "Discount_Availed": discount,
        "Reminder_Emails_Opened": reminders,
    }])

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0, 1]

    st.divider()
    if pred == 1:
        st.success(f"✅ Likely to **complete** the course — probability: {proba:.1%}")
    else:
        st.error(f"⚠️ At risk of **not completing** the course — completion probability: {proba:.1%}")

    st.progress(float(proba))
    st.caption("Probability shown is the model's estimated likelihood of course completion.")

st.divider()
st.caption("Model: tuned Hist Gradient Boosting Classifier · Test accuracy ≈ 95.9% · ROC-AUC ≈ 0.992")
