"""
Deployment inference script.
Load the saved pipeline (preprocessing + tuned model bundled together)
and predict course-completion probability for new learner records.

Usage:
    python predict_new_learner.py
"""
import pandas as pd
import joblib

MODEL_PATH = "final_model_pipeline.pkl"

def load_model(path=MODEL_PATH):
    return joblib.load(path)

def predict(model, records: pd.DataFrame):
    """
    records: DataFrame with the same raw columns used in training
             (Age, Gender, Education_Level, Employment_Status, Course_Category,
              Device_Used, Weekly_Study_Hours, Videos_Watched_Percentage,
              Quiz_Avg_Score, Assignments_Submitted, Forum_Participation_Count,
              Days_Since_Enrollment, Previous_Courses_Completed,
              Discount_Availed, Reminder_Emails_Opened)
    Missing values and out-of-range values are fine -- the pipeline's
    imputers handle them the same way they were handled during training.
    """
    pred = model.predict(records)                 # 1 = Completed, 0 = Not completed
    proba = model.predict_proba(records)[:, 1]     # probability of completion
    out = records.copy()
    out["Predicted_Completion"] = pred
    out["Predicted_Completion_Label"] = out["Predicted_Completion"].map({1: "Yes", 0: "No"})
    out["Completion_Probability"] = proba.round(4)
    return out

if __name__ == "__main__":
    model = load_model()

    # Example: a couple of new (unseen) learners
    new_learners = pd.DataFrame([
        {
            "Age": 24, "Gender": "Female", "Education_Level": "Bachelor's",
            "Employment_Status": "Student", "Course_Category": "Data Science",
            "Device_Used": "Laptop", "Weekly_Study_Hours": 9.5,
            "Videos_Watched_Percentage": 82.0, "Quiz_Avg_Score": 78.0,
            "Assignments_Submitted": 8, "Forum_Participation_Count": 3,
            "Days_Since_Enrollment": 60, "Previous_Courses_Completed": 2,
            "Discount_Availed": "No", "Reminder_Emails_Opened": 6,
        },
        {
            "Age": 41, "Gender": "Male", "Education_Level": "High School",
            "Employment_Status": "Employed", "Course_Category": "Finance",
            "Device_Used": "Mobile", "Weekly_Study_Hours": 1.2,
            "Videos_Watched_Percentage": 12.0, "Quiz_Avg_Score": 30.0,
            "Assignments_Submitted": 1, "Forum_Participation_Count": 0,
            "Days_Since_Enrollment": 200, "Previous_Courses_Completed": 0,
            "Discount_Availed": "Yes", "Reminder_Emails_Opened": 0,
        },
    ])

    result = predict(model, new_learners)
    print(result[["Predicted_Completion_Label", "Completion_Probability"]])
