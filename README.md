# 🎓 Course Completion Predictor

A machine learning web app that predicts whether an online learner will **complete** or **drop off** from a course, based on their engagement and profile data — built for SkillBridge's early-warning retention use case.

**Live app:** _add your Streamlit Cloud URL here after deploying_

---

## Problem Statement

Many learners on SkillBridge's online learning platform drop off before completing their courses, and the company only finds out after it's too late to help them. This project predicts, in advance, whether a learner will complete their course (**Yes/No**) using engagement signals (study hours, quiz scores, video-watch %, assignments, forum activity) and profile data (age, education, employment status, etc.), so the platform can trigger a retention action early.

## Model

- **Algorithm:** Hist Gradient Boosting Classifier (scikit-learn), tuned via `RandomizedSearchCV`
- **Test Accuracy:** ~95.9%
- **Test ROC-AUC:** ~0.992
- Compared against Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest, and Linear SVM — Hist Gradient Boosting was selected for its strong accuracy at a fraction of the model size (~1.8 MB vs ~148 MB for Random Forest), making it far better suited for deployment.
- Preprocessing (imputation, one-hot/ordinal encoding, scaling) and the trained model are bundled into a single `final_model_pipeline.pkl`, so the app only needs to call `.predict()` on raw input.

Full data cleaning, EDA, cross-validation, and hyperparameter tuning process is documented in `Project_1_Final.ipynb`.

## Repository Structure

```
.
├── app.py                        # Streamlit web app
├── final_model_pipeline.pkl      # Trained preprocessing + model pipeline
├── requirements.txt              # Python dependencies
├── Project_1_Final.ipynb         # Full ML workflow: cleaning, EDA, modeling, tuning
├── predict_new_learner.py        # Example script for offline/batch predictions
└── README.md
```

## Run Locally

```bash
git clone https://github.com/<your-username>/course-completion-predictor.git
cd course-completion-predictor
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repository and branch, set main file to `app.py`.
4. Click **Deploy**.

## Input Features

| Feature | Description |
|---|---|
| Age | Learner's age |
| Gender | Female / Male / Other |
| Education_Level | High School / Bachelor's / Master's / PhD |
| Employment_Status | Student / Unemployed / Freelancer / Employed |
| Course_Category | Programming / Finance / Digital Marketing / Data Science / Graphic Design |
| Device_Used | Laptop / Mobile / Tablet |
| Weekly_Study_Hours | Avg. hours studied per week |
| Videos_Watched_Percentage | % of course videos watched |
| Quiz_Avg_Score | Average quiz score (0–100) |
| Assignments_Submitted | Number of assignments submitted (0–10) |
| Forum_Participation_Count | Number of forum posts |
| Days_Since_Enrollment | Days since the learner enrolled |
| Previous_Courses_Completed | Number of prior courses completed |
| Discount_Availed | Whether a discount was used at signup |
| Reminder_Emails_Opened | Number of reminder emails opened |

## Output

- **Predicted_Completion**: 1 (Completed) / 0 (Not Completed)
- **Completion_Probability**: model's confidence (0–1)

## Tech Stack

- Python, pandas, scikit-learn, joblib
- Streamlit (web app / UI)

## Author

Suraj P More
