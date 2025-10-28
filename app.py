import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# ========== LOAD MODEL & ENCODERS ==========
@st.cache_resource
def load_model():
    model = joblib.load('random_forest_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
    return model, label_encoders, target_encoder

# ========== SAFE ENCODE FUNCTION ==========
def safe_encode(le, value):
    if value not in le.classes_:
        return le.transform([le.classes_[0]])[0]
    return le.transform([value])[0]

# ========== MAIN APP ==========
def main():
    st.set_page_config(page_title="Resume Screening AI", page_icon="📄", layout="wide")

    # Title & Description
    st.title("🤖 Resume Screening AI Assistant")
    st.markdown("""
    Use this tool to predict recruiter decisions and visualize key model insights.
    """)

    # Load model
    try:
        model, label_encoders, target_encoder = load_model()
    except FileNotFoundError:
        st.error("❌ Model files not found. Please run model_training.py first.")
        return

    # Sidebar
    with st.sidebar:
        st.header("📊 Model Info")
        st.info("Model: Random Forest Classifier")
        st.markdown("Predicts recruiter decision (Hire / Reject) based on candidate profile.")
        st.markdown("#### **Features Used:**")
        st.markdown("""
        - Skills  
        - Experience  
        - Education  
        - Certifications  
        - Job Role  
        - Salary Expectation  
        - Projects Count  
        """)

    # Input Form
    with st.form("candidate_form"):
        st.header("🧠 Candidate Information")
        col1, col2 = st.columns(2)
        with col1:
            skills = st.text_input("Skills", placeholder="Python, SQL, Machine Learning")
            experience = st.number_input("Experience (Years)", 0, 50, 2)
            education = st.text_input("Education", placeholder="B.Tech, MBA")
            certifications = st.text_input("Certifications", placeholder="AWS, PMP")
        with col2:
            job_role = st.text_input("Job Role", placeholder="Data Scientist")
            salary = st.number_input("Salary Expectation ($)", 0, 500000, 50000)
            projects = st.number_input("Projects Count", 0, 100, 3)

        submitted = st.form_submit_button("🔍 Predict Recruiter Decision")

    # ========== PREDICTION ==========
    if submitted:
        if all([skills, education, certifications, job_role]):
            try:
                new_data = pd.DataFrame({
                    'Skills': [safe_encode(label_encoders['Skills'], skills)],
                    'Experience (Years)': [experience],
                    'Education': [safe_encode(label_encoders['Education'], education)],
                    'Certifications': [safe_encode(label_encoders['Certifications'], certifications)],
                    'Job Role': [safe_encode(label_encoders['Job Role'], job_role)],
                    'Salary Expectation ($)': [salary],
                    'Projects Count': [projects]
                })

                prediction = model.predict(new_data)[0]
                result = target_encoder.inverse_transform([prediction])[0]

                # Result Display
                st.header("🧾 Prediction Result")
                if result == "Hire":
                    st.success(f"✅ **Recruiter Decision: {result}**")
                    st.balloons()
                else:
                    st.error(f"❌ **Recruiter Decision: {result}**")

                # Confidence Scores
                st.subheader("📈 Prediction Confidence")
                probs = model.predict_proba(new_data)[0]
                prob_df = pd.DataFrame({
                    "Decision": target_encoder.classes_,
                    "Probability": probs
                })
                st.bar_chart(prob_df.set_index("Decision"))

            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("⚠️ Please fill in all fields.")

    # ========== VISUALIZATIONS ==========
    st.divider()
    st.header("📊 Model Insights & Visualizations")

    col1, col2 = st.columns(2)

    # 1️⃣ Feature Importance Chart
    with col1:
        st.subheader("🎯 Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': ['Skills', 'Experience (Years)', 'Education', 'Certifications', 
                        'Job Role', 'Salary Expectation ($)', 'Projects Count'],
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=feature_importance, x='Importance', y='Feature', palette="Blues_r", ax=ax)
        st.pyplot(fig)

    # 2️⃣ Target Distribution
    with col2:
        st.subheader("⚖️ Recruiter Decision Distribution")
        try:
            df = pd.read_csv("resumes.csv")
            decision_counts = df['Recruiter Decision'].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(decision_counts, labels=decision_counts.index, autopct='%1.1f%%', startangle=90)
            ax2.axis('equal')
            st.pyplot(fig2)
        except FileNotFoundError:
            st.warning("Dataset not found for distribution visualization.")

    # 3️⃣ Correlation Heatmap (optional)
    st.subheader("🔍 Correlation Heatmap (Feature Relationships)")
    try:
        df_encoded = pd.read_csv("resumes.csv").drop(columns=['Resume_ID', 'Name', 'AI Score (0-100)'])
        df_encoded = df_encoded.apply(lambda x: LabelEncoder().fit_transform(x) if x.dtype == 'object' else x)
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.heatmap(df_encoded.corr(), cmap="YlGnBu", annot=True, fmt=".2f", ax=ax3)
        st.pyplot(fig3)
    except Exception as e:
        st.warning(f"Could not display correlation heatmap: {str(e)}")

    # Footer
    st.divider()
    st.caption("Developed with ❤️ using Streamlit and Random Forest AI")

if __name__ == "__main__":
    main()
