import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the saved model and encoders
@st.cache_resource
def load_model():
    model = joblib.load('random_forest_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
    return model, label_encoders, target_encoder

# Safe encoding function for unseen categories
def safe_encode(le, value):
    if value not in le.classes_:
        # For unseen categories, use the most common class or handle appropriately
        return le.transform([le.classes_[0]])[0]
    return le.transform([value])[0]

def main():
    st.set_page_config(
        page_title="Resume Screening AI",
        page_icon="📄",
        layout="wide"
    )
    
    # Title and description
    st.title("🤖 Resume Screening AI Assistant")
    st.markdown("""
    This AI tool helps predict recruiter decisions based on candidate profiles.
    Enter the candidate details below to get a prediction.
    """)
    
    # Load model
    try:
        model, label_encoders, target_encoder = load_model()
    except FileNotFoundError:
        st.error("❌ Model files not found. Please run model_training.py first.")
        return
    
    # Create input form
    with st.form("candidate_input_form"):
        st.header("Candidate Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            skills = st.text_input("Skills (e.g., Python, SQL, Machine Learning)", 
                                 placeholder="Enter skills separated by commas")
            experience = st.number_input("Experience (Years)", 
                                       min_value=0, max_value=50, value=2)
            education = st.text_input("Education (e.g., B.Tech, MBA)", 
                                    placeholder="Highest education degree")
            certifications = st.text_input("Certifications (e.g., AWS Certified, PMP)", 
                                         placeholder="Relevant certifications")
        
        with col2:
            job_role = st.text_input("Job Role (e.g., Data Scientist, Software Engineer)", 
                                   placeholder="Desired job role")
            salary = st.number_input("Salary Expectation ($)", 
                                   min_value=0, max_value=500000, value=50000)
            projects = st.number_input("Projects Count", 
                                     min_value=0, max_value=100, value=3)
        
        # Submit button
        submitted = st.form_submit_button("🔍 Predict Recruiter Decision")
    
    # Prediction logic
    if submitted:
        if all([skills, education, certifications, job_role]):
            try:
                # Prepare new data
                new_data = pd.DataFrame({
                    'Skills': [safe_encode(label_encoders['Skills'], skills)],
                    'Experience (Years)': [experience],
                    'Education': [safe_encode(label_encoders['Education'], education)],
                    'Certifications': [safe_encode(label_encoders['Certifications'], certifications)],
                    'Job Role': [safe_encode(label_encoders['Job Role'], job_role)],
                    'Salary Expectation ($)': [salary],
                    'Projects Count': [projects]
                })
                
                # Make prediction
                prediction = model.predict(new_data)[0]
                result = target_encoder.inverse_transform([prediction])[0]
                
                # Display result
                st.header("Prediction Result")
                
                if result == "Hire":
                    st.success(f"✅ **Recruiter Decision: {result}**")
                    st.balloons()
                else:
                    st.error(f"❌ **Recruiter Decision: {result}**")
                
                # Show confidence scores
                st.subheader("Prediction Confidence")
                probabilities = model.predict_proba(new_data)[0]
                
                for class_name, prob in zip(target_encoder.classes_, probabilities):
                    st.write(f"{class_name}: {prob:.2%}")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
        else:
            st.warning("⚠️ Please fill in all the fields.")
    
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This AI model uses Random Forest classification to predict 
        recruiter decisions based on:
        - Skills
        - Experience
        - Education
        - Certifications
        - Job Role
        - Salary Expectations
        - Project Count
        """)
        
        st.header("How to Use")
        st.markdown("""
        1. Fill in all the candidate details
        2. Click 'Predict Recruiter Decision'
        3. View the prediction and confidence scores
        """)
        
        st.header("Model Info")
        st.info("Trained with Random Forest Classifier")

if __name__ == "__main__":
    main()