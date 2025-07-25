import streamlit as st
import PyPDF2
import io
import joblib

vectorizer = joblib.load('tfidf_vectorizer.pkl')
model_nb = joblib.load('job_predictor_model.pkl')

# Title
st.title("📄 PDF Resume Text Extractor")

# File uploader
uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text.strip()

# Preprocess function (make sure you have it)
def preprocess_text(text):
    # Add your actual text cleaning logic here
    return text.lower()

# Handle file upload and prediction
if uploaded_file is not None:
    st.success("PDF uploaded successfully!")
    
    text = extract_text_from_pdf(uploaded_file)
    
    if text:
        sample_input_nb = text
        sample_message_nb = preprocess_text(sample_input_nb)
        sample_message_nb = vectorizer.transform([sample_message_nb])
        sample_pred_nb = model_nb.predict(sample_message_nb)

        st.subheader("Predicted Job Title:")
        st.text_area("Best job title match", sample_pred_nb[0])
    else:
        st.error("No extractable text found in the PDF.")
