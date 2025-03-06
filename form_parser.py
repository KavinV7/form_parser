import streamlit as st
import pdfplumber
import pytesseract
import cv2
import numpy as np
from PIL import Image
import json
from langdetect import detect
from deep_translator import GoogleTranslator
 
st.title("📄 Structured Form Processing with Auto-Mapping & Translation")
 
# translator = Translator()
 
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()
 
def extract_text_from_image(image_file):
    image = Image.open(image_file)
    image = np.array(image)  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()
 
def detect_language_and_translate(text, target_lang="en"):
    detected_lang = detect(text)
    translated_text = GoogleTranslator(source=detected_lang, target=target_lang).translate(text) if detected_lang != target_lang else text
    return detected_lang, translated_text
 
uploaded_file = st.file_uploader("Upload a PDF or Document", type=["pdf","docx"])
 
if uploaded_file:
    file_type = uploaded_file.type
    extracted_text = ""
 
    if "pdf" in file_type:
        extracted_text = extract_text_from_pdf(uploaded_file)
    else:
        extracted_text = extract_text_from_image(uploaded_file)
 
    detected_lang, translated_text = detect_language_and_translate(extracted_text)
 
    st.write("### Extracted Text:")
    st.text_area("", extracted_text, height=200)
 
    st.write(f"**Detected Language:** `{detected_lang}`")
    
    if detected_lang != "en":
        st.write("### Translated Text (English):")
        st.text_area("", translated_text, height=200)
 
    json_output = json.dumps({"original_text": extracted_text, "language": detected_lang, "translated_text": translated_text}, indent=4)
    from dotenv import load_dotenv
    import os
    load_dotenv()
    from langchain_openai import ChatOpenAI
    client = ChatOpenAI(model="gpt-4o-mini")
    
    prompt = f"""
    Convert the following extracted text into a structured JSON format:
    
    {extracted_text}
    
    Ensure the JSON format is well-structured and meaningful.
    """
    
    
    # response = client.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[
    #         {"role": "system", "content": "You are an AI that converts unstructured text into structured JSON."},
    #         {"role": "user", "content": prompt}
    #     ]
    # )
    structured_json = client.invoke(prompt)
    
    
    st.write("### Structured JSON Output:")
    st.json(structured_json)
