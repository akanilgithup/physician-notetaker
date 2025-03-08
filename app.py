import streamlit as st
import spacy
import scispacy
import torch
from transformers import pipeline
import subprocess

st.title("Physician Notetaker")

model_name = "en_core_web_md"
try:
    nlp = spacy.load(model_name)
except OSError:
    st.warning(f"{model_name} not found. Downloading now...")
    subprocess.run(["python", "-m", "spacy", "download", model_name])
    nlp = spacy.load(model_name)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

uploaded_file = st.file_uploader("Upload a transcript file", type=["txt"])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    st.subheader("Original Text")
    st.write(text)
    
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    
    st.subheader("Extracted Medical Entities")
    for entity, label in entities:
        st.write(f"{entity} ({label})")
    
    st.subheader("Generated Summary")
    st.write(summary)
