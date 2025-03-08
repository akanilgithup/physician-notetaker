import streamlit as st
import spacy
import scispacy
import torch
from transformers import pipeline

st.title("Physician Notetaker")

nlp = spacy.load("en_core_web_md")
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
