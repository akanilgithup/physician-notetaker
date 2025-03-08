import spacy
import scispacy
from scispacy.linking import EntityLinker
from transformers import pipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List


print("Loading SciSpacy model...")


nlp = spacy.load("en_core_web_md")


linker = EntityLinker(resolve_abbreviations=True, name="umls")
nlp.add_pipe(linker)

# Load Hugging Face models for summarization and sentiment analysis
print("Loading Transformer models...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis")

# FastAPI app
app = FastAPI()

class TranscriptInput(BaseModel):
    transcript: str

def extract_medical_entities(text: str) -> Dict[str, List[str]]:
    """Extracts symptoms, treatments, and diagnoses from text using SciSpacy."""
    doc = nlp(text)
    symptoms, treatments, diagnosis = [], [], []
    
    for ent in doc.ents:
        for umls_ent in ent._.umls_ents:
            category = umls_ent[1]
            if "Sign or Symptom" in category:
                symptoms.append(ent.text)
            elif "Therapeutic or Preventive Procedure" in category:
                treatments.append(ent.text)
            elif "Disease or Syndrome" in category:
                diagnosis.append(ent.text)
    
    return {
        "Symptoms": list(set(symptoms)),
        "Treatments": list(set(treatments)),
        "Diagnosis": list(set(diagnosis))
    }

def summarize_text(text: str) -> str:
    """Summarizes physician-patient conversation."""
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

def analyze_sentiment(text: str) -> str:
    """Analyzes sentiment of the conversation."""
    sentiment = sentiment_analyzer(text)
    return sentiment[0]['label']

@app.post("/analyze")
def analyze_transcript(input_data: TranscriptInput):
    """Processes transcript to extract entities, summarize, and analyze sentiment."""
    transcript = input_data.transcript
    if not transcript:
        raise HTTPException(status_code=400, detail="Empty transcript provided.")
    
    entities = extract_medical_entities(transcript)
    summary = summarize_text(transcript)
    sentiment = analyze_sentiment(transcript)
    
    return {
        "Summary": summary,
        "Sentiment": sentiment,
        "Medical Entities": entities
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
