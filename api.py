from fastapi import FastAPI, Depends, HTTPException, Header
from typing import Optional
from pydantic import BaseModel
import pandas as pd
import asyncio
import concurrent.futures
import nltk
import os
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import re

API_KEY = "2db61b79ba7339b6a4f44cd740fdd4c6b27ddade94b8fe504495e21c7b07d761"

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key is None:
        raise HTTPException(status_code=403, detail="Missing API Key.")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key

app = FastAPI(title="Secure BiztelAI API")

FILE_PATH = os.path.abspath("cleaned_dataset_OOP.csv")

def load_data_async():
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"CSV file not found: {FILE_PATH}")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(pd.read_csv, FILE_PATH)
        return future.result()

# Initialize BART model
model_name = "sshleifer/distilbart-cnn-6-6"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

class ChatRequest(BaseModel):
    chat_text: str

@app.post("/summarize_chat")
def summarize_chat(request: ChatRequest):
    text = request.chat_text
    prompt = "Summarize the following text accurately: " + request.chat_text
    inputs = tokenizer(text, return_tensors="pt",truncation=True, max_length=512)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True) # Placeholder, replace with LLM output
    return {"summary": summary_text}


class TextInput(BaseModel):
    text: str

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

@app.post("/sentiment_analysis")
async def analyze_sentiment(data: TextInput, api_key: Optional[str] = Depends(verify_api_key)):
    blob = TextBlob(data.text)
    sentiment_score = blob.sentiment.polarity
    sentiment = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"
    return {"original_text": data.text, "sentiment": sentiment, "score": sentiment_score}


@app.post("/extract_links")
async def extract_article_links(api_key: Optional[str] = Depends(verify_api_key)):
    df = await asyncio.to_thread(load_data_async)
    links = df["article_url"].dropna().unique().tolist()
    return {"article_links": links}

@app.get("/eda_summary")
def eda_summary():
    return {
        "total_messages": 1500,
        "unique_agents": 20,
        "missing_values": {
            "column_name": 5
        }
    }


@app.post("/generate_summary")
async def generate_summary(data: TextInput, api_key: Optional[str] = Depends(verify_api_key)):
    inputs = tokenizer(data.text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {"original_text": data.text, "summary": summary}

@app.get("/chat_transcripts")
def chat_transcripts():
    return {
        "data": [
            {"agent": "John", "message": "Hello! How can I help you?", "sentiment": "neutral"}
        ]
    }


@app.get("/")
def home():
    return {"message": "Welcome to the Secure BiztelAI API with Sentiment Analysis & Summarization!"}
