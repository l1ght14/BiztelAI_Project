from fastapi import FastAPI, Depends, HTTPException, Header
from typing import Optional
from pydantic import BaseModel
import pandas as pd
import asyncio
import concurrent.futures
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# API Key for authentication
API_KEY = "my_secure_api_key"  # Change this to a secure key

# Function to validate API key (optional for testing)
def verify_api_key(x_api_key: Optional[str] = Header(None)):
    print("Received x-api-key:", x_api_key)  # Debugging line
    if x_api_key is None:
        raise HTTPException(status_code=403, detail="❌ Missing API Key. Please provide `x-api-key` in headers.")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="❌ Invalid API Key")


# Initialize FastAPI app
app = FastAPI(title="Secure BiztelAI API")

# 🚀 Step 1: Get Full Path to CSV File
FILE_PATH = os.path.abspath("cleaned_dataset_OOP.csv")

# 🚀 Step 2: Load Data Asynchronously
def load_data_async():
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"❌ CSV file not found: {FILE_PATH}")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(pd.read_csv, FILE_PATH)
        return future.result()

# 🚀 Step 3: Secure Summary API
@app.get("/summary")
async def get_summary(api_key: Optional[str] = Depends(verify_api_key)):
    """Returns dataset summary securely."""
    df = await asyncio.to_thread(load_data_async)
    summary = df['agent'].value_counts().to_dict()
    return {"message_count_per_agent": summary}

# 🚀 Step 4: Secure Preprocessing API
class TextInput(BaseModel):
    text: str

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

@app.post("/preprocess")
async def preprocess_input(data: TextInput, api_key: Optional[str] = Depends(verify_api_key)):
    """Preprocess text securely."""
    processed_text = preprocess_text(data.text)
    return {"original_text": data.text, "processed_text": processed_text}

# 🚀 Step 5: Secure Insights API
class TranscriptInput(BaseModel):
    transcript_id: str

@app.post("/insights")
async def get_insights(data: TranscriptInput, api_key: Optional[str] = Depends(verify_api_key)):
    """Returns insights securely."""
    df = await asyncio.to_thread(load_data_async)
    transcript_df = df[df['transcript_id'] == data.transcript_id]

    if transcript_df.empty:
        return {"error": "Transcript not found"}

    article_link = transcript_df['article_url'].iloc[0]
    agent_stats = transcript_df['agent'].value_counts().to_dict()
    sentiment_summary = transcript_df.groupby('agent')['sentiment'].value_counts().unstack().fillna(0).to_dict()

    return {
        "article_url": article_link,
        "agent_message_count": agent_stats,
        "sentiment_summary": sentiment_summary
    }

# 🚀 Step 6: Root Endpoint (Unprotected)
@app.get("/")
def home():
    return {"message": "Welcome to the Secure BiztelAI API!"}
