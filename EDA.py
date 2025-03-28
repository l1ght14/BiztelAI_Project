import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load dataset
file_path = "cleaned_dataset.csv"  # Ensure the correct path
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: File not found!")
    exit()

# Check basic info
def dataset_summary(df):
    print("\nDataset Overview:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDescriptive Statistics:")
    print(df.describe())

dataset_summary(df)

# Grouping by agent and article
article_summary = df.groupby("article_url").agg({
    "processed_message": "count", 
    "agent": pd.Series.nunique
}).rename(columns={"processed_message": "message_count", "agent": "unique_agents"})

print("\nArticle-wise Summary:")
print(article_summary.head())

# Count messages per agent
agent_counts = df["agent"].value_counts()
print("\nAgent-wise Message Counts:")
print(agent_counts)

# Visualizing message counts
plt.figure(figsize=(8, 4))
sns.barplot(x=agent_counts.index, y=agent_counts.values, palette="viridis")
plt.xlabel("Agent")
plt.ylabel("Message Count")
plt.title("Messages Sent by Each Agent")
plt.show()

# Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()
def analyze_sentiment(text):
    if pd.isna(text) or not text.strip():
        return 0.0  # Neutral sentiment for empty text
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']

def textblob_sentiment(text):
    if pd.isna(text) or not text.strip():
        return 0.0
    return TextBlob(text).sentiment.polarity

df["sentiment_score"] = df["processed_message"].apply(analyze_sentiment)
df["textblob_sentiment"] = df["processed_message"].apply(textblob_sentiment)

# Agent-wise Sentiment Summary
agent_sentiment = df.groupby("agent")["sentiment_score"].mean()
print("\nAgent-wise Sentiment Analysis:")
print(agent_sentiment)

# Agent-wise Sentiment Count
agent_sentiment_count = df.groupby("agent")["sentiment_score"].agg(['count', 'mean'])
print("\nAgent-wise Sentiment Count & Average:")
print(agent_sentiment_count)

# Visualizing Sentiment Distribution
plt.figure(figsize=(8, 4))
sns.boxplot(x=df["agent"], y=df["sentiment_score"], palette="coolwarm")
plt.xlabel("Agent")
plt.ylabel("Sentiment Score")
plt.title("Sentiment Distribution by Agent")
plt.show()

# Save processed results
df.to_csv("eda_sentiment_analysis.csv", index=False)
print("Processed data saved to 'eda_sentiment_analysis.csv'")
