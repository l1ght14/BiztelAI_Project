import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the JSON dataset
dataset_path = "BiztelAI_DS_Dataset_Mar'25.json"
with open(dataset_path, 'r') as f:
    data = json.load(f)

# Convert JSON structure into a DataFrame
records = []
for transcript_id, transcript in data.items():
    article_url = transcript.get('article_url')
    config = transcript.get('config')
    conversation = transcript.get('content', [])
    for turn in conversation:
        turn['transcript_id'] = transcript_id
        turn['article_url'] = article_url
        turn['config'] = config
        records.append(turn)

df = pd.DataFrame(records)

# Identify columns with list-type values
for col in df.columns:
    print(f"Column: {col}, Type: {df[col].dtype}")

# Convert list-type columns to strings before dropping duplicates
if 'knowledge_source' in df.columns:
    df['knowledge_source'] = df['knowledge_source'].astype(str)

if 'turn_rating' in df.columns:
    df['turn_rating'] = df['turn_rating'].astype(str)

# Drop duplicate records
df.drop_duplicates(inplace=True)

# Fill missing values with an empty string
df.fillna("", inplace=True)

# Convert categorical data (e.g., agent names to numeric codes)
df['agent_code'] = df['agent'].astype('category').cat.codes

# Basic text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]  # Remove punctuation and numbers
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

df['processed_message'] = df['message'].apply(preprocess_text)


df.to_csv("cleaned_dataset.csv", index=False)
print("\n Processed dataset saved as 'cleaned_dataset.csv'")
# Count the number of messages referencing each article URL
article_counts = df['article_url'].value_counts()

print("\n Most Discussed Articles:")
print(article_counts.head(10))  # Display the top 10 most mentioned articles
# Count messages sent by each agent
agent_message_counts = df['agent'].value_counts()

print("\n Number of Messages Sent by Each Agent:")
print(agent_message_counts)
import matplotlib.pyplot as plt
import seaborn as sns

# Filter only messages from agent_1
agent1_sentiments = df[df['agent'] == 'agent_1']['sentiment'].value_counts()

print("\n Sentiment Distribution for Agent 1:")
print(agent1_sentiments)

# Plot sentiment distribution for Agent 1
plt.figure(figsize=(8,5))
sns.barplot(x=agent1_sentiments.index, y=agent1_sentiments.values, palette="viridis")
plt.title("Sentiment Distribution for Agent 1")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()
# Filter only messages from agent_2
agent2_sentiments = df[df['agent'] == 'agent_2']['sentiment'].value_counts()

print("\n Sentiment Distribution for Agent 2:")
print(agent2_sentiments)

# Plot sentiment distribution for Agent 2
plt.figure(figsize=(8,5))
sns.barplot(x=agent2_sentiments.index, y=agent2_sentiments.values, palette="coolwarm")
plt.title("Sentiment Distribution for Agent 2")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()





