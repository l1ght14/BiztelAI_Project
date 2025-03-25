import pandas as pd
import numpy as np
import json
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


#  Step 1: Data Loader Class
class DataLoader:
    """Loads dataset from a file (JSON or CSV)."""

    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """Loads data from a JSON or CSV file."""
        if self.file_path.endswith(".json"):
            with open(self.file_path, "r") as f:
                data = json.load(f)
            return pd.DataFrame.from_dict(data, orient='index').explode('content').reset_index()
        elif self.file_path.endswith(".csv"):
            return pd.read_csv(self.file_path)
        else:
            raise ValueError("Unsupported file format. Use .json or .csv")


#  Step 2: Data Cleaner Class

class DataCleaner:
    """Handles missing values, duplicates, and fixes data issues."""

    def __init__(self, df):
        self.df = df

    def remove_duplicates(self):
        """Removes duplicate rows."""
        self.df.drop_duplicates(inplace=True)

    def fill_missing(self):
        """Fills missing values efficiently."""
        self.df.fillna({"knowledge_source": "Unknown", "turn_rating": "Average"}, inplace=True)

    def convert_list_columns(self, columns):
        """Converts list-type columns to string format efficiently."""
        self.df[columns] = self.df[columns].astype(str)

    def clean_data(self):
        """Runs all cleaning operations."""
        self.remove_duplicates()
        self.fill_missing()
        self.convert_list_columns(["knowledge_source", "turn_rating"])
        return self.df



#  Step 3: Data Transformer Class
import re

class DataTransformer:
    """Handles text preprocessing efficiently."""

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def fast_preprocess_text(self, text):
        """Efficiently preprocess text using regex & vectorization."""
        if pd.isna(text):  # Handle missing values
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation & numbers
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)

    def transform(self, df):
        """Vectorized text processing for the 'message' column."""
        df['processed_message'] = np.vectorize(self.fast_preprocess_text)(df['message'])
        return df


# Step 4: Run the Pipeline
if __name__ == "__main__":
    # Load the data
    loader = DataLoader("cleaned_dataset.csv")
    df = loader.load_data()

    # Clean the data
    cleaner = DataCleaner(df)
    df = cleaner.clean_data()

    # Transform the text data
    transformer = DataTransformer()
    df = transformer.transform(df)

    # Save the final cleaned dataset
    df.to_csv("cleaned_dataset_OOP.csv", index=False)
    print("\nProcessed dataset saved as 'cleaned_dataset_OOP.csv'")
