# Use the official Python image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install required Python packages
RUN pip install --no-cache-dir fastapi uvicorn pandas nltk

# Download NLTK resources inside the container
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run FastAPI inside the container
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
