# BiztelAI Data Processing & API

## Overview

This project processes BiztelAI data and provides a FastAPI-based REST API for insights. The API allows users to interact with the BiztelAI dataset, retrieve summary information, preprocess text, and get insights based on transcript IDs. The project also includes optimizations for faster data processing and a secure authentication mechanism for the API endpoints.

## Folder Structure

- **data/** → Contains dataset files.
  - `cleaned_dataset_OOP.csv` → Cleaned dataset used for analysis (OOP version).
  - `cleaned_dataset.csv` → Cleaned dataset used for analysis (standard version).
  
- **scripts/** → Python scripts for data processing, profiling, and analysis.
  - `data_pipeline.py` → Scripts for data processing and transformation.
  - `profile_api.py` → Performance profiling script to analyze bottlenecks and optimizations.
  
- **api/** → FastAPI implementation.
  - `api.py` → Main API file where the FastAPI app is defined.

- **README.md** → Project documentation with setup instructions.
- **requirements.txt** → List of dependencies required for the project.

## Installation

1. **Clone the repository:**

```bash
   git clone https://github.com/l1ght14/BiztelAI_Project.git
   cd BiztelAI_Project

## 🌍 Deployment
The API is deployed and accessible publicly.

### Deployment URL:
- **API URL:** [https://biztelaiproject-production.up.railway.app](https://biztelaiproject-production.up.railway.app)
- **API Docs:** [https://biztelaiproject-production.up.railway.app/docs](https://biztelaiproject-production.up.railway.app/docs)

### Instructions:
1. Use the `/summary` endpoint to get the summary.
2. Use the `/preprocess` endpoint to preprocess text.
3. Use the `/insights` endpoint to get transcript-based insights.
