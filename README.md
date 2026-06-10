# 🎵 Spotify Music Analytics & Song Popularity Prediction Platform

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green.svg)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Visualizations-purple.svg)

---

# 🚀 Live Demo

🔗 **Application:** [Add Your Streamlit URL Here]

🔗 **GitHub Repository:** [Add Your GitHub Repo URL Here]

---

# 📌 Project Overview

Spotify Music Analytics & Song Popularity Prediction Platform is a complete end-to-end Machine Learning and Data Analytics application built using Python, Streamlit, Scikit-Learn, Pandas, and Plotly.

The platform allows users to:

- Explore Spotify music trends
- Analyze artist and track characteristics
- Compare multiple machine learning models
- Predict song popularity scores
- Generate intelligent music insights
- Visualize audio feature relationships
- Understand factors influencing hit songs

This project combines Data Analytics, Feature Engineering, Machine Learning, Model Evaluation, and Dashboard Development into a single production-ready application.

---

# 🎯 Business Problem

Spotify contains millions of songs with varying popularity levels.

Artists, producers, and music analysts often want to understand:

- What makes a song popular?
- Which audio features influence success?
- Can popularity be predicted before release?
- Which machine learning model performs best?

This platform answers all these questions through data-driven analytics and predictive modeling.

---

# 📊 Dataset Information

### Source

Spotify Tracks Dataset

### Records

- 150,000+ Tracks
- Artist Information
- Audio Features
- Popularity Scores
- Genre Information

### Key Features

| Feature | Description |
|----------|-------------|
| Danceability | How suitable a track is for dancing |
| Energy | Intensity and activity level |
| Loudness | Overall loudness of track |
| Speechiness | Spoken word presence |
| Acousticness | Acoustic confidence measure |
| Instrumentalness | Instrumental content |
| Liveness | Live audience presence |
| Valence | Musical positivity |
| Tempo | BPM |
| Duration | Track duration |
| Popularity | Spotify popularity score |

---

# 🛠 Technology Stack

## Programming Language

- Python

## Frontend

- Streamlit

## Data Processing

- Pandas
- NumPy

## Machine Learning

- Scikit-Learn
- XGBoost

## Visualization

- Plotly

## Model Serialization

- Joblib

## Deployment

- Streamlit Community Cloud

---

# 🏗 Project Architecture

```text
Spotify Project
│
├── app.py
├── ml_core.py
├── train_models.py
├── ui.py
│
├── data
│   ├── tracks.csv
│   └── artists.csv
│
├── artifacts
│   ├── best_rf.pkl
│   ├── scaler.pkl
│   ├── encoder.pkl
│   ├── final_features.pkl
│   ├── model_results.pkl
│   └── dashboard_data.parquet
│
├── assets
│
├── requirements.txt
│
└── README.md
```

# ⚙️ Machine Learning Pipeline

## Step 1 – Data Cleaning

- Missing value treatment
- Duplicate removal
- Feature consistency checks

## Step 2 – Feature Engineering

Created additional features such as:

- Track Age
- Duration in Minutes
- Artist Popularity Metrics
- Genre Statistics
- Loudness Transformation

## Step 3 – Data Preprocessing

- Encoding categorical features
- Feature scaling
- Outlier handling

## Step 4 – Model Training

Multiple regression models were trained and evaluated.

### Models Used

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor

---

# 🏆 Best Model

### Tuned Random Forest Regressor

Performance:

| Metric | Score |
|----------|--------|
| R² Score | 0.66+ |
| RMSE | ~10.65 |

This model was selected as the final production model.

---

# 📈 Dashboard Features

## 🏠 Overview

Provides:

- Dataset summary
- Music ecosystem insights
- Popularity distribution
- High-level KPIs

---

## 📊 Data Explorer

Users can:

- Browse tracks
- Filter records
- Explore artist information
- Search songs

---

## 📉 Analytics

Includes:

- Popularity distributions
- Feature correlations
- Genre analysis
- Trend visualizations

---

## 🧠 Model Lab

Provides:

- Model performance comparison
- Evaluation metrics
- Training insights

---

## 🏆 Model Comparison

Compare:

- Random Forest
- XGBoost
- Linear Models

Using:

- R² Score
- RMSE
- Performance rankings

---

## 🎯 Prediction Studio

Allows users to:

Input song characteristics and receive:

- Predicted Popularity Score
- Popularity Category
- Success Potential Insights

---

## 💡 AI Insights

Generates:

- Trend observations
- Feature importance findings
- Popularity recommendations

---

# 📷 Application Screenshots

## Dashboard

(Add Screenshot Here)

---

## Analytics

(Add Screenshot Here)

---

## Prediction Studio

(Add Screenshot Here)

---

# 🚀 Installation Guide

Clone Repository

```bash
git clone https://github.com/yourusername/spotify-popularity-prediction.git

cd spotify-popularity-prediction
```

Create Virtual Environment

```bash
python -m venv .venv
```

Activate Environment

Mac/Linux

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

Train Models

```bash
python train_models.py --tracks data/tracks.csv --artists data/artists.csv
```

Run Application

```bash
streamlit run app.py
```

---

# 📂 Generated Artifacts

The training pipeline automatically generates:

| Artifact | Purpose |
|-----------|---------|
| best_rf.pkl | Final trained model |
| scaler.pkl | Feature scaler |
| encoder.pkl | Encoders |
| final_features.pkl | Feature list |
| model_results.pkl | Evaluation results |
| dashboard_data.parquet | Dashboard optimized dataset |

---

# 🎯 Learning Outcomes

Through this project:

- End-to-End ML Workflow
- Feature Engineering
- Model Optimization
- Hyperparameter Tuning
- Interactive Dashboard Development
- Data Visualization
- Cloud Deployment
- Production-Level Project Structuring

---

# 🔮 Future Enhancements

- Deep Learning Models
- Spotify API Integration
- Real-Time Music Analytics
- Recommendation System
- Genre Classification
- Song Hit Probability Predictor
- LLM-Powered Music Insights

---

# 👨‍💻 Author

### Krishna Namdev

Data Science | Machine Learning | Analytics


---

# ⭐ Support

If you found this project useful:

⭐ Star the repository

🍴 Fork the project

🧠 Share feedback

---

# 📜 License

This project is intended for educational, research, and portfolio purposes.

© 2026 Krishna Namdev
