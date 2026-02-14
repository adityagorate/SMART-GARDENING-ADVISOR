# 🌱 SMART-GARDENING-ADVISOR

**An AI-powered crop and fertilizer recommendation system for home gardeners and small-scale farmers**

This project helps users select the most suitable crops and get practical fertilizer recommendations based on:

- Soil nutrients (Nitrogen, Phosphorus, Potassium, pH)
- Climate conditions (temperature, humidity, rainfall)

It uses a trained **Random Forest** classifier (achieving ~98–99% accuracy) and simple rule-based logic for fertilizer suggestions.

---

## ✨ Features

- Predicts the **best crop** (and top 3 alternatives) with confidence scores
- Provides **fertilizer recommendations** (NPK additions/deficiencies) for the suggested crop
- Clean, interactive **Streamlit web interface** with sliders
- Easy to run locally — no complex setup required
- Pure Python implementation (no Jupyter notebooks needed in final version)

---

## Demo Screenshots

<img width="1355" height="588" alt="Image" src="https://github.com/user-attachments/assets/386dc76b-af27-43ad-8b48-2f4d36261967" />

<img width="1330" height="553" alt="Image" src="https://github.com/user-attachments/assets/c44a1a18-b47c-4070-bbf8-6178b86c63ef" />

---

## Tech Stack

- **Language**: Python 3.9+
- **Machine Learning**: scikit-learn (RandomForestClassifier), joblib
- **Web App**: Streamlit
- **Data handling**: pandas, numpy
- **Dataset**: Crop Recommendation Dataset (~22 crops, 2200 samples)

---

## Project Structure

```text
Smart-Gardening-Advisor/
├── data/
│   └── Crop_recommendation.csv               # Kaggle dataset
├── models/
│   ├── crop_model.pkl
│   └── scaler.pkl                           # trained model & scaler
├── train_crop_model.py                      # one-time training script
├── app.py                                   # Streamlit application
├── requirements.txt
└── README.md

---

## 🚀 Quick Start

1. Clone the repository

Bashgit clone https://github.com/adityagorate/Smart-Gardening-Advisor.git

cd Smart-Gardening-Advisor

2. Install dependencies

Bashpip install -r requirements.txt

3. Download the dataset

Download from:

https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

Place the file here:

data/Crop_recommendation.csv

4. Train the model (run once)

Bashpython train_crop_model.py

You should see ~98–99% accuracy.

5. Launch the app

Bashstreamlit run app.py

Open http://localhost:8501 in your browser.

---
## 📊 Model Performance

[table.csv](https://github.com/user-attachments/files/25314902/table.csv)

---
