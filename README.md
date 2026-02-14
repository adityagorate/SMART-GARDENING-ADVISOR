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

*(Add 2–4 screenshots here later – e.g. input screen, result with top crops, fertilizer advice)*

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
