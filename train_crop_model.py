# train_crop_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings

warnings.filterwarnings('ignore')

def main():
    print("=== Smart Gardening Advisor - Model Training ===")
    print("Loading data...")

    # Adjust path if your folder structure is different
    data_path = "data/Crop_recommendation.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        print("Please download from: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
        return

    print("Shape:", df.shape)
    print("\nCrop distribution:\n", df['label'].value_counts())

    # Features & Target
    X = df.drop('label', axis=1)
    y = df['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Scaling
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train)

    # Evaluation
    print("\nEvaluating...")
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save
    print("\nSaving model and scaler...")
    joblib.dump(model, "models/crop_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("Done! Model and scaler saved in 'models/' folder.")

if __name__ == "__main__":
    main()