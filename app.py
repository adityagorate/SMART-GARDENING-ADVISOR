import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page config
st.set_page_config(
    page_title="Smart Gardening Advisor",
    page_icon="🌱",
    layout="wide"
)

# Load model & scaler
@st.cache_resource
def load_model():
    model = joblib.load('models/crop_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Ideal NPK & pH ranges (approximate – rule-based fertilizer suggestion)
# You can expand this dictionary a lot more

CROP_REQUIREMENTS = {
    'rice':         {'N': (80, 120),  'P': (30, 60),   'K': (30, 60),   'ph': (5.5, 7.0)},
    'maize':        {'N': (100,150),  'P': (40, 70),   'K': (30, 60),   'ph': (5.5, 7.5)},
    'chickpea':     {'N': (20, 40),   'P': (40, 60),   'K': (30, 50),   'ph': (6.0, 7.5)},
    'kidneybeans':  {'N': (20, 40),   'P': (50, 80),   'K': (30, 60),   'ph': (5.5, 7.0)},
    'pigeonpeas':   {'N': (20, 40),   'P': (40, 70),   'K': (20, 50),   'ph': (5.5, 7.5)},
    'mothbeans':    {'N': (20, 40),   'P': (40, 60),   'K': (20, 40),   'ph': (5.0, 7.5)},
    'mungbean':     {'N': (20, 40),   'P': (35, 60),   'K': (15, 35),   'ph': (6.0, 7.5)},
    'blackgram':    {'N': (20, 40),   'P': (40, 60),   'K': (20, 40),   'ph': (6.0, 7.5)},
    'lentil':       {'N': (20, 40),   'P': (40, 60),   'K': (30, 50),   'ph': (6.0, 8.0)},
    'pomegranate':  {'N': (60,100),   'P': (30, 60),   'K': (40, 80),   'ph': (5.5, 7.5)},
    'banana':       {'N': (150,250),  'P': (50, 90),   'K': (200,350),  'ph': (5.5, 7.0)},
    'mango':        {'N': (100,200),  'P': (40, 80),   'K': (100,200),  'ph': (5.5, 7.5)},
    'grapes':       {'N': (80,150),   'P': (40, 80),   'K': (100,200),  'ph': (5.5, 7.0)},
    'watermelon':   {'N': (80,120),   'P': (40, 70),   'K': (100,150),  'ph': (6.0, 7.0)},
    'muskmelon':    {'N': (80,120),   'P': (40, 70),   'K': (100,150),  'ph': (6.0, 7.0)},
    'apple':        {'N': (80,150),   'P': (40, 80),   'K': (100,200),  'ph': (5.5, 6.5)},
    'orange':       {'N': (100,200),  'P': (40, 80),   'K': (100,200),  'ph': (5.5, 7.5)},
    'papaya':       {'N': (150,250),  'P': (50, 100),  'K': (150,300),  'ph': (6.0, 7.0)},
    'coconut':      {'N': (100,200),  'P': (40, 80),   'K': (150,300),  'ph': (5.5, 8.0)},
    'cotton':       {'N': (100,150),  'P': (40, 70),   'K': (40, 80),   'ph': (5.5, 8.0)},
    'jute':         {'N': (80,120),   'P': (40, 70),   'K': (40, 80),   'ph': (5.5, 7.5)},
    'coffee':       {'N': (150,250),  'P': (50, 100),  'K': (150,300),  'ph': (5.0, 6.0)}
}

# UI
st.title("🌱 Smart Gardening Advisor")
st.markdown("Enter your soil and weather conditions to get crop & fertilizer recommendations.")

# Input columns 
col1, col2 = st.columns(2)

with col1:
    st.subheader("Soil Nutrients")
    N = st.slider("Nitrogen (kg/ha)", 0, 140, 50, step=1)
    P = st.slider("Phosphorus (kg/ha)", 0, 145, 50, step=1)
    K = st.slider("Potassium (kg/ha)", 0, 205, 50, step=1)
    ph = st.slider("Soil pH", 3.5, 10.0, 7.0, step=0.1)

with col2:
    st.subheader("Climate")
    temperature = st.slider("Temperature (°C)", 8.0, 45.0, 25.0, step=0.5)
    humidity    = st.slider("Humidity (%)", 10.0, 100.0, 60.0, step=1.0)
    rainfall    = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0, step=5.0)

# Predict button 
if st.button("Get Recommendations", type="primary"):
    # Prepare input
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)

    # Predict crop
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    # Top 3 crops
    classes = model.classes_
    top_indices = np.argsort(probabilities)[::-1][:3]
    top_crops = [(classes[i], probabilities[i]) for i in top_indices]

    # Display results 
    st.success(f"**Recommended Crop:** {prediction}")
    
    st.subheader("Top 3 Crop Recommendations")
    for crop, prob in top_crops:
        st.write(f"• **{crop.title()}** – Confidence: {prob:.1%}")

    # Fertilizer suggestion (rule-based) 
    st.subheader("Fertilizer Recommendation")
    
    if prediction in CROP_REQUIREMENTS:
        req = CROP_REQUIREMENTS[prediction]
        
        def suggest(amount, nutrient):
            if amount < req[nutrient][0]:
                deficit = req[nutrient][0] - amount
                return f"**{nutrient} deficiency** – Add ≈ {deficit:.0f} kg/ha (e.g., Urea for N, DAP for P, MOP for K)"
            elif amount > req[nutrient][1]:
                return f"**{nutrient} is high** – No addition needed, consider crop rotation"
            else:
                return f"**{nutrient} is adequate**"

        st.markdown(f"**For {prediction.title()}:**")
        st.write(f"- pH range: {req['ph'][0]} – {req['ph'][1]} → Current: {ph:.1f}")
        st.write(suggest(N, 'N'))
        st.write(suggest(P, 'P'))
        st.write(suggest(K, 'K'))
        
        st.info("Note: These are general guidelines. Get a detailed soil test for precise fertilizer application.")
    else:
        st.warning("Fertilizer suggestion not available for this crop yet.")