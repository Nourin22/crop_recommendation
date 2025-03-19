import streamlit as st
import base64
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('Model/rf_model.pkl')
scaler = joblib.load('Model/scaler.pkl')

def add_bg_from_local(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .title {{
            font-size: 100px;
            font-weight: bold;
            font-family: 'Times New Roman', serif;
            text-align: center;
            color: #000000; /* Jet Black */
            text-shadow: 5px 5px 10px white;
        }}
        .input-box {{
            background: rgba(255, 255, 255, 0.9);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 6px 6px 12px rgba(0, 0, 0, 0.3);
        }}
        label, .stNumberInput, .stMarkdown {{
            color: #000000 !important; /* Jet Black */
            font-size: 30px;
            font-weight: bold;
            font-family: 'Times New Roman', serif;
        }}
        .stButton button {{
            background-color: #FFD700; /* Gold */
            color: #000000;
            font-size: 30px;
            font-weight: bold;
            font-family: 'Times New Roman', serif;
            padding: 15px;
            border-radius: 10px;
            border: none;
            transition: 0.3s;
            display: block;
            margin: 0 auto;
        }}
        .stButton button:hover {{
            background-color: #DAA520; /* Darker Gold */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
add_bg_from_local("bg.png")

st.markdown(
    "<h1 class='title'>🌾 Crop Recommendation 🌱</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align: center; font-size: 24px; font-weight: bold; font-family: Times New Roman, serif; color: #FFFFFF;'>Enter Soil & Climate Details:</h3>",
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("🌿 Nitrogen (N)", min_value=0, max_value=100, value=50)
    P = st.number_input("🌾 Phosphorus (P)", min_value=0, max_value=100, value=50)
    K = st.number_input("🍃 Potassium (K)", min_value=0, max_value=100, value=50)
    ph = st.number_input("🧪 pH Level", min_value=0.0, max_value=14.0, value=7.0)

with col2:
    temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
    rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col_space1, col_btn, col_space2 = st.columns([2, 1, 2])

with col_btn:
    if st.button("🔍 **Predict**"):

        input_data = np.array([N, P, K, temperature, humidity, ph, rainfall])

        Label_Mapping = {
            'Apple': 0, 'Banana': 1, 'Blackgram': 2, 'Chickpea': 3, 'Coconut': 4, 'Coffee': 5, 'Cotton': 6, 'Grapes': 7, 
            'Jute': 8, 'Kidneybeans': 9, 'Lentil': 10, 'Maize': 11, 'Mango': 12, 'Mothbeans': 13, 'Mungbean': 14, 
            'Muskmelon': 15, 'Orange': 16, 'Papaya': 17, 'Pigeonpeas': 18, 'Pomegranate': 19, 'Rice': 20, 'Watermelon': 21
        }

        reverse_mapping = {v: k for k, v in Label_Mapping.items()}

        input_data = input_data.reshape(1, -1) 
        scaled_data = scaler.transform(input_data)

        pred = model.predict(scaled_data)[0]
        predicted_crop = reverse_mapping.get(pred, "Unknown Crop")

        st.markdown(
            f"""
            <div style="
                text-align: left;
                background-color: #008000;
                padding: 10px;
                border-radius: 10px;
                box-shadow: 5px 5px 15px rgba(255, 255, 255, 0.3);
            ">
                <h2 style="
                    color: white;
                    font-size: 15px;
                    font-weight: bold;
                    font-family: 'Times New Roman', serif;
                ">
                Crop : {predicted_crop}
                </h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("Close"):
            st.stop()
