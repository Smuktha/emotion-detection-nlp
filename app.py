# ------------------------------------------------
# 💫 EmotionSense AI 
# Author: Muktha
# ------------------------------------------------

import streamlit as st
import pickle
import emoji
import numpy as np

# Load Model and Vectorizer
model = pickle.load(open("emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Emoji mapping
emoji_dict = {
    'happiness': '😄',
    'sadness': '😢',
    'love': '❤️',
    'fun': '😂',
    'surprise': '😲',
    'neutral': '😐',
    'worry': '😟'
}

# Streamlit page config
st.set_page_config(page_title="EmotionSense AI", page_icon="💫", layout="centered")

# Inject custom CSS
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
            color: white;
            font-family: 'Poppins', sans-serif;
        }
        .stTextArea textarea {
            background-color: rgba(255, 255, 255, 0.1);
            color: white;
            border-radius: 12px;
            border: none;
            font-size: 1rem;
        }
        .stButton>button {
            background: linear-gradient(90deg, #00c6ff, #0072ff);
            color: white;
            font-weight: 600;
            border-radius: 10px;
            border: none;
            padding: 0.6rem 1.2rem;
            transition: 0.3s ease;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #0072ff, #00c6ff);
        }
        .emotion-box {
            text-align: center;
            font-size: 1.4rem;
            background-color: rgba(255,255,255,0.08);
            border-radius: 15px;
            padding: 1rem;
            margin-top: 20px;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# App title
st.title("💫 EmotionSense AI")
st.markdown("### Detect the emotion behind any text using Machine Learning 🧠")

# User input
text_input = st.text_area("🗨️ Enter your message below:", placeholder="Type something like 'I’m so excited for my new job!'")

if st.button("🔍 Analyze Emotion"):
    if text_input.strip():
        # Transform and predict
        X_vec = vectorizer.transform([text_input])
        pred = model.predict(X_vec)[0]
        prob = model.predict_proba(X_vec)[0]
        confidence = np.max(prob)

        # Display results
        st.markdown(
            f"<div class='emotion-box'>"
            f"Predicted Emotion: <b>{pred.capitalize()}</b> {emoji_dict.get(pred, '🙂')}<br>"
            f"<small>Confidence: {confidence*100:.2f}%</small>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Progress bar
        st.progress(float(confidence))

    else:
        st.warning("⚠️ Please enter some text before analyzing!")

st.markdown("---")
st.caption("✨ Developed by Muktha | Powered by Streamlit & ML 💻")
