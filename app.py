import streamlit as st
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("📰 Fake News Detector")

user_input = st.text_area("Enter a news article:", height=200)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        X_input = vectorizer.transform([user_input])
        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input).max()
        label = "🟢 Real" if pred == 1 else "🔴 Fake"
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence:** `{prob:.2%}`")
