import streamlit as st
import joblib
import re

# Load the trained model and vectorizer
model = joblib.load("body_only_fake_news_model.pkl")
vectorizer = joblib.load("body_only_tfidf_vectorizer.pkl")

# Clean text before prediction
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    return text

# Streamlit UI Setup
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector")
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
.result-box {
    border-radius: 10px;
    padding: 15px;
    margin-top: 15px;
    font-size: 18px;
    background-color: #f9f9f9;
    border: 2px solid #ccc;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-font">Enter the <b>body text</b> of a news article below to check if it is real or fake.</div>', unsafe_allow_html=True)

# Input field
news_body = st.text_area("✍️ News Body Content", height=300, help="Paste or type the full content of the news article here.")

# Predict button
if st.button("🔍 Check for Fake News"):
    if news_body.strip() == "":
        st.warning("⚠️ Please enter some news content.")
    else:
        cleaned = clean_text(news_body)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        proba = model.decision_function(vectorized)[0]
        confidence = abs(proba)

        # Display result
        if prediction == 1:
            st.success("✅ The news appears to be **REAL**.")
            st.markdown(f'<div class="result-box">🧠 Model Confidence: **{confidence:.2f}**</div>', unsafe_allow_html=True)
        else:
            st.error("❌ The news appears to be **FAKE**.")
            st.markdown(f'<div class="result-box">🧠 Model Confidence: **{confidence:.2f}**</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("📌 This app uses a machine learning model trained on labeled fake and real news datasets. For best results, use full article content.")
