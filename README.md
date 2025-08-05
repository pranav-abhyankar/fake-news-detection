# 📰 Fake News Detection App

A web-based application that detects whether a news article is real or fake using a machine learning model trained on labeled data. This project is built using **Streamlit**, making it interactive and easy to use.

---

## 🚀 Demo

👉 **[Try the App Locally]**  
📌 Enter the **news body text** and the model will predict whether it's real or fake with a confidence score.

---

## 💡 Features

- 🧠 **ML-Powered Detection**: Uses a trained machine learning model with TF-IDF vectorization.
- 💬 **User Input**: Accepts full body text of news articles.
- 📊 **Model Confidence**: Shows how confident the model is in its prediction.
- 🌐 **Web Interface**: Built with Streamlit for fast and simple interaction.

---

## 📂 Project Structure

```
├── app.py                        # Streamlit web application
├── body_only_fake_news_model.pkl  # Trained ML model
├── body_only_tfidf_vectorizer.pkl # Fitted TF-IDF vectorizer
├── Fake_news_prediction.ipynb   # Notebook used for model training
├── requirements.txt # Required libraries for the project
```

---

## 🛠️ How It Works

1. **Text Cleaning**: News text is cleaned by removing URLs, special characters, and converting to lowercase.
2. **Vectorization**: Cleaned text is transformed into TF-IDF features.
3. **Prediction**: A trained model (likely Logistic Regression or similar) predicts the label (real/fake).
4. **Display**: Result and confidence score are shown in the UI.

---

## 🧪 Sample Usage

![WhatsApp Image 2025-07-30 at 23 15 57_289af43f](https://github.com/user-attachments/assets/edf5d3c3-1244-4827-9799-6357aff10f13)



## 🖥️ Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add model and vectorizer files
Make sure the following files are in the same directory:
- `body_only_fake_news_model.pkl`
- `body_only_tfidf_vectorizer.pkl`

### 4. Run the app
```bash
streamlit run app.py
```

---


## 🧠 Model Training

The model was trained using a labeled dataset of real and fake news. The process involved:
- Text preprocessing (cleaning and normalization)
- TF-IDF vectorization
- Model training with supervised learning algorithm
- Evaluation using standard metrics

Check out the `Fake_news_prediction.ipynb` for full training details.

---

## 🧾 Requirements

- Python 3.7+
- Streamlit
- scikit-learn
- joblib

```bash
pip install streamlit scikit-learn joblib
```

---

## 📌 Disclaimer

This tool is for educational purposes and should not be used as a definitive source for verifying the authenticity of news. Always cross-check from trusted sources.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## ✨ Acknowledgements

- [Streamlit](https://streamlit.io)
- [scikit-learn](https://scikit-learn.org/)
- [Kaggle Dataset](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection/data)

