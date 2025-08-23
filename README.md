# Fake_news_detect

Streamlit demo for Fake News / Fake-Real classification.  
This app loads pre-trained scikit-learn models saved as `.pkl` files inside `models/` and a `tfidf_vectorizer.pkl` used for text vectorization.

## Repo structure
Fake_news_detect/
├─ app.py
├─ requirements.txt
├─ README.md
├─ .gitignore
└─ models/
├─ tfidf_vectorizer.pkl
├─ logistic_model.pkl
└─ (other .pkl model files)