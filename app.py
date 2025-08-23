# app.py
"""
Streamlit app for Fake News Detection.
- Automatically downloads models from Google Drive into models/ folder
- Loads tfidf_vectorizer.pkl and other model .pkl files
- Provides single-message classification + batch CSV classify and download
"""

import os
import re
import joblib
import numpy as np
import streamlit as st
from pathlib import Path

# -------------------------
# Google Drive download setup
# -------------------------
import gdown

MODEL_DIR = Path(__file__).parent.joinpath("models")
MODEL_DIR.mkdir(exist_ok=True)

# 🔴 Replace with your actual Google Drive file IDs
MODEL_FILES = {
    "logistic_model.pkl": "1KM3gaJoIjv3qYD6wJq-Y70zjdHzotfTO",
    "naive_model.pkl": "1jbguELIRqHe_EvU4j3coe_lI5fWDFgV9",
    "rf_model.pkl": "1gzbDtI6HmcMH8qF7geuaCRW3Xjs-U7S4",
    "svm_model.pkl": "1QDFE5Y1CG5XH6Q_gxVo6zfqN1_4ACQro",
    "tfidf_vectorizer.pkl": "1L3BGBkZ_kawIvWdZ6t5_rk5gl6xcebkA",
}

for fname, file_id in MODEL_FILES.items():
    fpath = MODEL_DIR / fname
    if not fpath.exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        st.write(f"Downloading {fname} from Google Drive...")
        gdown.download(url, str(fpath), quiet=False)

# -------------------------
# NLTK & spaCy setup
# -------------------------
try:
    import nltk
    nltk.data.find("corpora/stopwords")
except Exception:
    import nltk
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
if nlp is not None:
    stop_words = stop_words | nlp.Defaults.stop_words

lemma = WordNetLemmatizer()

# -------------------------
# Text cleaning
# -------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    contractions = {
        r"i'm": "i am", r"he's": "he is", r"she's": "she is", r"that's": "that is",
        r"what's": "what is", r"where's": "where is", r"\'ll": " will",
        r"\'ve": " have", r"\'re": " are", r"\'d": " would", r"won't": "will not",
        r"can't": "cannot", r"n't": " not"
    }
    for pat, repl in contractions.items():
        text = re.sub(pat, repl, text)

    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [lemma.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# -------------------------
# Load vectorizer + models
# -------------------------
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detector (Google Drive Models)")

VECT_FILE = MODEL_DIR.joinpath("tfidf_vectorizer.pkl")
if not VECT_FILE.exists():
    st.error("tfidf_vectorizer.pkl missing. Check Google Drive file IDs.")
    st.stop()
vectorizer = joblib.load(str(VECT_FILE))

models = {}
for p in MODEL_DIR.glob("*.pkl"):
    if p.name == "tfidf_vectorizer.pkl":
        continue
    try:
        models[p.stem] = joblib.load(str(p))
    except Exception as e:
        st.warning(f"Failed to load {p.name}: {e}")

if not models:
    st.error("No models found. Check Google Drive file IDs.")
    st.stop()

st.sidebar.header("Available models")
for name in models:
    st.sidebar.write(f"- {name}")

# -------------------------
# Prediction helper
# -------------------------
def get_confidence_and_pred(model, X):
    X_try = X
    used_dense = False
    try:
        pred = model.predict(X_try)[0]
    except ValueError:
        X_try = X.toarray()
        used_dense = True
        pred = model.predict(X_try)[0]

    conf = None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_try)[0]
            conf = round(float(np.max(probs)) * 100, 2)
        elif hasattr(model, "decision_function"):
            df = model.decision_function(X_try)
            df = np.array(df)
            if df.ndim == 1 and df.size > 1:
                exp = np.exp(df - np.max(df))
                probs = exp / exp.sum()
                conf = round(float(np.max(probs)) * 100, 2)
            else:
                val = float(df) if df.shape == () else float(df[0])
                prob = 1.0 / (1.0 + np.exp(-val))
                conf = round(prob * 100, 2)
    except Exception:
        conf = None

    return pred, conf, used_dense

# -------------------------
# UI: Single input
# -------------------------
st.subheader("Classify a single message")

example = "Breaking: government announces free money for everyone!"
user_input = st.text_area("Enter text", value=example, height=160)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter text to classify.")
    else:
        cleaned = clean_text(user_input)
        X = vectorizer.transform([cleaned])
        st.markdown("**Cleaned input:**")
        st.write(cleaned)

        st.markdown("**Predictions:**")
        for name, m in models.items():
            try:
                pred, conf, _ = get_confidence_and_pred(m, X)
                if conf is not None:
                    st.write(f"**{name}** → {pred} ({conf}%)")
                else:
                    st.write(f"**{name}** → {pred}")
            except Exception as e:
                st.write(f"**{name}** → Error: {e}")

# -------------------------
# Batch classify CSV
# -------------------------
st.subheader("Batch classify (CSV upload)")
st.markdown("Upload a CSV with a `text` column. Adds `pred_<model>` and `conf_<model>`.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    import pandas as pd
    df = pd.read_csv(uploaded_file)
    if "text" not in df.columns:
        st.error("CSV must contain a `text` column.")
    else:
        out_df = df.copy()
        for name, m in models.items():
            preds, confs = [], []
            for txt in out_df["text"].astype(str):
                cleaned = clean_text(txt)
                X = vectorizer.transform([cleaned])
                try:
                    pred, conf, _ = get_confidence_and_pred(m, X)
                except Exception:
                    pred, conf = None, None
                preds.append(pred)
                confs.append(conf)
            out_df[f"pred_{name}"] = preds
            out_df[f"conf_{name}"] = confs

        st.success("Done")
        st.dataframe(out_df.head(50))
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results CSV", data=csv_bytes,
                           file_name="classified_results.csv", mime="text/csv")
