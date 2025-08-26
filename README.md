# 📰 Fake News Detection using Machine Learning

This project aims to classify news articles as **"Fake"** or **"Real"** using various machine learning models. The trained models are deployed as an interactive web application using **Streamlit** and hosted on **Hugging Face Spaces**.

---

## 🚀 Deployed Application

The final model is deployed as a user-friendly web application where you can classify single news articles or upload a CSV file for batch predictions.  

🔗 **Live Demo**: [Fake News Detector on Hugging Face](https://huggingface.co/spaces/Utkarsh524/Fake_news_detector_models)  

### Application Interface  
![Homepage Screenshot](homepage.png)


---

## 📝 Project Overview

The core objective of this project is to build a **robust system** for identifying misinformation in news articles.  

### Key Steps:
- **Data Collection & Preprocessing**: Gathered and cleaned datasets of real & fake news articles.  
- **Feature Extraction**: Converted text into numerical format using **TF-IDF vectorization**.  
- **Model Training**: Trained multiple ML classifiers to distinguish fake vs. real news.  
- **Evaluation**: Assessed models based on accuracy, robustness, and efficiency.  
- **Deployment**: Created a web application for real-world usability.  

---

## 📊 Dataset

The dataset is a combination of two CSV files:  
- **Fake.csv** → 23,481 fake news articles  
- **True.csv** → 21,417 real news articles  

### Data Preprocessing Steps:
1. **Loading Data**: Read the CSVs into pandas DataFrames.  
2. **Labeling**: Added a column (`0 = Fake`, `1 = Real`).  
3. **Concatenation**: Merged into a unified dataset.  
4. **Feature Combination**: Combined article **title + text** into a single feature.  
5. **Text Cleaning**:  
   - Lowercasing  
   - Expanding contractions (e.g., `i'm → i am`)  
   - Removing special chars, numbers, extra spaces  
   - Lemmatization (e.g., `running → run`)  
   - Stop word removal  

---

## ⚙️ Methodology

### 1. Feature Extraction → **TF-IDF**
- Converts cleaned text into feature vectors.  
- Assigns higher weights to unique, informative words across documents.  

### 2. Model Training
- Dataset split: **80% Training | 20% Testing**  
- Trained **4 models** on extracted features.  

---

## 🤖 Models & Performance  

| Model                   | Training Accuracy |
|--------------------------|------------------|
| Logistic Regression      | 99.1%            |
| Naive Bayes              | 95.2%            |
| Random Forest            | 100%             |
| Support Vector Machine   | 99.9%            |

📌 **Note on SVM**:  
- SVM achieved high accuracy but training was computationally **slower** due to quadratic complexity **O(n²)** when computing pairwise relationships between data points.  
- While precise, it is less scalable compared to Naive Bayes or Logistic Regression.  

---

## 🔧 How to Use the Deployed App

### 1. Classify a Single Article  
- Go to **"Classify Single Article"** tab.  
- Paste article text → Click **"Analyze Article"**.  
- Get predictions & confidence scores from all models.  

### 2. Batch Classify CSV File  
- Go to **"Batch Classify CSV"** tab.  
- Upload a CSV with a column named **text**.  
- Click **"Start Batch Processing"**.  
- Download processed CSV with predictions + confidence scores.  

---

## 📂 Repository Structure
.
- Fake_news_detector.ipynb # Jupyter Notebook with preprocessing & training
-  app.py # Streamlit web app script
- requirements.txt # Python dependencies
- models/ # (Not in repo) – store downloaded trained models
- README.md # Project documentation (this file)


---

✨ **Future Improvements**:
- Incorporate **deep learning (LSTM/Transformer-based)** models for improved generalization.  
- Optimize SVM training using approximate kernel methods.  
- Add multilingual fake news detection support.  

---





