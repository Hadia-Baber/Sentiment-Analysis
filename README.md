# Sentiment Analysis

A project that uses machine learning for sentiment analysis on text, classifying it as **positive** or **negative**.  
It features a clean and modern **GUI** for easy use and provides **graph visualizations** of results for better understanding.

---

##  Features
- Data preprocessing steps including:
  - Lowercasing all text  
  - Removing URLs  
  - Removing user mentions  
  - Removing hashtags  
  - Removing punctuation  
  - Removing numbers  
  - Removing extra whitespace  
  - Tokenization with TF-IDF vectorization  
- Model training using **Logistic Regression**  
- **Graphical User Interface (GUI)** for user-friendly predictions (built with customtkinter in Python)  
- **Graph visualizations** in the form of a bar chart  
- Evaluation metrics: accuracy, precision, recall, F1-score  

---

##  Data Preprocessing
Before training, the text data is cleaned and prepared:  
1. Converted to lowercase  
2. Removed URLs, mentions, hashtags, punctuation, numbers  
3. Normalized whitespace  
4. Transformed into feature vectors using **TF-IDF**  

---
## Usage
python gui.py

## 📈 Results 
- Model: Logistic Regression -
- Accuracy: 79%
-  Confusion Matrix:
   [[123918 35576] [ 31404 129102]]

