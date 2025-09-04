import joblib
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load saved model and vectorizer
model = joblib.load("sentiment_model_140.pkl")
vectorizer = joblib.load("tfidf_vectorizer_140.pkl")

# Load dataset (same one used for training)
df = pd.read_csv("dataset/Sentiment140 dataset.csv", encoding="latin-1", header=None, skiprows=1)
df.columns = ['target', 'id', 'date', 'query', 'user', 'text']
df = df[['target', 'text']]
df['target'] = df['target'].map({0: 0, 4: 1})

# Clean text (reuse same function)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text'] = df['text'].apply(clean_text)

# Split again to get test data (same as before)
X = df['text']
y = df['target']
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform test data using saved vectorizer
X_test_vec = vectorizer.transform(X_test)

# Predict
y_pred = model.predict(X_test_vec)

# Accuracy
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\n🧮 Confusion Matrix:")
print(conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
