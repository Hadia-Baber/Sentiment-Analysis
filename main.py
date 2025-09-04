import joblib

# Load model and vectorizer
model = joblib.load('sentiment_model_140.pkl')
vectorizer = joblib.load('tfidf_vectorizer_140.pkl')

# Sample sentences
sample_sentences = [
    "I absolutely loved this product! It was amazing.",
    "This was the worst experience I've ever had.",
    "My name is Hadia.",
    "AI Lab is interesting.",
    "This was boring"
]

# Predict and print results
for sentence in sample_sentences:
    vectorized_input = vectorizer.transform([sentence])
    prediction = model.predict(vectorized_input)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f"Sentence: {sentence}\nPredicted Sentiment: {sentiment}\n")

