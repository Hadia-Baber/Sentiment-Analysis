import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------------
# Step 1: Load and clean dataset
# -------------------------------

# Load dataset and skip the first row (wrong header)
df = pd.read_csv("dataset/Sentiment140 dataset.csv", encoding="latin-1", header=None, skiprows=1)

# Rename columns
df.columns = ['target', 'id', 'date', 'query', 'user', 'text']

# Keep only relevant columns
df = df[['target', 'text']]

# Handle missing values
df['text'] = df['text'].fillna("")           # Fill missing text with empty string
df = df.dropna(subset=['target'])            # Drop rows with missing target (very rare)

# Map sentiment labels: 0 → negative, 4 → positive
df['target'] = df['target'].map({0: 0, 4: 1})

# Drop any row with unexpected target values (if any)
df = df[df['target'].isin([0, 1])]

# -------------------------------
# Step 2: Clean text
# -------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)      # Remove URLs
    text = re.sub(r'@\w+', '', text)         # Remove mentions
    text = re.sub(r'#\w+', '', text)         # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text)      # Remove punctuation
    text = re.sub(r'\d+', '', text)          # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    return text

df['text'] = df['text'].apply(clean_text)

# -------------------------------
# Step 3: Split data
# -------------------------------

X = df['text']
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 4: TF-IDF Vectorization
# -------------------------------

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# Step 5: Train model
# -------------------------------

model = LogisticRegression()
model.fit(X_train_vec, y_train)

# # -------------------------------
# # Step 6: Evaluate model
# # -------------------------------

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

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





# -------------------------------
# Step 7: Save model and vectorizer
# -------------------------------

joblib.dump(model, 'sentiment_model_140.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer_140.pkl')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2:
df = pd.read_csv("Exam/Iris.csv")

# Step 3: 
print("Missing values per column:")
print(df.isnull().sum())

# # Step 4: 
df['PetalLengthCm'] = df['PetalLengthCm'].fillna(df['PetalLengthCm'].mean())

# # Step 5: 
df.drop_duplicates(inplace=True)

# # Step 6:
sns.boxplot(x=df['PetalLengthCm'])
plt.title('Boxplot of Transaction Amount (Before Outlier Removal)')
plt.show()

# # Step 7: 
Q1 = df['PetalLengthCm'].quantile(0.25)
Q3 = df['PetalLengthCm'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['PetalLengthCm'] >= Q1 - 1.5 * IQR) & (df['PetalLengthCm'] <= Q3 + 1.5 * IQR)]


sns.boxplot(x=df['PetalLengthCm'])
plt.title('Boxplot of Transaction Amount (After Outlier Removal)')
plt.show()

X = df[['SepalLengthCm','SepalWidthCm',	'PetalLengthCm','PetalWidthCm']]
y = df['Species']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, log_preds))


lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_preds = lin_model.predict(X_test)
lin_preds_binary = [1 if p > 0.5 else 0 for p in lin_preds] 
print("Linear Regression Accuracy (rounded):", accuracy_score(y_test, lin_preds_binary))


knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_preds = knn_model.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, knn_preds))


nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_preds))


# print("\nLogistic Regression Confusion Matrix:")
# print(confusion_matrix(y_test, log_preds))

# print("\nLogistic Regression Classification Report:")
# print(classification_report(y_test, log_preds))
weights = [5, [3, [2, [1]]], 4]
def sum_weights(weights):
    total = 0
    for item in weights:
        if isinstance(item, list):
            total += sum_weights(item)
        else:
            total +=item
    return total
print("Total sum is: ", sum_weights(weights))

def astar(graph, start, goal, heuristics):
    open_list = []
    heapq.heappush(open_list, (0 + heuristics[start], 0, start, [start]))
    closed_set = set()

    while open_list:
        _, cost, current, path = heapq.heappop(open_list)
        if current == goal:
            return path

        if current in closed_set:
            continue
        closed_set.add(current)

        for neighbor, weight in graph.get(current, []):
            if neighbor not in closed_set:
                heapq.heappush(open_list, (
                    cost + weight + heuristics[neighbor],
                    cost + weight,
                    neighbor,
                    path + [neighbor]
                ))
    return None
       