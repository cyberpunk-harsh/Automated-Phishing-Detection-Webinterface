import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset (Ensure 'phishing_dataset.csv' is present)
df = pd.read_csv("phishing_dataset.csv")

# Features (URLs) and Labels (0 = Legitimate, 1 = Phishing)
X = df["url"]
y = df["label"]

# Convert URLs into numerical features using TfidfVectorizer
vectorizer = TfidfVectorizer()
X_features = vectorizer.fit_transform(X)

# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Train a Machine Learning model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

# Create 'model' directory if not exists
if not os.path.exists("model"):
    os.makedirs("model")

# Save trained model
with open("model/phishing_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Save vectorizer
with open("model/vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("✅ Model and Vectorizer saved successfully in 'model/' folder!")
