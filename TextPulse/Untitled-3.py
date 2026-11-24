
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load the datasets
train_df = pd.read_csv("dataset/train_data.csv")
val_df = pd.read_csv("dataset/validation_data.csv")
test_df = pd.read_csv("dataset/test-data.csv")

# Keep only rows with valid text and labels
train_df = train_df.dropna(subset=["text", "label"])
val_df = val_df.dropna(subset=["text", "label"])
test_df = test_df.dropna(subset=["text", "label"])

# Prepare training and validation data
X_train = train_df["text"].astype(str)
y_train = train_df["label"].astype(int)

X_val = val_df["text"].astype(str)
y_val = val_df["label"].astype(int)

# Create the TF-IDF and SVM model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2)
    )),
    ("svm", LinearSVC())
])

# Train the model
print("Training TextPulse model...")
model.fit(X_train, y_train)

# Validate the model
val_predictions = model.predict(X_val)

print("\nValidation Accuracy:",
      accuracy_score(y_val, val_predictions) * 100, "%")

print("\nValidation Report:")
print(classification_report(y_val, val_predictions))

# Test the model
X_test = test_df["text"].astype(str)
y_test = test_df["label"].astype(int)

test_predictions = model.predict(X_test)

print("\nTest Accuracy:",
      accuracy_score(y_test, test_predictions) * 100, "%")

print("\nTest Report:")
print(classification_report(y_test, test_predictions))

# Save the trained model
joblib.dump(model, "models/textpulse_model.pkl")

print("\nTextPulse model trained and saved successfully!")
