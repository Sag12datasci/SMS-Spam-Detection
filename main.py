import pickle
from sklearn.ensemble import RandomForestClassifier
from analyze import load_and_preprocess_data
import pathlib

BASE_DIR=pathlib.path(__file__).parent

DATASET_PATH=BASE_DIR / 'spam.csv'

# Filepath to the dataset


# Load and preprocess data
X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data(DATASET_PATH)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the trained model and vectorizer
with open("model.pkl", "wb") as model_file:
    pickle.dump(clf, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully!")
