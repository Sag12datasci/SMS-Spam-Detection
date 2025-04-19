import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath):
    # Load the dataset
    data = pd.read_csv(filepath, encoding='ISO-8859-1')
    data = data[['v1', 'v2']]  # Keep only the relevant columns
    data.columns = ['label', 'message']  # Rename columns for clarity

    # Encode labels (ham -> 0, spam -> 1)
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data['message'], data['label'], test_size=0.2, random_state=42
    )

    # Vectorize the text data
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, y_train, y_test, vectorizer
