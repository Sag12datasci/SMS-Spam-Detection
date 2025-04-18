import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# Load the dataset
df = pd.read_csv(r'D:/Codes/Ml_EG_INtership_project/spam.csv', encoding='latin-1')

# Display the first 10 rows of the dataset
print(df.head(10))

# Keep only the relevant columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Check the cleaned data
print(df.head(10))

# Download the stopwords from nltk
nltk.download('stopwords')

# Cache the stopwords into a set
cached_stopwords = set(stopwords.words('english'))

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()  # Convert to lowercase
    words = text.split()  # Tokenize the text
    words = [word for word in words if word not in cached_stopwords]  # Remove stopwords
    return " ".join(words)  # Join the words back into a string

# Apply the cleaning function to the 'message' column
df['clean_message'] = df['message'].apply(clean_text)

print(df[['message', 'clean_message']].head(10))

#Initialize the vectorizer with parameters
vectorizer = TfidfVectorizer(max_df=0.9, min_df=5)

#fit the vectorizer to the cleaned messages and transform
x_tfidf = vectorizer.fit_transform(df['clean_message'])

#Check the shape of the resulting feature matrix
print("TF-IDF feature matrix shape:", x_tfidf.shape)

#Split the feature matrix (x_tfidf) and the labels (df['label']) 
x_train, x_test, y_train, y_test=train_test_split(x_tfidf, df['label'], test_size=0.2, random_state=42, stratify=df['label'])


#Train a Multinomial Naive Bayes classifier
#Initialize the classifier
model= MultinomialNB(alpha=0.1)
model.fit(x_train, y_train) 


#Evalute the Model 
y_pred= model.predict(x_test)

#Print overall accuracy
print(f"Accuracy:, {accuracy_score(y_test, y_pred):.4f}")

#Print detailed classification metrics
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

#Display the  confusion matrix
cm= confusion_matrix(y_test, y_pred)
print("Confusion Matrix:", cm)

