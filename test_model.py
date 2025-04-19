import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from analyze import load_and_preprocess_data

# Load the model and vectorizer
with open("model.pkl", "rb") as model_file:
    clf = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Filepath to the dataset
DATASET_PATH = r"c:\Users\User\Downloads\SMS-Spam-Detection\Ml_EG_INtership_project\spam.csv"

# Load and preprocess data
_, X_test, _, y_test, _ = load_and_preprocess_data(DATASET_PATH)

# Predict on test data
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Display metrics
print("\n=== Classification Metrics ===")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Create directory for saving plots
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# Enhanced visualization section with better error handling
try:
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig('plots/confusion_matrix.png')
    plt.close()

    # Feature Importance Plot
    plt.figure(figsize=(12, 6))
    feature_names = vectorizer.get_feature_names_out()
    top_n = 15
    top_indices = np.argsort(clf.feature_importances_)[-top_n:]
    top_features = feature_names[top_indices]
    top_importances = clf.feature_importances_[top_indices]

    sns.barplot(x=top_importances, y=top_features)
    plt.title("Top 15 Most Important Features")
    plt.xlabel("Feature Importance")
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()

    # ROC Curve
    plt.figure(figsize=(8, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('plots/roc_curve.png')
    plt.close()

    # Distribution of Message Lengths (Fixed)
    plt.figure(figsize=(10, 6))
    # Convert sparse matrix to dense array for the test data
    message_lengths = [len(str(msg).split()) for msg in X_test.toarray()]
    sns.histplot(data=message_lengths, bins=50, hue=y_test, multiple="stack")
    plt.title('Distribution of Message Lengths')
    plt.xlabel('Number of Words')
    plt.ylabel('Count')
    plt.savefig('plots/message_length_distribution.png')
    plt.close()

except Exception as e:
    print(f"Error occurred while generating plots: {str(e)}")

# User Input Testing
while True:
    user_input = input("Enter a message to classify (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    user_vec = vectorizer.transform([user_input])
    prediction = clf.predict(user_vec)
    probability = clf.predict_proba(user_vec)[0]
    print(f"Prediction: {'Spam' if prediction[0] == 1 else 'Ham'}")
    print(f"Confidence: {max(probability)*100:.2f}%")

'''
Sample Input Examples:

1. Obvious Spam:
   "CONGRATULATIONS! You've won a $1000 gift card! Call 1-800-555-0123 now to claim!"
   Expected: Spam (High confidence)

2. Regular Ham:
   "Hey, can we meet at the coffee shop at 3pm?"
   Expected: Ham (High confidence)

3. Marketing but legitimate:
   "Your order #1234 has been shipped. Track it here: https://legitsite.com/track"
   Expected: Ham (Medium confidence)

4. Borderline case:
   "20% OFF at our store! Valid only today!"
   Expected: Could be either (Lower confidence)

5. Normal conversation:
   "Will be home in 10 minutes. Need anything from the store?"
   Expected: Ham (High confidence)
'''
