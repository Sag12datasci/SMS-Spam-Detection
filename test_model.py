import pickle
import seaborn as sns
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from analyze import load_and_preprocess_data



BASE_DIR=pathlib.Path(__file__).parent

MODEL_PATH=BASE_DIR / 'model.pkl'
VECTORIZER_PATH=BASE_DIR / 'vectorizer.pkl'
DATASET_PATH=BASE_DIR / 'spam.csv'
PLOTS_DIR= BASE_DIR / 'plots'

# Load the model and vectorizer
with open(MODEL_PATH, "rb") as model_file:
    clf = pickle.load(model_file) #Pcikle.load() is basiclly converting the byte stream into a python object 

with open(VECTORIZER_PATH, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

df = pd.read_csv(DATASET_PATH, encoding="latin-1")
df = df.rename(columns={"v1": "label", "v2": "text"})[["label", "text"]]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Load and preprocess data
X_train, X_test, y_train, y_test, df_full = load_and_preprocess_data(DATASET_PATH) # Only "_" to intentionally ignore the unpacked value and here the function returns 5 different values and those values are directly unpacke into the variables (_, X_test, _, y_test, _)

# Predict on test data
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1] # Get probabilities for the positive class (Spam)

_, df_test = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)


# Display metrics
print("\n=== Classification Metrics ===")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Create directory for saving plots
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

import traceback
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
    plt.savefig(PLOTS_DIR /'confusion_matrix.png')
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
    plt.savefig(PLOTS_DIR /'feature_importance.png')
    plt.close()

    # ROC Curve
    plt.figure(figsize=(8, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr) # calculates the area under the curve


    plt.plot(fpr, tpr, color='darkorange', lw=2,label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(PLOTS_DIR /'roc_curve.png')
    plt.close()

    # # Distribution of Message Lengths (Fixed)
    # plt.figure(figsize=(10, 6))
    # # Convert sparse matrix to dense array for the test data
    # import pandas as pd
    # message_lengths = [len(str(msg).split()) for msg in X_test.toarray()]
    # df=pd.DataFrame({
    #     "length": message_lengths,
    #     "label": y_test,
    # })

    # df['label'] = df['label'].map({0: 'Ham', 1: 'Spam'})
    # sns.histplot(data=df,x='length' ,bins=50, hue="label", multiple="stack")
    # plt.title('Distribution of Message Lengths')
    # plt.xlabel('Number of Words')
    # plt.ylabel('Count')
    # plt.savefig('plots/message_length.png')
    # plt.close()

     # Message Length Distribution
    df_test['length'] = df_test['text'].str.split().str.len()
    df_test['label'] = df_test['label'].map({0: 'Ham', 1: 'Spam'})


    plt.figure(figsize=(10,6))
    sns.histplot(
        data=df_test,
        x='length',
        hue='label',
        bins=50,
        multiple='stack'
    )
    plt.title('Distribution of Message Lengths (Ham vs. Spam)')
    plt.xlabel('Number of Words')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR /'length_dist.png')
    plt.close()

except Exception as e:
    # print(f"Error occurred while generating plots: {str(e)}")
    traceback.print_exc()
except KeyboardInterrupt:
    print("Process interrupted by user.")
# User Input Testing
while True:
    raw = input("Enter a message to classify (or type 'exit' to quit): ")
    user_input = ' '.join(raw.strip().split()) 
    if not raw:
        print("Empty input. Please enter a message.")
        continue
    elif user_input.lower() == 'exit':
        break
    user_vec = vectorizer.transform([user_input]) # Transform the user input using the same vectorizer
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
