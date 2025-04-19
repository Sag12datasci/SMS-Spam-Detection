import pickle
from django.shortcuts import render
from .forms import MessageForm
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load models
model_path = os.path.join(BASE_DIR, 'model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'vectorizer.pkl')

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(vectorizer_path, 'rb') as file:
    vectorizer = pickle.load(file)

def home(request):
    form = MessageForm()
    result = None
    confidence = None
    message = None

    if request.method == 'POST':
        form = MessageForm(request.POST)
        if form.is_valid():
            message = form.cleaned_data['message']
            # Transform the message
            message_vec = vectorizer.transform([message])
            # Get prediction and probability
            prediction = model.predict(message_vec)[0]
            proba = model.predict_proba(message_vec)[0]
            result = 'SPAM' if prediction == 1 else 'HAM'
            confidence = f"{max(proba)*100:.2f}%"

    return render(request, 'detector/home.html', {
        'form': form,
        'result': result,
        'confidence': confidence,
        'message': message
    })
