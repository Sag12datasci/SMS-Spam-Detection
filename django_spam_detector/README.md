# SMS Spam Detection Django Project

A minimal Django web application for detecting spam messages using machine learning.

## Project Structure
```
django_spam_detector/
├── manage.py                # Django management script
├── model.pkl               # Trained ML model
├── vectorizer.pkl          # Text vectorizer
├── requirements.txt        # Project dependencies
├── static/
│   └── css/
│       └── style.css      # Basic styling
├── templates/
│   ├── base.html         # Base template
│   └── detector/
│       └── home.html     # Main page
├── spam_detector/         # Project configuration
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── detector/             # Main application
    ├── __init__.py
    ├── forms.py
    ├── urls.py
    └── views.py
```

## Quick Setup

1. Create virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

2. Install minimal requirements:
```bash
pip install -r requirements.txt
```

3. Run migrations:
```bash
python manage.py migrate
```

4. Start server:
```bash
python manage.py runserver
```

Visit http://localhost:8000

## PythonAnywhere Deployment (Free Tier)

1. Create PythonAnywhere account and upload files
2. In Bash console:
```bash
cd ~
git clone <your-repo>
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Web app configuration:
   - Create new web app
   - Manual config -> Python 3.9
   - Set source code: /home/yourusername/django_spam_detector
   - Set working directory: /home/yourusername/django_spam_detector
   - Set virtual env: /home/yourusername/venv

4. WSGI configuration:
```python
import os
import sys
path = '/home/yourusername/django_spam_detector'
if path not in sys.path:
    sys.path.append(path)
os.environ['DJANGO_SETTINGS_MODULE'] = 'spam_detector.settings'
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
```

5. settings.py changes:
```python
DEBUG = False
ALLOWED_HOSTS = ['yourusername.pythonanywhere.com']
STATIC_ROOT = '/home/yourusername/django_spam_detector/static'
```

## Testing

Sample messages:
```
# Spam
CONGRATULATIONS! You've won $1000! Call now!

# Ham
Hey, can we meet at 3pm?
```

## Troubleshooting

- Server error: Check error logs in PythonAnywhere dashboard
- Static files not loading: Verify STATIC_ROOT and run collectstatic
- Model loading error: Check model.pkl path in views.py
