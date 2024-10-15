from django.shortcuts import render
from django.http import JsonResponse
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pickle
from numpy import floor

loaded_model = pickle.load(open('review_classifier/models/log_res_model.sav', 'rb'))

def preprocessor(text):
    text = re.sub(r'<[^>]*>', '', text)
    emotes = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub(r'[\W]+', ' ', text.lower())) + ' '.join(emotes).replace('-', '')
    st_words = stopwords.words('english')
    text = ' '.join([w for w in text.split() if w not in st_words])
    return text

def index(request):
    return render(request, 'review_classifier/index.html')

def classify(request):
    if request.method == 'POST':
        review = preprocessor(request.POST.get('review_text', ''))
        pred = loaded_model.predict([review])
        proba = loaded_model.predict_proba([review])[0, 1]
        label = 'позитивное' if pred[0] else 'негативное'
        rating = floor(proba * 10) + 1
        return JsonResponse({'label': label, 'rating': rating})
    return render(request, 'review_classifier/index.html')